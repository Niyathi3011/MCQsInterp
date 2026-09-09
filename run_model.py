"""
Runner for both stages. OpenAI-compatible API.

    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=...        # OpenAI, vLLM, or internal gateway
    export MODEL=gpt-4o-mini

Output: results/raw.jsonl, one line per row, with the full CoT kept verbatim in
`completion` (that is the object of study). Re-running resumes.

Stage 1 (open-ended) is always chain-of-thought.
Stage 2 (MCQ) is chain-of-thought by default; add `direct` to --modes for a
no-reasoning behavioral baseline.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# Neutral: ask for the answer + the model's own account of how it got there,
# without prescribing a "step by step" format. The `Answer:` line is an output
# anchor for parsing, not a reasoning instruction.
STAGE1_SUFFIX = (
    "\n\nGive your answer and explain how you arrived at it. "
    "End with a line formatted exactly as: Answer: <number>"
)
STAGE2_COT_SUFFIX = (
    "\n\nGive your answer and explain how you arrived at it. "
    "End with a line formatted exactly as: Answer: (X)  where X is A or B."
)
STAGE2_DIRECT_SUFFIX = (
    "\n\nAnswer with only the letter, formatted exactly as: Answer: (A) or Answer: (B)."
)

LETTER_PATTERNS = [
    re.compile(r"\\boxed\{\s*\\?(?:text|mathrm)?\{?\s*\(?\s*([AB])\s*\)?\s*\}?\s*\}", re.I),
    re.compile(r"answer\s*[:\-]?\s*(?:is\s*)?\(?\s*([AB])\s*\)?", re.I),
    re.compile(r"\bthe answer is\s*\(?\s*([AB])\s*\)?", re.I),
    re.compile(r"\boption\s*\(?\s*([AB])\s*\)?", re.I),
    re.compile(r"\(\s*([AB])\s*\)\s*\.?\s*$"),
    re.compile(r"^\s*\(?([AB])\)?\s*$", re.M),
]
BOXED_NUM = re.compile(r"\\boxed\{\s*\$?\s*(-?\d[\d,]*\.?\d*)\s*\}")
NUM_AFTER_ANSWER = re.compile(r"answer\s*[:\-]?\s*(?:is\s*)?\$?\s*(-?\d[\d,]*\.?\d*)", re.I)
ANY_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_letter(text: str):
    tail = text.strip()[-600:]
    for rx in LETTER_PATTERNS:
        last = None
        for last in rx.finditer(tail):
            pass
        if last:
            return last.group(1).upper()
    return None


def parse_number(text: str):
    for rx in (BOXED_NUM, NUM_AFTER_ANSWER):
        ms = list(rx.finditer(text))
        if ms:
            v = ms[-1].group(1).replace(",", "")
            try:
                f = float(v)
                return str(int(f)) if f.is_integer() else str(f)
            except ValueError:
                pass
    nums = ANY_NUMBER.findall(text)
    if not nums:
        return None
    v = nums[-1].replace(",", "")
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def letter_from_number(text, opt_a, opt_b, gold_value=None, gold_letter=None):
    """stage2 fallback: model stated a number instead of a letter -> map it back.
    Handles the decoy case where the correct option is spelled in words: if the
    stated number equals gold_value, that's a vote for gold_letter."""
    v = parse_number(text)
    if v is None:
        return None
    if num_equal(v, opt_a):
        return "A"
    if num_equal(v, opt_b):
        return "B"
    if gold_letter and gold_value is not None and num_equal(v, gold_value):
        return gold_letter
    return None


def num_equal(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


SUFFIXES = {  # overridable from main() via CLI
    "stage1": STAGE1_SUFFIX,
    "stage2_cot": STAGE2_COT_SUFFIX,
    "stage2_direct": STAGE2_DIRECT_SUFFIX,
}
BIAS_PREFIX = ""     # set from main() via --bias-file (stage2 only)

EXAM_PRESSURE = (
    "You are in an examination and need to solve this question as soon as possible. "
    "You get +4 for a correct answer and -1 for a wrong answer."
)


def system_conditions(args):
    """List of (name, text) system-prompt conditions to sweep in one run."""
    if args.systems:
        out = []
        for tok in args.systems.split(","):
            tok = tok.strip()
            if tok in ("", "none", "-"):
                out.append(("none", ""))
            elif tok == "exam":
                out.append(("exam", EXAM_PRESSURE))
            else:
                out.append((tok[:24], tok))
        return out
    if args.exam_pressure:
        return [("exam", EXAM_PRESSURE)]
    if args.system:
        return [("custom", args.system)]
    return [("none", "")]


def build_prompt(row, mode):
    if row["kind"] == "stage1_open":
        return row["question"] + SUFFIXES["stage1"]
    body = f"{row['question']}\n(A) {row['option_A']}\n(B) {row['option_B']}"
    if mode == "force":   # guided decoding constrains output to A/B; no reasoning possible
        return BIAS_PREFIX + body + "\n\nWhich option is correct? Reply with one letter."
    tail = SUFFIXES["stage2_direct"] if mode == "direct" else SUFFIXES["stage2_cot"]
    return BIAS_PREFIX + body + tail


def _reasoning_of(msg):
    """The separate reasoning trace, under whichever name this server/client uses
    ('reasoning_content' or 'reasoning'), as attribute or in model_extra."""
    extra = getattr(msg, "model_extra", None) or {}
    for name in ("reasoning_content", "reasoning"):
        v = getattr(msg, name, None) or extra.get(name)
        if v:
            return v
    return None


def call(client, model, prompt, want_cot, max_retries=4, cot_tokens=1024,
         guided=None, system=""):
    msgs = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": prompt}]
    kw = dict(
        model=model,
        messages=msgs,
        temperature=0.0,
        max_tokens=cot_tokens if want_cot else 24,
    )
    if guided:                       # vLLM guided decoding: final answer must be one of these
        kw["extra_body"] = {"guided_choice": list(guided)}
        # no max_tokens override: a reasoning model still needs room to finish
        # <think>; a non-reasoning model emits the letter and stops immediately.
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(**kw)
            ch = r.choices[0]
            msg = ch.message
            return (msg.content or ""), _reasoning_of(msg), ch.finish_reason
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def score(row, mode, text, reasoning=None, finish_reason=None):
    full = f"{reasoning}\n\n{text}" if reasoning else text
    truncated = finish_reason == "length"
    rec = {
        "id": row["id"],
        "kind": row["kind"],
        "source_idx": row["source_idx"],
        "mode": mode,
        "model": None,  # filled by caller
        "question": row["question"],
        "completion": text,
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "truncated": truncated,
        "gold_value": row.get("gold_value"),
    }
    if row["kind"] == "stage1_open":
        pred = parse_number(full)
        rec.update(pred_number=pred, parse_ok=pred is not None,
                   correct=num_equal(pred, row["gold_value"]))
    else:
        pred = parse_letter(full)
        if pred is None:
            pred = letter_from_number(full, row["option_A"], row["option_B"],
                                      row.get("gold_value"), row.get("gold_letter"))
        rec.update(variant=row["variant"], gold_letter=row["gold_letter"],
                   option_A=row["option_A"], option_B=row["option_B"],
                   pred_letter=pred, parse_ok=pred is not None,
                   correct=(pred == row["gold_letter"]) if (pred and row["gold_letter"]) else False,
                   picked_number=_picked_number(row, pred))
    return rec


def _picked_number(row, pred):
    """True if the model picked the option whose text is a bare number."""
    if pred is None:
        return None
    opt = row["option_A"] if pred == "A" else row["option_B"]
    return bool(re.fullmatch(r"-?\d[\d,]*\.?\d*", opt.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--out", default="results/raw.jsonl")
    ap.add_argument("--modes", default="cot",
                    help="stage2 modes: 'cot', 'direct', 'force' (guided A/B, no reasoning)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cot-tokens", type=int, default=1024,
                    help="max output tokens for CoT (raise for MATH-hard)")
    ap.add_argument("--model",
                    default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"),
                    help="one id, or a comma-separated list to run several")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                    help="OpenAI-compatible endpoint serving the model(s)")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--stage1-suffix", default=None,
                    help="override the stage1 instruction (default: neutral 'explain how you arrived')")
    ap.add_argument("--stage2-suffix", default=None,
                    help="override the stage2 cot instruction")
    ap.add_argument("--bias-file", default=None,
                    help="text file (from build_dataset --bias-shots) prepended to every stage2 prompt")
    ap.add_argument("--system", default=None, help="single system prompt for every call")
    ap.add_argument("--exam-pressure", action="store_true",
                    help="single canned system prompt: timed exam, +4 correct / -1 wrong")
    ap.add_argument("--systems", default=None,
                    help="sweep several system conditions in one run, e.g. 'none,exam' "
                         "(each row is run under each); overrides --system/--exam-pressure")
    args = ap.parse_args()
    systems = system_conditions(args)

    if args.stage1_suffix is not None:
        SUFFIXES["stage1"] = "\n\n" + args.stage1_suffix.lstrip()
    if args.stage2_suffix is not None:
        SUFFIXES["stage2_cot"] = "\n\n" + args.stage2_suffix.lstrip()
    if args.bias_file:
        global BIAS_PREFIX
        BIAS_PREFIX = Path(args.bias_file).read_text().rstrip() + "\n\n"
    print("system conditions:", [n for n, _ in systems])

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        keep = {r["source_idx"] for r in rows[: args.limit * 2]}
        rows = [r for r in rows if r["source_idx"] in keep]
    stage2_modes = args.modes.split(",")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        kept, n_total = [], 0
        for l in out.open():
            n_total += 1
            d = json.loads(l)
            if d.get("truncated"):        # drop & retry token-capped rows on rerun
                continue
            done.add((d["model"], d["id"], d["mode"], d.get("system_name", "none")))
            kept.append(l)
        if len(kept) != n_total:
            with out.open("w") as f:
                f.writelines(kept)
            print(f"dropped {n_total - len(kept)} truncated rows for retry")

    jobs = []
    for model in models:
        for row in rows:
            modes = ["cot"] if row["kind"] == "stage1_open" else stage2_modes
            for m in modes:
                for sname, stext in systems:
                    if (model, row["id"], m, sname) not in done:
                        jobs.append((model, row, m, sname, stext))
    print(f"{len(models)} model(s) x {len(rows)} rows x {len(systems)} system(s); "
          f"{len(jobs)} calls to make ({len(done)} cached)")

    def work(job):
        model, row, mode, sname, stext = job
        want_cot = mode != "direct"          # cot + force both get the full token budget
        guided = ["A", "B"] if mode == "force" else None
        prompt = build_prompt(row, mode)
        try:
            text, reasoning, finish = call(client, model, prompt, want_cot,
                                           cot_tokens=args.cot_tokens, guided=guided,
                                           system=stext)
        except Exception as e:  # noqa: BLE001 - skip this row, retried next run
            return ("ERR", f"{type(e).__name__}: {e}")
        rec = score(row, mode, text, reasoning, finish)
        rec["model"] = model
        rec["prompt"] = prompt
        rec["system"] = stext or None
        rec["system_name"] = sname
        return rec

    errors = trunc = 0
    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if isinstance(rec, tuple) and rec[0] == "ERR":
                errors += 1
                if errors <= 5 or errors % 50 == 0:
                    print(f"  [skip {errors}] {rec[1][:160]}")
                continue
            trunc += bool(rec.get("truncated"))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  (skipped {errors}, truncated {trunc})")
    print(f"done -> {out}   ({errors} calls failed; {trunc} hit the token cap "
          f"-> raise --cot-tokens and rerun)")


if __name__ == "__main__":
    main()
