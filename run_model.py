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

STAGE1_SUFFIX = (
    "\n\nSolve this problem. Think step by step. "
    "On the final line write exactly: Answer: <number>"
)
STAGE2_COT_SUFFIX = (
    "\n\nThink step by step. On the final line write exactly: "
    "Answer: (X)  where X is A or B."
)
STAGE2_DIRECT_SUFFIX = (
    "\n\nAnswer with only the letter, formatted exactly as: Answer: (A) or Answer: (B)."
)

LETTER_PATTERNS = [
    re.compile(r"answer\s*[:\-]?\s*\(?\s*([AB])\s*\)?", re.I),
    re.compile(r"\bthe answer is\s*\(?\s*([AB])\s*\)?", re.I),
    re.compile(r"\(\s*([AB])\s*\)\s*$"),
    re.compile(r"^\s*([AB])\s*$", re.M),
]
NUM_AFTER_ANSWER = re.compile(r"answer\s*[:\-]?\s*\$?\s*(-?\d[\d,]*\.?\d*)", re.I)
ANY_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def parse_letter(text: str):
    tail = text.strip()[-400:]
    for rx in LETTER_PATTERNS:
        last = None
        for last in rx.finditer(tail):
            pass
        if last:
            return last.group(1).upper()
    return None


def parse_number(text: str):
    m = NUM_AFTER_ANSWER.search(text)
    if not m:
        nums = ANY_NUMBER.findall(text)
        if not nums:
            return None
        m_val = nums[-1]
    else:
        m_val = m.group(1)
    v = m_val.replace(",", "")
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def num_equal(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def build_prompt(row, mode):
    if row["kind"] == "stage1_open":
        return row["question"] + STAGE1_SUFFIX
    body = f"{row['question']}\n(A) {row['option_A']}\n(B) {row['option_B']}"
    return body + (STAGE2_DIRECT_SUFFIX if mode == "direct" else STAGE2_COT_SUFFIX)


def call(client, model, prompt, want_cot, max_retries=4):
    kw = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024 if want_cot else 24,
    )
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(**kw)
            return r.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def score(row, mode, text):
    rec = {
        "id": row["id"],
        "kind": row["kind"],
        "source_idx": row["source_idx"],
        "mode": mode,
        "model": None,  # filled by caller
        "completion": text,
        "gold_value": row.get("gold_value"),
    }
    if row["kind"] == "stage1_open":
        pred = parse_number(text)
        rec.update(pred_number=pred, parse_ok=pred is not None,
                   correct=num_equal(pred, row["gold_value"]))
    else:
        pred = parse_letter(text)
        rec.update(variant=row["variant"], gold_letter=row["gold_letter"],
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
    ap.add_argument("--modes", default="cot", help="stage2 modes, e.g. 'cot,direct'")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=os.environ.get("MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    client = OpenAI()
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        keep = {r["source_idx"] for r in rows[: args.limit * 2]}
        rows = [r for r in rows if r["source_idx"] in keep]
    stage2_modes = args.modes.split(",")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.open():
            d = json.loads(l)
            done.add((d["id"], d["mode"]))

    jobs = []
    for row in rows:
        modes = ["cot"] if row["kind"] == "stage1_open" else stage2_modes
        for m in modes:
            if (row["id"], m) not in done:
                jobs.append((row, m))
    print(f"{len(rows)} rows; {len(jobs)} calls to make ({len(done)} cached)")

    def work(job):
        row, mode = job
        want_cot = mode != "direct"
        text = call(client, args.model, build_prompt(row, mode), want_cot)
        rec = score(row, mode, text)
        rec["model"] = args.model
        return rec

    with out.open("a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            f.write(json.dumps(fut.result()) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
