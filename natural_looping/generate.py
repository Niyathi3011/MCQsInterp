"""
Run a reasoning model over the natural_looping datasets through an
OpenAI-compatible vLLM server; store the full trace and score the answer.

One output row per problem:
  id, dataset, idx, format, gold, model, temperature, top_p, seed, max_tokens,
  prompt, reasoning (the <think> text), completion (text after </think>),
  finish_reason, truncated (hit max_tokens = never finished = "loop"),
  completion_tokens, pred (boxed answer), correct

Resumable: rows already in --out are skipped, INCLUDING truncated ones -- here
the truncation is the result, not a failure to retry.

    python natural_looping/generate.py --dataset gsm8k \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max-tokens 12288 \
        --out natural_looping/results/r1-distill-qwen-7b/greedy_max12k/gsm8k.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from run_model import _reasoning_of  # noqa: E402

DATA = {
    "gsm8k":   "gsm8k_test_500.jsonl",
    "aqua":    "aqua_test_254.jsonl",
    "logiqa":  "logiqa_test_500.jsonl",
    "math500": "math500_test_500.jsonl",
}


# --------------------------------------------------------------------- scoring
def last_boxed(text):
    """Content of the last \\boxed{...}, braces balanced; None if absent."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k].strip()
    return None


def letter_of_stated_number(text, row):
    """The model stated a number instead of a letter: map it to the option with
    that value (same fallback as run_model.py's letter_from_number)."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", (last_boxed(text) or "") or text[-200:])
    if not nums:
        return None
    for L, opt in zip(row["letters"], row["options"]):
        try:
            if abs(float(nums[-1]) - float(str(opt).replace(",", ""))) < 1e-6:
                return L
        except ValueError:
            continue
    return None


def extract(text, row):
    """The model's final answer from the text after </think>."""
    b = last_boxed(text)
    if row["format"] == "mcq":
        letters = "".join(row["letters"])
        if b:                                   # \boxed{B}, \boxed{\text{B}}, \boxed{(B)}
            m = re.search(rf"\b([{letters}])\b", b)
            if m:
                return m.group(1)
        m = re.findall(rf"(?:answer|option)\s*(?:is|:)?\s*\(?\**([{letters}])\b",
                       text[-400:], re.I)       # fallback: "the answer is (B)"
        if m:
            return m[-1].upper()
        return letter_of_stated_number(text, row)   # "Answer: 5" when option A is 5
    if b is not None:
        return b
    m = re.findall(r"answer\s*(?:is|:)\s*\$?([^\n$]+)", text, re.I)
    return m[-1].strip().rstrip(".") if m else None


def is_correct(pred, row):
    if row.get("gold") is None:        # catch: no correct option exists -> not applicable
        return None
    if pred is None:
        return False
    if row["format"] == "mcq":
        return pred == row["gold"]
    if row["dataset"] == "gsm8k":
        try:
            return abs(float(pred.replace(",", "").replace("$", "").rstrip(".")) -
                       float(row["gold"])) < 1e-6
        except ValueError:
            pass
    from math_verify import parse, verify
    try:
        # timeouts off: math-verify's default timeout uses signal.alarm, which raises
        # in worker threads -- and that error was being counted as "wrong"
        return bool(verify(parse(f"${row['gold']}$", parsing_timeout=None),
                           parse(f"\\boxed{{{pred}}}", parsing_timeout=None),
                           timeout_seconds=None))
    except Exception:  # noqa: BLE001 -- unparsable answer counts as wrong
        return False


def rescore(rec, item):
    """Re-derive pred / correct for a stored result row from its answer text, with
    the current parser (item = the data row: options, letters, gold)."""
    if rec["truncated"]:
        return dict(rec, pred=None, correct=None if item.get("gold") is None else False)
    pred = extract(rec["completion"] or "", item)
    return dict(rec, pred=pred, correct=is_correct(pred, item))


# --------------------------------------------------------------------- calling
def call(client, args, row):
    kw = dict(model=args.model, messages=[{"role": "user", "content": row["prompt"]}],
              temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    if args.seed is not None:
        kw["seed"] = args.seed
    for attempt in range(4):
        try:
            r = client.chat.completions.create(**kw)
            ch = r.choices[0]
            return dict(reasoning=_reasoning_of(ch.message) or "",
                        completion=ch.message.content or "",
                        finish_reason=ch.finish_reason,
                        completion_tokens=getattr(r.usage, "completion_tokens", None))
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
            last = e  # noqa: F841


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATA), help="one of the natural datasets")
    ap.add_argument("--data", default=None,
                    help="any row file in the same format instead (e.g. controlled_looping/data/...)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not (args.dataset or args.data):
        ap.error("give --dataset or --data")
    path = args.data or os.path.join(HERE, "data", DATA[args.dataset])
    rows = [json.loads(l) for l in open(path)]
    args.dataset = args.dataset or os.path.splitext(os.path.basename(path))[0]
    if args.limit:
        rows = rows[: args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out)}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.dataset}: {len(rows)} rows, {len(done)} already done, {len(todo)} to run"
          f"  | {args.model}  T={args.temperature} top_p={args.top_p} seed={args.seed}"
          f" max_tokens={args.max_tokens}")

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    n = trunc = corr = 0
    t0 = time.time()
    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(call, client, args, r): r for r in todo}
        for fut in as_completed(futs):
            row = futs[fut]
            try:
                out = fut.result()
            except Exception as e:  # noqa: BLE001 -- left for the next resume
                print(f"  ERROR {row['id']}: {type(e).__name__}: {e}")
                continue
            truncated = out["finish_reason"] == "length"
            pred = None if truncated else extract(out["completion"], row)
            rec = dict(id=row["id"], dataset=row["dataset"], idx=row["idx"],
                       format=row["format"], gold=row["gold"], model=args.model,
                       temperature=args.temperature, top_p=args.top_p, seed=args.seed,
                       max_tokens=args.max_tokens, prompt=row["prompt"], **out,
                       truncated=truncated, pred=pred, correct=is_correct(pred, row))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            trunc += truncated
            corr += bool(rec["correct"])
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  correct {corr}  hit cap (loop) {trunc}"
                      f"  {time.time() - t0:6.0f}s")
    print(f"done {args.dataset} -> {args.out}")


if __name__ == "__main__":
    main()
