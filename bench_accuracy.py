"""
Screen the model's plain open-ended accuracy on several math datasets, to see
which land in a useful difficulty band before running the full two-stage
experiment. Uses only the stage1 prompt ("solve, think step by step,
Answer: <number>") and scores the numeric answer.

  python bench_accuracy.py --datasets gsm8k,math500 --n 200 \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct

Registered names: gsm8k, gsm_hard, math500  (loaded from the HF cache).
Extra datasets by path (parquet / arrow / jsonl), columns auto-detected:
  --datasets "gsm8k,math500,aime=/workspace/data/aime_1983_2024.parquet"

MATH-500 difficulty filter:  --min-level 3
Per-item results -> results/bench_<name>.jsonl
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from build_dataset import DATASETS, load_rows
from run_model import STAGE1_SUFFIX, num_equal, parse_number

QCOLS = ["problem", "question", "Problem", "Question", "input", "prompt", "Question:"]
ACOLS = ["answer", "Answer", "solution", "Solution", "target", "gt_answer",
         "expected_answer", "final_answer", "gold"]
BOXED = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")


def load_any(spec, split_override=None):
    if "=" in spec:
        name, path = spec.split("=", 1)
        import pandas as pd
        if path.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(path)]
        elif path.endswith(".arrow"):
            import pyarrow as pa
            with pa.memory_map(path, "r") as s:
                try:
                    rows = pa.ipc.open_file(s).read_all().to_pandas().to_dict("records")
                except pa.lib.ArrowInvalid:
                    rows = pa.ipc.open_stream(s).read_all().to_pandas().to_dict("records")
        else:
            rows = pd.read_parquet(path).to_dict("records")
    else:
        name = spec
        rows = load_rows(name, split_override or DATASETS[name]["split"])
    keys = set(rows[0].keys()) if rows else set()
    qk = next((c for c in QCOLS if c in keys), None)
    ak = next((c for c in ACOLS if c in keys), None)
    return name, rows, qk, ak


def gold_from(raw):
    raw = str(raw)
    m = re.search(r"####\s*(.+)", raw)
    if m:
        raw = m.group(1)
    bs = BOXED.findall(raw)
    if bs:
        raw = bs[-1]
    raw = raw.strip().replace(",", "").replace("$", "").replace("\\!", "")
    raw = re.sub(r"^\\text\{(.*)\}$", r"\1", raw)
    raw = re.sub(r"\\[a-zA-Z]+", "", raw).replace("{", "").replace("}", "").strip()
    try:
        f = float(raw)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def level_of(row):
    return int(re.sub(r"\D", "", str(row.get("level", "") or "")) or 0)


def call(client, model, prompt, cot_tokens, retries=4):
    for a in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=cot_tokens,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        except Exception:  # noqa: BLE001
            if a == retries - 1:
                raise
            time.sleep(2 ** a)


def bench_one(spec, args, client):
    name, rows, qk, ak = load_any(spec, args.split)
    if not qk or not ak:
        print(f"[{name}] could not find question/answer columns in {list(rows[0].keys())[:8]}")
        return None
    items = []
    for i, row in enumerate(rows):
        if args.min_level and level_of(row) and level_of(row) < args.min_level:
            continue
        g = gold_from(row[ak])
        if g is None:
            continue
        items.append((i, str(row[qk]).strip(), g))
    import random
    random.Random(args.seed).shuffle(items)
    items = items[: args.n]

    out_path = f"results/bench_{name}.jsonl"
    recs, correct, parsed = [], 0, 0

    def work(it):
        idx, q, gold = it
        text = call(client, args.model, q + STAGE1_SUFFIX, args.cot_tokens)
        pred = parse_number(text)
        return dict(dataset=name, idx=idx, gold=gold, pred=pred,
                    parse_ok=pred is not None,
                    correct=num_equal(pred, gold),
                    gen_words=len(text.split()), completion=text)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(work, items):
            recs.append(r)
            correct += r["correct"]
            parsed += r["parse_ok"]
    import os
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    n = len(recs)
    return dict(dataset=name, n=n,
                accuracy=correct / n if n else float("nan"),
                parse_ok=parsed / n if n else float("nan"),
                mean_words=sum(r["gen_words"] for r in recs) / n if n else 0,
                numeric_kept=f"{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="gsm8k,math500")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-level", type=int, default=0, help="math500: keep level >= this")
    ap.add_argument("--split", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cot-tokens", type=int, default=2048)
    ap.add_argument("--stage1-suffix", default=None,
                    help="override the instruction appended to each problem")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()

    global STAGE1_SUFFIX
    if args.stage1_suffix is not None:
        STAGE1_SUFFIX = "\n\n" + args.stage1_suffix.lstrip()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    summary = []
    for spec in [s for s in args.datasets.split(",") if s.strip()]:
        print(f"\n=== {spec} ===")
        r = bench_one(spec.strip(), args, client)
        if r:
            summary.append(r)
            print(f"  n={r['n']}  accuracy={r['accuracy']:.3f}  "
                  f"parse_ok={r['parse_ok']:.3f}  mean_words={r['mean_words']:.0f}")

    print("\n" + "=" * 60)
    print(f"{'dataset':<16}{'n':>6}{'accuracy':>10}{'parse_ok':>10}{'words':>8}")
    for r in summary:
        print(f"{r['dataset']:<16}{r['n']:>6}{r['accuracy']:>10.3f}"
              f"{r['parse_ok']:>10.3f}{r['mean_words']:>8.0f}")
    print("\nUseful band for the two-stage experiment: accuracy ~0.35-0.75")
    print("(need enough 'solved open-ended' to measure unfaithful selection)")


if __name__ == "__main__":
    main()
