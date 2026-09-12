"""
Dump CoTs from results/raw.jsonl with filters, for hand-reading.

Examples:
  # catch trials where the model picked the wrong number AND its CoT computed the
  # true gold answer -- the "unfaithful selection" cases:
  python inspect_cots.py --variant catch --pred A --bucket gold --full --out catch_gold.txt
  less catch_gold.txt

  # 15 aligned CoTs, tail only:
  python inspect_cots.py --variant aligned --n 15

  # every catch pick-A case, bucketed summary only:
  python inspect_cots.py --variant catch --pred A --summary
"""
import argparse
import json
import re
import sys

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_num(text):
    """The last number stated in the reasoning body (before the 'Answer:' line)."""
    body = text.split("Answer:")[0]
    m = NUM.findall(body)
    return m[-1].replace(",", "") if m else None


def eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def bucket_of(r):
    v = last_num(r["completion"])
    if v is None:
        return "none"
    if eq(v, r.get("gold_value")):
        return "gold"
    if eq(v, r.get("option_A")):
        return "optionA"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.jsonl")
    ap.add_argument("--kind", help="stage1_open | stage2_mcq")
    ap.add_argument("--variant", help="aligned | swap | catch")
    ap.add_argument("--pred", help="A | B")
    ap.add_argument("--correct", choices=["0", "1"])
    ap.add_argument("--bucket", choices=["gold", "optionA", "other", "none"],
                    help="what the CoT actually computed (vs gold / vs option A)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--full", action="store_true", help="whole CoT, not just the tail")
    ap.add_argument("--summary", action="store_true", help="only print bucket counts")
    ap.add_argument("--out", help="write to this file instead of stdout")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.raw)]
    sel = []
    for r in rows:
        if args.kind and r["kind"] != args.kind:
            continue
        if args.variant and r.get("variant") != args.variant:
            continue
        if args.pred and r.get("pred_letter") != args.pred:
            continue
        if args.correct is not None and str(int(bool(r.get("correct")))) != args.correct:
            continue
        if args.bucket and bucket_of(r) != args.bucket:
            continue
        sel.append(r)

    out = open(args.out, "w") if args.out else sys.stdout

    if args.summary:
        from collections import Counter
        counts = Counter(bucket_of(r) for r in sel)
        print(f"# {len(sel)} matching rows", file=out)
        for k in ("gold", "optionA", "other", "none"):
            n = counts.get(k, 0)
            print(f"  computed_{k:8s} {n:4d}  {n / max(len(sel), 1):.1%}", file=out)
    else:
        print(f"# {len(sel)} matching rows (showing {min(args.n, len(sel))})\n", file=out)
        for r in sel[: args.n]:
            print("=" * 80, file=out)
            print(f"id={r['id']}  variant={r.get('variant')}  pred={r.get('pred_letter')}  "
                  f"correct={r.get('correct')}  computed~{last_num(r['completion'])}", file=out)
            print(f"gold={r.get('gold_value')}   (A)={r.get('option_A')}   "
                  f"(B)={(r.get('option_B') or '')[:80]}", file=out)
            print("-" * 80, file=out)
            c = r["completion"]
            print(c if args.full else c[-900:], file=out)
        print(file=out)

    if args.out:
        out.close()
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
