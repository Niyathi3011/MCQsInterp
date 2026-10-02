"""
Collect NATURAL loops: traces that hit the token cap although a valid answer
exists -- open-ended (stage1), aligned / swap (gold vs unrelated sentence) and
numpair (gold vs a plausible wrong number).  CPU only, no model.

One line per looping trace, in the shape rq3_resolve.py --loops expects:
    {"source_idx", "variant", "loop": <slim piece: question, options, gold,
      reasoning, ans_off, ...>}

    python phase2/prep_solvable_loops.py \
        --raw results/raw_r1.jsonl --raw results/raw_numpair_r1.jsonl \
        --out phase2/solvable_loops.jsonl
"""
import argparse
import collections
import json

from prep_loop import slim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True)
    ap.add_argument("--variants", default="open,aligned,swap,numpair")
    ap.add_argument("--system", default="none")
    ap.add_argument("--out", default="phase2/solvable_loops.jsonl")
    args = ap.parse_args()
    want = set(args.variants.split(","))

    out, n = open(args.out, "w"), collections.Counter()
    for path in args.raw:
        for line in open(path):
            r = json.loads(line)
            if r.get("mode", "cot") != "cot" or (r.get("system_name") or "none") != args.system:
                continue
            v = "open" if r["kind"] == "stage1_open" else r.get("variant")
            if v not in want or not r.get("truncated"):
                continue
            r.setdefault("variant", v)
            piece = slim(r)
            piece["variant"] = v
            out.write(json.dumps(dict(source_idx=r["source_idx"], variant=v, loop=piece)) + "\n")
            n[v] += 1
    out.close()
    print(f"{sum(n.values())} looping traces with a valid answer -> {args.out}: {dict(n)}")


if __name__ == "__main__":
    main()
