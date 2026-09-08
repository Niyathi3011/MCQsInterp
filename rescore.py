"""
Re-derive pred / parse_ok / correct / picked_number from the stored `completion`
text in a results file, using the current parsers in run_model.py. No model calls.

Use after improving the parsers (e.g. adding \\boxed{} support for MATH):

  python rescore.py --raw results/raw_math500.jsonl
  python analyze.py --raw results/raw_math500.jsonl --out_csv results/paired_math500.csv

Writes in place (a .bak copy is kept).
"""
import argparse
import json
import re
import shutil

from run_model import parse_letter, parse_number, num_equal, letter_from_number

BARE_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def picked_number(opt):
    return bool(BARE_NUM.fullmatch((opt or "").strip()))


def rescore(d):
    t = d.get("completion", "") or ""
    if d["kind"] == "stage1_open":
        p = parse_number(t)
        d["pred_number"] = p
        d["parse_ok"] = p is not None
        d["correct"] = num_equal(p, d.get("gold_value"))
    else:
        p = parse_letter(t) or letter_from_number(
            t, d.get("option_A"), d.get("option_B"))
        gl = d.get("gold_letter")
        opt = d.get("option_A") if p == "A" else d.get("option_B") if p == "B" else None
        d["pred_letter"] = p
        d["parse_ok"] = p is not None
        d["correct"] = (p == gl) if (p and gl) else False
        d["picked_number"] = picked_number(opt) if p else None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.jsonl")
    ap.add_argument("--out", help="default: overwrite --raw (keeps a .bak)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.raw)]
    before = sum(bool(r.get("parse_ok")) for r in rows)
    rows = [rescore(r) for r in rows]
    after = sum(bool(r.get("parse_ok")) for r in rows)

    out = args.out or args.raw
    if out == args.raw:
        shutil.copyfile(args.raw, args.raw + ".bak")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    s2 = [r for r in rows if r["kind"] == "stage2_mcq"]
    print(f"rows={len(rows)}  parse_ok {before} -> {after}")
    print(f"stage2 accuracy now: {sum(r['correct'] for r in s2) / max(len(s2), 1):.3f}")
    print(f"wrote {out}" + ("  (.bak kept)" if out == args.raw else ""))


if __name__ == "__main__":
    main()
