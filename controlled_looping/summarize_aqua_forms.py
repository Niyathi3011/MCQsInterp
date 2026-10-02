"""
AQuA in three forms on the same 227 questions (sampled T=0.6, 4 seeds):
  open        no options
  mcq         the original 5 options (one correct)   -- from natural_looping
  minus_gold  the correct option removed (4 wrong options, no valid answer)

    python controlled_looping/summarize_aqua_forms.py   # -> .../aqua_forms/summary.md
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
from generate import rescore  # noqa: E402

B = os.path.join(HERE, "results", "r1-distill-qwen-7b", "sampled_t0.6_max12k", "aqua_forms")
NAT = os.path.join(ROOT, "natural_looping", "results", "r1-distill-qwen-7b", "sampled_t0.6_max12k")


def main():
    items = {f: {r["idx"]: r for r in map(json.loads, open(os.path.join(
        HERE, "data", "aqua_forms", f"{f}.jsonl")))} for f in ("open", "minus_gold")}
    keep = set(items["minus_gold"])
    nat_items = {r["idx"]: r for r in map(json.loads, open(os.path.join(
        ROOT, "natural_looping", "data", "aqua_test_254.jsonl")))}
    runs = collections.defaultdict(list)
    for s in range(4):
        for f in ("open", "minus_gold"):
            runs[f].append([rescore(r, items[f][r["idx"]]) for r in
                            map(json.loads, open(os.path.join(B, f"seed{s}", f"{f}.jsonl")))])
        runs["mcq"].append([rescore(r, nat_items[r["idx"]]) for r in
                            map(json.loads, open(os.path.join(NAT, f"seed{s}", "aqua.jsonl")))
                            if r["idx"] in keep])
    L = ["# AQuA in three forms (same 227 questions, sampled T=0.6, 4 seeds)", "",
         "| form | valid answer? | loop rate, mean [range] | accuracy, mean |", "|---|---|---|---|"]
    desc = {"open": ("open-ended (no options)", "yes"), "mcq": ("original 5 options", "yes"),
            "minus_gold": ("correct option removed (4 wrong)", "**no**")}
    for f in ("open", "mcq", "minus_gold"):
        lr = [np.mean([r["truncated"] for r in R]) for R in runs[f]]
        acc = ([np.mean([bool(r["correct"]) for r in R]) for R in runs[f]] if f != "minus_gold" else None)
        L.append(f"| {desc[f][0]} | {desc[f][1]} | **{100 * np.mean(lr):.1f}%** "
                 f"[{100 * min(lr):.1f}-{100 * max(lr):.1f}] | "
                 f"{'n/a' if acc is None else f'{100 * np.mean(acc):.1f}%'} |")
    fin = [r for R in runs["minus_gold"] for r in R if not r["truncated"]]
    picks = collections.Counter(r["pred"] or "none" for r in fin)
    L += ["", f"Correct option removed, runs that finished ({len(fin)}): pick one of the wrong "
          f"letters {100 * sum(v for k, v in picks.items() if k != 'none') / max(len(fin), 1):.0f}%, "
          f"no letter {100 * picks['none'] / max(len(fin), 1):.0f}%.", "",
          "Open-ended accuracy is scored with math-verify against the correct option's text, which",
          "is not always a clean expression (units, words), so it is a lower bound."]
    open(os.path.join(B, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
