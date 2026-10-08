"""
Allow-None results: the same items with vs without "If none of the options is correct,
put None within \\boxed{}." (T=0.6, the seeds run with allow-None; the baseline uses the
same seeds of the original runs).

Per set: loop rate before -> after; of all runs: answers None / picks a letter;
answerable sets: accuracy before -> after and false Nones.

    python looping_mechanism/allow_none_summary.py --model-tag r1-distill-qwen-7b
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
from generate import extract, is_correct, last_boxed  # noqa: E402

BASE = {"math_catch": "prompt_boxed/seed{}/catch", "gsm8k_catch": "gsm8k_ctrl/seed{}/catch",
        "aqua_minus_gold": "aqua_forms/seed{}/minus_gold", "logiqa_minus_gold": "logiqa_forms/seed{}/minus_gold",
        "math_aligned": "prompt_boxed/seed{}/aligned", "gsm8k_aligned": "gsm8k_ctrl/seed{}/aligned",
        "aqua_original": "natural:aqua", "logiqa_original": "natural:logiqa"}


def jl(p):
    return [json.loads(l) for l in open(p)]


def is_none(r):
    return not r["truncated"] and bool(re.search(r"none", last_boxed(r["completion"] or "") or "", re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="r1-distill-qwen-7b")
    a = ap.parse_args()
    C = os.path.join(ROOT, "controlled_looping", "results", a.model_tag, "sampled_t0.6_max12k")
    N = os.path.join(ROOT, "natural_looping", "results", a.model_tag, "sampled_t0.6_max12k")
    D = os.path.join(ROOT, "controlled_looping", "data", "allow_none")
    L = [f"# Allow-None ({a.model_tag}, T=0.6)", "",
         "Same items, plus: \"If none of the options is correct, put None within \\boxed{}.\" "
         "Baseline = the same seeds without that sentence.", "",
         "| set | valid option? | seeds | runs | loops: before -> allow-None | answers None | picks a letter | "
         "correct: before -> allow-None |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    for name, bp in BASE.items():
        items = {r["id"]: r for r in jl(os.path.join(D, f"{name}.jsonl"))}
        idx = {r["idx"] for r in items.values()}
        R, B = [], []
        for p in sorted(glob.glob(os.path.join(C, "allow_none", "seed*", f"{name}.jsonl"))):
            s = int(p.split("seed")[-1].split("/")[0])
            rr = jl(p)
            if len(rr) < len(items):
                continue                                   # partial set (still running): skip
            R += rr
            if bp.startswith("natural:"):
                B += [r for r in jl(os.path.join(N, f"seed{s}", f"{bp[8:]}.jsonl")) if r["idx"] in idx]
            else:
                B += jl(os.path.join(C, bp.format(s) + ".jsonl"))
        if not R:
            continue
        has = next(iter(items.values()))["gold"] is not None
        n = len(R)
        none = sum(is_none(r) for r in R)
        letter = sum(1 for r in R if not r["truncated"] and not is_none(r)
                     and extract(r["completion"] or "", items[r["id"]]))
        cor = sum(1 for r in R if not r["truncated"] and not is_none(r)
                  and is_correct(extract(r["completion"] or "", items[r["id"]]), items[r["id"]]))
        bl = sum(r["truncated"] for r in B) / len(B)
        bc = sum(bool(r["correct"]) for r in B) / len(B)
        seeds = sorted({r["seed"] for r in R})
        rows.append(dict(set=name, valid=has, seeds=seeds, runs=n, loops_before=bl, loops_after=(sum(r["truncated"] for r in R)) / n,
                         none=none / n, letter=letter / n, correct_before=bc if has else None,
                         correct_after=cor / n if has else None))
        L.append(f"| {name} | {'yes' if has else '**no**'} | {seeds} | {n} | {100 * bl:.0f}% -> "
                 f"**{100 * sum(r['truncated'] for r in R) / n:.0f}%** | {100 * none / n:.0f}% | {100 * letter / n:.0f}% | "
                 + (f"{100 * bc:.0f}% -> {100 * cor / n:.0f}% |" if has else "n/a |"))
    out = os.path.join(HERE, "results", a.model_tag, "allow_none")
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(os.path.join(out, "summary.json"), "w"), indent=1)
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
