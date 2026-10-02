"""
Summarise the sampled controlled run: loop rate and accuracy per condition and
prompt (mean and range over seeds), next to the original greedy run
(results/raw_r1.jsonl, answerline prompt, system prompt 'none'), and what the
model does on `catch` when it finishes (picks A = the wrong number, picks B = the
unrelated sentence, or commits to neither).

    python controlled_looping/summarize.py   # -> results/.../sampled_t0.6_max12k/summary.md
"""
import collections
import glob
import json
import os

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(HERE, "results", "r1-distill-qwen-7b", "sampled_t0.6_max12k")
CONDS = ["open", "aligned", "swap", "catch"]
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
from generate import rescore  # noqa: E402

# every row is re-scored from its stored answer text with ONE parser, so greedy
# and sampled (and both prompts) are scored identically
ITEMS = {(st, c): {d["idx"]: d for d in map(json.loads, open(
    os.path.join(HERE, "data", st, f"{c}.jsonl")))}
    for st in ("boxed", "answerline") for c in CONDS}
DESC = {"open": "open-ended, no options", "aligned": "(A) correct, (B) sentence",
        "swap": "(A) sentence, (B) correct", "catch": "(A) wrong number, (B) sentence"}


def greedy():
    out = collections.defaultdict(list)
    for l in open(os.path.join(ROOT, "results", "raw_r1.jsonl")):
        r = json.loads(l)
        if r.get("mode") != "cot" or (r.get("system_name") or "none") != "none":
            continue
        c = "open" if r["kind"] == "stage1_open" else r.get("variant")
        if c in CONDS:          # greedy used the answerline prompt
            out[c].append(rescore(dict(truncated=r["truncated"], completion=r["completion"]),
                                  ITEMS[("answerline", c)][r["source_idx"]]))
    return out


def main():
    G = greedy()
    L = ["# Controlled looping, sampled (T=0.6, top-p 0.95, 4 seeds) vs the original greedy run",
         "", "Same 150 MATH-500 problems and items as the greedy catch experiment. Loop = hit "
         "the 12k-token cap without finishing (no answer). Sampled: mean over seeds [min-max].", ""]
    summ = {}
    for style in ("boxed", "answerline"):
        seeds = sorted(glob.glob(os.path.join(BASE, f"prompt_{style}", "seed*")))
        L += [f"## Prompt: `{style}`" + (" (identical to the greedy run)" if style == "answerline"
                                          else " (identical to natural_looping)"), "",
              "| condition | options | loop rate, sampled | loop rate, greedy | accuracy, sampled "
              "| accuracy, greedy |", "|---|---|---|---|---|---|"]
        for c in CONDS:
            runs = [[rescore(r, ITEMS[(style, c)][r["idx"]])
                     for r in map(json.loads, open(os.path.join(s, f"{c}.jsonl")))] for s in seeds]
            lr = [np.mean([r["truncated"] for r in R]) for R in runs]
            acc = ([np.mean([bool(r["correct"]) for r in R]) for R in runs]
                   if c != "catch" else None)
            g = G.get(c, [])
            glr = np.mean([r["truncated"] for r in g]) if g else None
            gacc = np.mean([bool(r["correct"]) for r in g]) if g and c != "catch" else None
            summ[f"{style}/{c}"] = dict(loop=float(np.mean(lr)), loop_min=float(min(lr)),
                                        loop_max=float(max(lr)),
                                        acc=None if acc is None else float(np.mean(acc)))
            L.append(f"| {c} | {DESC[c]} | **{100 * np.mean(lr):.1f}%** "
                     f"[{100 * min(lr):.1f}-{100 * max(lr):.1f}] | "
                     f"{'–' if glr is None else f'{100 * glr:.1f}%'} | "
                     f"{'n/a' if acc is None else f'{100 * np.mean(acc):.1f}%'} | "
                     f"{'n/a' if gacc is None else f'{100 * gacc:.1f}%'} |")
        # catch: what happens when it finishes
        fin = collections.Counter()
        n_fin = 0
        for s in seeds:
            for r in map(json.loads, open(os.path.join(s, "catch.jsonl"))):
                r = rescore(r, ITEMS[(style, "catch")][r["idx"]])
                if not r["truncated"]:
                    n_fin += 1
                    fin[r["pred"] or "neither"] += 1
        g = [r for r in G["catch"] if not r["truncated"]]
        gf = collections.Counter(r["pred"] or "neither" for r in g)
        L += ["", f"`catch` runs that finished ({n_fin} sampled, {len(g)} greedy):", "",
              "| pick | sampled | greedy |", "|---|---|---|"]
        for k, lab in (("A", "A = the wrong number"), ("B", "B = the unrelated sentence"),
                       ("neither", "no letter")):
            L.append(f"| {lab} | {100 * fin[k] / max(n_fin, 1):.0f}% | "
                     f"{100 * gf[k] / max(len(g), 1):.0f}% |")
        L.append("")
    open(os.path.join(BASE, "summary.md"), "w").write("\n".join(L) + "\n")
    json.dump(summ, open(os.path.join(BASE, "summary.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
