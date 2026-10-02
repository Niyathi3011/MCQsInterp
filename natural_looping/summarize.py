"""
Summarise one results folder (one model x one decoding setting) across the four
datasets: accuracy, loop rate (hit the token cap = never answered), and the
SOPHIA Fig. 2 view -- mean trace length of correct vs incorrect answers.

Writes summary.md and summary.json into the results folder.

    python natural_looping/summarize.py natural_looping/results/r1-distill-qwen-7b/greedy_max12k
"""
import json
import os
import re
import sys

import numpy as np

ORDER = ["gsm8k", "aqua", "logiqa", "math500"]
FORMAT = {"gsm8k": "open-ended", "aqua": "MCQ (5 options)", "logiqa": "MCQ (4 options)",
          "math500": "open-ended"}
PAPER_N = {"gsm8k": 500, "aqua": 254, "logiqa": 500, "math500": 500}


def tail_dup(text, k=60):
    s = [re.sub(r"\s+", " ", x).strip().lower()
         for x in re.split(r"(?<=[.\n])", text) if len(x.strip()) > 15][-k:]
    return 1 - len(set(s)) / len(s) if s else 0.0


def ci95(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n) if n else float("nan")


def main():
    folder = sys.argv[1]
    res = {}
    for d in ORDER:
        path = os.path.join(folder, f"{d}.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        n = len(rows)
        corr = [r for r in rows if r["correct"]]
        loop = [r for r in rows if r["truncated"]]
        wrong = [r for r in rows if not r["correct"] and not r["truncated"]]
        tok = lambda L: float(np.mean([r["completion_tokens"] for r in L])) if L else None  # noqa: E731
        res[d] = dict(
            n=n, expected_n=PAPER_N[d], format=FORMAT[d],
            accuracy=len(corr) / n, loop_rate=len(loop) / n,
            loop_ci95=ci95(len(loop) / n, n), wrong_answer_rate=len(wrong) / n,
            tokens_correct=tok(corr), tokens_incorrect=tok(wrong + loop),
            tokens_wrong_answer=tok(wrong), tokens_loop=tok(loop),
            loop_tail_dup=float(np.median([tail_dup(r["reasoning"]) for r in loop])) if loop else None,
            finished_tail_dup=float(np.median([tail_dup(r["reasoning"]) for r in corr + wrong]))
            if corr + wrong else None,
        )
    json.dump(res, open(os.path.join(folder, "summary.json"), "w"), indent=1)

    f = lambda x, fmt="{:.0f}": "–" if x is None else fmt.format(x)  # noqa: E731
    L = [f"# Results: `{folder}`", "",
         "Loop = the model hit the token cap without finishing its reasoning (no answer).",
         "", "| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |",
         "|---|---|---|---|---|---|"]
    for d, r in res.items():
        L.append(f"| {d} | {r['format']} | {r['n']}"
                 f"{'' if r['n'] == r['expected_n'] else ' (of ' + str(r['expected_n']) + ')'}"
                 f" | {100 * r['accuracy']:.1f}% | **{100 * r['loop_rate']:.1f}%** "
                 f"(±{100 * r['loop_ci95']:.1f}) | {100 * r['wrong_answer_rate']:.1f}% |")
    L += ["", "## Trace length (tokens) - SOPHIA Fig. 2 view", "",
          "| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |",
          "|---|---|---|---|---|"]
    for d, r in res.items():
        L.append(f"| {d} | {f(r['tokens_correct'])} | {f(r['tokens_incorrect'])} |"
                 f" {f(r['tokens_wrong_answer'])} | {f(r['tokens_loop'])} |")
    L += ["", "## Are loops repetitive?", "",
          "Median fraction of the last 60 sentences that repeat an earlier one.", "",
          "| dataset | looping traces | finished traces |", "|---|---|---|"]
    for d, r in res.items():
        L.append(f"| {d} | {f(r['loop_tail_dup'], '{:.2f}')} | {f(r['finished_tail_dup'], '{:.2f}')} |")
    open(os.path.join(folder, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
