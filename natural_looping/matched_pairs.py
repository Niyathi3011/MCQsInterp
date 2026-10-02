"""
Aggregate a multi-seed sampled run and extract MATCHED loop/finish pairs: the
same question (identical prompt) that looped in some seeds and finished in
others. These pairs are the natural-loop analogue of catch-vs-aligned for the
mechanism analysis (logit lens / attribution / patching).

Writes into the sampled results folder:
  summary.md             loop rate and accuracy per dataset, mean and range over
                         seeds, next to the greedy run
  matched_pairs.jsonl    one line per question: the looping run and a finished
                         run (a correct one if there is any), plus how many seeds looped

    python natural_looping/matched_pairs.py \
        natural_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k \
        --greedy natural_looping/results/r1-distill-qwen-7b/greedy_max12k
"""
import argparse
import collections
import glob
import json
import os
import re

import numpy as np

ORDER = ["gsm8k", "aqua", "logiqa", "math500"]
FORMAT = {"gsm8k": "open-ended", "aqua": "MCQ (5 options)", "logiqa": "MCQ (4 options)",
          "math500": "open-ended"}
MISMATCH = re.compile(
    r"not (among|one of|in|listed in) the (given |provided |answer )?(options|choices)"
    r"|none of the (given |provided )?(options|choices)|isn['’]?t (an|one of the|among the) "
    r"options?|not an option|doesn['’]?t match any|does not match any|not listed", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="sampled results folder containing seed*/ subfolders")
    ap.add_argument("--greedy", default=None, help="greedy results folder, for comparison")
    args = ap.parse_args()

    seeds = sorted(glob.glob(os.path.join(args.folder, "seed*")))
    runs = collections.defaultdict(dict)                   # dataset -> seed -> {id: row}
    for sd in seeds:
        s = os.path.basename(sd)
        for d in ORDER:
            f = os.path.join(sd, f"{d}.jsonl")
            if os.path.exists(f):
                runs[d][s] = {r["id"]: r for r in map(json.loads, open(f))}

    greedy = {}
    if args.greedy:
        for d in ORDER:
            f = os.path.join(args.greedy, f"{d}.jsonl")
            if os.path.exists(f):
                R = [json.loads(l) for l in open(f)]
                greedy[d] = (np.mean([r["truncated"] for r in R]), np.mean([r["correct"] for r in R]))

    L = [f"# Sampled results: `{args.folder}`", "",
         f"{len(seeds)} seeds, T=0.6, top-p 0.95. Loop = hit the token cap without finishing "
         "(no answer). Mean over seeds, [min-max].", "",
         "| dataset | format | n | loop rate | greedy loop | accuracy | greedy acc | "
         "loops in 0 seeds | in all seeds | **matched pairs** |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    pairs_out = open(os.path.join(args.folder, "matched_pairs.jsonl"), "w")
    for d in ORDER:
        if d not in runs:
            continue
        S = runs[d]
        k = len(S)
        ids = sorted(set.intersection(*[set(v) for v in S.values()]))
        lr = [np.mean([S[s][i]["truncated"] for i in ids]) for s in S]
        ac = [np.mean([S[s][i]["correct"] for i in ids]) for s in S]
        nloop = collections.Counter()
        npairs = 0
        for i in ids:
            rs = [S[s][i] for s in sorted(S)]
            loops = [r for r in rs if r["truncated"]]
            fins = [r for r in rs if not r["truncated"]]
            nloop[len(loops)] += 1
            if loops and fins:
                fin = next((r for r in fins if r["correct"]), fins[0])
                lp = loops[0]
                pairs_out.write(json.dumps(dict(
                    id=i, dataset=d, format=lp["format"], gold=lp["gold"],
                    prompt=lp["prompt"], n_seeds=k, n_loop=len(loops),
                    loop=dict(seed=lp["seed"], reasoning=lp["reasoning"],
                              mismatch_talk=bool(MISMATCH.search(lp["reasoning"]))),
                    finish=dict(seed=fin["seed"], reasoning=fin["reasoning"],
                                completion=fin["completion"], pred=fin["pred"],
                                correct=fin["correct"]),
                ), ensure_ascii=False) + "\n")
                npairs += 1
        g = greedy.get(d)
        L.append(f"| {d} | {FORMAT[d]} | {len(ids)} | {100 * np.mean(lr):.1f}% "
                 f"[{100 * min(lr):.1f}-{100 * max(lr):.1f}] | "
                 f"{'–' if g is None else f'{100 * g[0]:.1f}%'} | {100 * np.mean(ac):.1f}% "
                 f"[{100 * min(ac):.1f}-{100 * max(ac):.1f}] | "
                 f"{'–' if g is None else f'{100 * g[1]:.1f}%'} | {nloop[0]} | {nloop[k]} | "
                 f"**{npairs}** |")
    pairs_out.close()
    L += ["", "Matched pairs = questions that looped in at least one seed and finished in at "
          "least one other (identical prompt). Written to `matched_pairs.jsonl`."]
    open(os.path.join(args.folder, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
