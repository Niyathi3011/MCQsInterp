"""
Print detected commit points in context, for hand-checking the detector.
The marker  ⟦COMMIT⟧  sits where the analysis reads the looping run (the token
right before the backtrack phrase).

    python looping_mechanism/check_commit_points.py --per-group 4 --seed 1
"""
import argparse
import collections
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--before", type=int, default=260)
    ap.add_argument("--after", type=int, default=90)
    args = ap.parse_args()
    by = collections.defaultdict(list)
    for p in map(json.loads, open(os.path.join(HERE, "data", "pairs.jsonl"))):
        by[p["group"]].append(p)
    rng = random.Random(args.seed)
    k = 0
    for g, P in sorted(by.items()):
        for p in rng.sample(P, min(args.per_group, len(P))):
            k += 1
            t, o = p["loop"]["reasoning"], p["loop"]["commit_off"]
            ctx = (t[max(0, o - args.before):o] + " ⟦COMMIT⟧ " + t[o:o + args.after])
            print(f"#{k} [{g}] {p['id']}  gold={p['gold']}  committed={p['loop']['commit_value']}"
                  f" ({p['loop']['commit_how']})  at {100 * o / max(len(t), 1):.0f}% of trace")
            print("   …" + re.sub(r"\s+", " ", ctx) + "…\n")


if __name__ == "__main__":
    main()
