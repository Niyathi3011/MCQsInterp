"""
Read and summarize an E3b --save-transcripts output -- turns the raw JSONL
into the actual coherence-check deliverable: every catch_rescue continuation
(short enough to read all of, not just a sample), matched before/after pairs
for the same problem, a look at what replaces </think> under aligned_lesion,
and simple aggregate stats per condition.

    python phase2/read_transcripts.py --transcripts phase2/raw/e3b_transcripts.jsonl
    python phase2/read_transcripts.py ... --pairs 5     # how many before/after examples
    python phase2/read_transcripts.py ... --condition catch_rescue --all   # dump one condition in full

No model, no GPU -- pure text processing on the file rq3_patch.py --save-transcripts wrote.
"""
import argparse
import collections
import json
import re

CONDS = ["catch_unpatched", "catch_rescue", "catch_control", "aligned_unpatched", "aligned_lesion"]


def load(path):
    by_cond = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_cond[r["condition"]][r["source_idx"]] = r
    return by_cond


def looks_complete(text):
    """Naive scan aid, not a verdict: does the continuation end on something
    that reads like a finished thought?  Flags candidates worth a closer read;
    doesn't replace actually reading them."""
    t = text.strip()
    if not t:
        return False
    tail = t[-40:]
    return bool(re.search(r'[.!?"”’]\s*$', t)) or "think" in tail.lower() or "option" in tail.lower()


def stats(rows_by_idx):
    rows = list(rows_by_idx.values())
    n = len(rows)
    if not n:
        return
    stopped = sum(1 for r in rows if r["stop"] is not None)
    lens = [len(r["text"]) for r in rows]
    complete = sum(1 for r in rows if looks_complete(r["text"]))
    print(f"    n={n}  stopped={stopped} ({100*stopped/n:.0f}%)  "
          f"mean chars={sum(lens)/n:.0f}  “looks complete”={complete}/{n}")


def show(tag, r, width=400):
    print(f"\n--- {tag}  (source_idx {r['source_idx']}, stop@{r['stop']}, "
          f"max_repeat={r['max_repeat']}) ---")
    print(r["text"][:width].strip() or "(empty)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="phase2/raw/e3b_transcripts.jsonl")
    ap.add_argument("--pairs", type=int, default=5, help="# before/after examples to print")
    ap.add_argument("--lesion", type=int, default=3, help="# aligned_lesion examples to print")
    ap.add_argument("--condition", default=None, help="dump every transcript for this one condition")
    ap.add_argument("--all", action="store_true", help="with --condition, print full text not a preview")
    ap.add_argument("--width", type=int, default=400, help="chars of each transcript to print")
    args = ap.parse_args()

    by_cond = load(args.transcripts)
    print(f"{args.transcripts}: {sum(len(v) for v in by_cond.values())} rows, "
          f"{len(set().union(*[set(v) for v in by_cond.values()]))} problems\n")

    print("=" * 78)
    print("AGGREGATE STATS BY CONDITION")
    print("=" * 78)
    for c in CONDS:
        if c in by_cond:
            print(f"  {c}:")
            stats(by_cond[c])

    if args.condition:
        print("\n" + "=" * 78)
        print(f"ALL TRANSCRIPTS -- {args.condition}")
        print("=" * 78)
        w = 10 ** 9 if args.all else args.width
        for idx, r in sorted(by_cond[args.condition].items()):
            show(args.condition, r, width=w)
        return

    print("\n" + "=" * 78)
    print(f"EVERY catch_rescue CONTINUATION (n={len(by_cond.get('catch_rescue', {}))}) "
          "-- the actual coherence-check material")
    print("=" * 78)
    for idx, r in sorted(by_cond.get("catch_rescue", {}).items()):
        flag = "" if looks_complete(r["text"]) else "  [[ WORTH A CLOSER READ ]]"
        show(f"catch_rescue{flag}", r, width=args.width)

    print("\n" + "=" * 78)
    print(f"BEFORE / AFTER -- same problem, catch_unpatched vs catch_rescue ({args.pairs} examples)")
    print("=" * 78)
    shared = sorted(set(by_cond.get("catch_unpatched", {})) & set(by_cond.get("catch_rescue", {})))
    for idx in shared[: args.pairs]:
        show("catch_unpatched (before)", by_cond["catch_unpatched"][idx], width=args.width)
        show("catch_rescue    (after)", by_cond["catch_rescue"][idx], width=args.width)

    print("\n" + "=" * 78)
    print(f"WHAT REPLACES </think> UNDER LESION -- aligned_lesion ({args.lesion} examples)")
    print("=" * 78)
    for idx, r in sorted(by_cond.get("aligned_lesion", {}).items())[: args.lesion]:
        show("aligned_lesion", r, width=args.width)


if __name__ == "__main__":
    main()
