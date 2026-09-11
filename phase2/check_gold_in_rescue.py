"""
Does forcing the stop signal ever produce the GOLDEN (correct, unlisted)
answer -- not garbage, not a listed option, the actual right number -- in the
text generated after </think>?

Needs a transcripts file from a --post-think-tokens > 0 run (otherwise there's
no text after </think> to check at all). Joins it against the loop_pairs file
for each problem's gold_value.

    python phase2/check_gold_in_rescue.py \
        --transcripts phase2/raw/e3b_transcripts_with_answer.jsonl \
        --pairs phase2/loop_pairs_raw.jsonl

No model, no GPU -- pure text processing.
"""
import argparse
import json


def post_think_text(text):
    """Text after </think> -- empty if the trace never stopped."""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="phase2/raw/e3b_transcripts_with_answer.jsonl")
    ap.add_argument("--pairs", default="phase2/loop_pairs_raw.jsonl")
    ap.add_argument("--condition", default="catch_rescue")
    ap.add_argument("--examples", type=int, default=4, help="# of each bucket to print")
    args = ap.parse_args()

    gold = {}
    for line in open(args.pairs):
        r = json.loads(line)
        gold[r["source_idx"]] = str(r["catch"]["gold_value"])

    rows = [json.loads(l) for l in open(args.transcripts)]
    rows = [r for r in rows if r["condition"] == args.condition]

    never_stopped, stated_gold, other = [], [], []
    for r in rows:
        tail = post_think_text(r["text"])
        if tail is None:
            never_stopped.append(r)
            continue
        g = gold.get(r["source_idx"])
        if g and g in tail:
            stated_gold.append((r, tail))
        else:
            other.append((r, tail))

    n = len(rows)
    print(f"{args.condition}: n={n}")
    print(f"  stopped, golden answer appears after </think> : {len(stated_gold):3d}  "
          f"({100*len(stated_gold)/n:.0f}%)")
    print(f"  stopped, something else (wrong/garbled/empty)  : {len(other):3d}  "
          f"({100*len(other)/n:.0f}%)")
    print(f"  never stopped (no </think> at all)              : {len(never_stopped):3d}  "
          f"({100*len(never_stopped)/n:.0f}%)")

    def show(tag, items):
        print(f"\n{'=' * 78}\n{tag}\n{'=' * 78}")
        for r, tail in items[: args.examples]:
            print(f"\n--- source_idx {r['source_idx']}  gold={gold.get(r['source_idx'])!r}  "
                  f"stop@{r['stop']} ---")
            print(tail.strip()[:300] or "(empty)")

    if stated_gold:
        show(f"STOPPED, GOLDEN ANSWER STATED (up to {args.examples})", stated_gold)
    if other:
        show(f"STOPPED, GOLDEN ANSWER NOT FOUND (up to {args.examples})", other)


if __name__ == "__main__":
    main()
