"""
Does forcing the stop signal ever produce the GOLDEN (correct, unlisted)
answer -- not garbage, not a listed option, the actual right number -- in the
text generated after </think>?  And separately: does it go on to actually
COMMIT to a letter (A/B), the way the prompt's own "Answer: (X)" instruction
asks for -- and if so, is that letter consistent with having just stated the
golden value doesn't match either option?

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
import re

# same patterns run_model.py's parse_letter uses, copied in (not imported) so
# this stays a standalone text-processing script with no openai/model deps.
LETTER_PATTERNS = [
    re.compile(r"\\boxed\{\s*\\?(?:text|mathrm)?\{?\s*\(?\s*([AB])\s*\)?\s*\}?\s*\}", re.I),
    re.compile(r"answer\s*[:\-]?\s*(?:is\s*)?\(?\s*([AB])\s*\)?", re.I),
    re.compile(r"\bthe answer is\s*\(?\s*([AB])\s*\)?", re.I),
    re.compile(r"\boption\s*\(?\s*([AB])\s*\)?", re.I),
    re.compile(r"\(\s*([AB])\s*\)\s*\.?\s*$"),
    re.compile(r"^\s*\(?([AB])\)?\s*$", re.M),
]


def post_think_text(text):
    """Text after </think> -- empty if the trace never stopped."""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return None


def states_value(gold, tail):
    """Does `gold` appear as its own number in `tail` -- not glued to other
    digits?  Plain substring search false-positives on e.g. gold="-5" matching
    inside "-55", or "27" matching inside "127"/"270"; this requires no digit
    immediately before or after the match."""
    if not gold:
        return False
    pat = re.escape(gold)
    return re.search(rf"(?<!\d){pat}(?!\d)", tail) is not None


def parse_letter(tail):
    """Last A/B commitment in the tail, or None if it never gets that far --
    mirrors run_model.py's parse_letter so 'did it pick an option' is judged
    the same way the original MCQ scoring does."""
    t = tail.strip()[-600:]
    for rx in LETTER_PATTERNS:
        last = None
        for last in rx.finditer(t):
            pass
        if last:
            return last.group(1).upper()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="phase2/raw/e3b_transcripts_with_answer.jsonl")
    ap.add_argument("--pairs", default="phase2/loop_pairs_raw.jsonl")
    ap.add_argument("--condition", default="catch_rescue")
    ap.add_argument("--examples", type=int, default=0,
                    help="# of each bucket to print (0 = all, the default)")
    ap.add_argument("--full", action="store_true",
                    help="print the complete tail text, not truncated to 300 chars")
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
        letter = parse_letter(tail)
        item = (r, tail, letter)
        if states_value(g, tail):
            stated_gold.append(item)
        else:
            other.append(item)

    n = len(rows)
    print(f"{args.condition}: n={n}")
    print(f"  stopped, golden answer appears after </think> : {len(stated_gold):3d}  "
          f"({100*len(stated_gold)/n:.0f}%)")
    print(f"  stopped, something else (wrong/garbled/empty)  : {len(other):3d}  "
          f"({100*len(other)/n:.0f}%)")
    print(f"  never stopped (no </think> at all)              : {len(never_stopped):3d}  "
          f"({100*len(never_stopped)/n:.0f}%)")

    # Every parsed letter here is inherently a WRONG pick, by construction:
    # catch has no correct option, so committing to A or B at all -- whether
    # or not the golden value was also stated -- is a choice with no right
    # answer to have chosen correctly.
    stopped = stated_gold + other
    committed = [x for x in stopped if x[2] is not None]
    contradicts = [x for x in stated_gold if x[2] is not None]
    print(f"\n  of {len(stopped)} stopped: {len(committed)} go on to commit to a letter "
          f"(A or B) -- inherently a wrong pick, catch has no correct option")
    print(f"  of {len(stated_gold)} that stated the golden value: {len(contradicts)} STILL "
          f"pick a letter afterward (states the right answer isn't there, picks anyway)")

    def show(tag, items):
        limit = len(items) if args.examples == 0 else args.examples
        print(f"\n{'=' * 78}\n{tag}  ({min(limit, len(items))}/{len(items)} shown)\n{'=' * 78}")
        for r, tail, letter in items[:limit]:
            print(f"\n--- source_idx {r['source_idx']}  gold={gold.get(r['source_idx'])!r}  "
                  f"stop@{r['stop']}  picked_letter={letter} ---")
            body = tail.strip()
            print((body if args.full else body[:300]) or "(empty)")

    if stated_gold:
        show("STOPPED, GOLDEN ANSWER STATED", stated_gold)
    if other:
        show("STOPPED, GOLDEN ANSWER NOT FOUND", other)


if __name__ == "__main__":
    main()
