"""
Build the RQ2 study set from a Phase-1 R1 results file.  CPU only, no model.

Per source problem (mode=cot, system_name=none):
    aligned : not truncated & correct   -> probe label 0  + terminated trace
    swap    : not truncated & correct   -> probe label 0  (position-balanced)
    catch   : truncated (looped)        -> probe label 1  + looping trace
A problem is kept only if all three survive.

Output: phase2/loop_pairs.jsonl -- one line per problem with the raw pieces and
character offsets INTO the `reasoning` string for:
    ans_off        first occurrence of str(gold_value)   (model states its answer)
    loop_onset_off first backtrack phrase                (catch only)

Chat-templating + tokenisation happen later in loop_probe.py, against the model
that will actually run, so token indices line up.

    python phase2/prep_loop.py --raw ~/Desktop/raw_r1.jsonl
"""
import argparse
import json
import os
import re

BACKTRACK = re.compile(
    r"(wait\b|hold on|but that['’]?s not|isn['’]?t (an|one of the)?\s*option"
    r"|not an option|none of (the|these)|maybe (the|i)\b|let me reconsider"
    r"|that['’]?s not right|does(n['’]?t| not) match|not among)",
    re.I,
)


def first_offset(text, needle):
    if needle is None:
        return None
    i = text.find(str(needle))
    return i if i >= 0 else None


def backtrack_onset(text):
    m = BACKTRACK.search(text or "")
    return m.start() if m else None


def find_row(rows, sidx, variant, want_trunc):
    for r in rows:
        if r.get("source_idx") != sidx or r.get("variant") != variant:
            continue
        if r.get("mode") != "cot" or r.get("system_name", "none") != "none":
            continue
        if bool(r.get("truncated")) != want_trunc:
            continue
        if not want_trunc and not r.get("correct"):
            continue
        return r
    return None


def slim(r):
    reasoning = r.get("reasoning") or ""
    return dict(
        id=r["id"],
        variant=r["variant"],
        truncated=bool(r.get("truncated")),
        question=r["question"],
        option_A=r.get("option_A"),
        option_B=r.get("option_B"),
        gold_value=r.get("gold_value"),
        gold_letter=r.get("gold_letter"),
        pred_letter=r.get("pred_letter"),
        correct=bool(r.get("correct")),
        reasoning=reasoning,
        completion=r.get("completion") or "",
        ans_off=first_offset(reasoning, r.get("gold_value")),
        loop_onset_off=backtrack_onset(reasoning) if r.get("truncated") else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.expanduser("~/Desktop/raw_r1.jsonl"))
    ap.add_argument("--out", default="phase2/loop_pairs.jsonl")
    ap.add_argument("--catch-window", type=int, default=800,
                    help="tokens past loop onset that loop_probe.py should keep")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.raw)]
    sidxs = sorted(set(r["source_idx"] for r in rows))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    kept = drop = 0
    with open(args.out, "w") as f:
        for s in sidxs:
            a = find_row(rows, s, "aligned", False)
            w = find_row(rows, s, "swap", False)
            c = find_row(rows, s, "catch", True)
            if not (a and w and c):
                drop += 1
                continue
            f.write(json.dumps(dict(
                source_idx=s, catch_window=args.catch_window,
                aligned=slim(a), swap=slim(w), catch=slim(c),
            )) + "\n")
            kept += 1

    print(f"{kept} problems written, {drop} dropped -> {args.out}")
    if kept:
        recs = [json.loads(l) for l in open(args.out)]
        no_ans = sum(1 for r in recs if r["catch"]["ans_off"] is None)
        no_lo = sum(1 for r in recs if r["catch"]["loop_onset_off"] is None)
        print(f"  catch traces missing gold-value marker : {no_ans}/{kept}")
        print(f"  catch traces missing loop-onset marker : {no_lo}/{kept}")
        print(f"  (rows with a missing marker are skipped for the affected positions)")


if __name__ == "__main__":
    main()
