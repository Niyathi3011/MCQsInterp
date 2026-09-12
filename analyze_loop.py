"""
Focused analysis for the R1 looping study.

  python analyze_loop.py [results.jsonl]     (default: ~/Desktop/raw_r1.jsonl)

Splits every stage-2 MCQ by whether a correct option exists:
  has-correct : aligned, swap, decoy   (gold_letter is 'A' or 'B')
  no-correct  : catch, broken          (gold_letter is null)

Reports, per system prompt (none / exam):
  1. open-ended (stage1) accuracy
  2. MCQ accuracy WHEN a correct option exists
  3. what the model does WHEN no correct option exists (pick rates, non-termination)
  4. the core contrast: non-termination rate  has-correct  vs  no-correct
  5. within-problem: same problem as aligned vs catch vs decoy
  6. repetition signature of the truncated no-correct traces
"""
import collections
import json
import os
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/raw_r1.jsonl")
HAS_CORRECT = {"aligned", "swap", "decoy"}
NO_CORRECT = {"catch", "broken"}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def body(r):
    """reasoning + answer text, up to the final 'Answer:' anchor."""
    t = (r.get("reasoning") or "") + "\n" + (r.get("completion") or "")
    return t.split("Answer:")[0]


def stated_gold(r):
    g = str(r.get("gold_value", "")).strip()
    return bool(g) and g in body(r)


def max_line_repeat(text):
    """longest run of (near-)identical consecutive sentences - a loop fingerprint."""
    parts = [re.sub(r"\s+", " ", s).strip().lower()
             for s in re.split(r"(?<=[.\n])", text) if len(s.strip()) > 15]
    best = run = 1
    for a, b in zip(parts, parts[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    # also: most common sentence and its count
    c = collections.Counter(parts)
    top = c.most_common(1)[0] if c else ("", 0)
    return best, top[1], len(parts)


def pct(x, n):
    return f"{100 * x / n:5.1f}%" if n else "   -  "


def main():
    rows = [json.loads(l) for l in open(PATH)]
    print(f"file: {PATH}")
    print(f"rows: {len(rows)}   problems: {len(set(r['source_idx'] for r in rows))}   "
          f"modes: {sorted(set(r['mode'] for r in rows))}\n")

    s1 = [r for r in rows if r["kind"] == "stage1_open"]
    s2 = [r for r in rows if r["kind"] == "stage2_mcq"]
    systems = sorted(set(r.get("system_name", "none") for r in rows))

    # ---- 1. open-ended ----
    print("=" * 78)
    print("1. OPEN-ENDED (stage1)")
    print(f"   {'system':6} {'n':>4} {'trunc':>7} {'acc(all)':>9} {'acc(finished)':>14}")
    for sy in systems:
        g = [r for r in s1 if r.get("system_name", "none") == sy]
        fin = [r for r in g if not r.get("truncated")]
        acc_all = sum(r["correct"] for r in g)
        acc_fin = sum(r["correct"] for r in fin)
        tr = sum(bool(r.get("truncated")) for r in g)
        print(f"   {sy:6} {len(g):4d} {pct(tr,len(g)):>7} "
              f"{pct(acc_all,len(g)):>9} {pct(acc_fin,len(fin)):>14}")

    # ---- 2. MCQ, correct option present ----
    print("\n" + "=" * 78)
    print("2. MCQ  --  a correct option EXISTS  (aligned / swap / decoy)")
    print(f"   {'variant':8} {'system':6} {'n':>4} {'trunc':>7} {'parseFail':>10} "
          f"{'acc(all)':>9} {'acc(finished)':>14}")
    for v in ("aligned", "swap", "decoy"):
        for sy in systems:
            g = [r for r in s2 if r.get("variant") == v and r.get("system_name", "none") == sy]
            if not g:
                continue
            fin = [r for r in g if not r.get("truncated")]
            tr = sum(bool(r.get("truncated")) for r in g)
            pf = sum(not r.get("parse_ok") for r in g)
            acc_all = sum(r["correct"] for r in g)
            acc_fin = sum(r["correct"] for r in fin)
            print(f"   {v:8} {sy:6} {len(g):4d} {pct(tr,len(g)):>7} {pct(pf,len(g)):>10} "
                  f"{pct(acc_all,len(g)):>9} {pct(acc_fin,len(fin)):>14}")

    # ---- 3. MCQ, NO correct option ----
    print("\n" + "=" * 78)
    print("3. MCQ  --  NO correct option  (catch / broken)")
    print(f"   {'variant':8} {'system':6} {'n':>4} {'NON-TERM':>9} {'pickA(num)':>11} "
          f"{'pickB':>7} {'no-answer':>10} {'CoT-had-true-ans':>17}")
    for v in ("catch", "broken"):
        for sy in systems:
            g = [r for r in s2 if r.get("variant") == v and r.get("system_name", "none") == sy]
            if not g:
                continue
            tr = sum(bool(r.get("truncated")) for r in g)
            pa = sum(r.get("pred_letter") == "A" for r in g)
            pb = sum(r.get("pred_letter") == "B" for r in g)
            na = sum(r.get("pred_letter") not in ("A", "B") for r in g)
            gh = sum(stated_gold(r) for r in g)
            print(f"   {v:8} {sy:6} {len(g):4d} {pct(tr,len(g)):>9} {pct(pa,len(g)):>11} "
                  f"{pct(pb,len(g)):>7} {pct(na,len(g)):>10} {pct(gh,len(g)):>17}")

    # ---- 4. the contrast ----
    print("\n" + "=" * 78)
    print("4. NON-TERMINATION RATE  --  the core contrast")
    for sy in systems:
        oe = [r for r in s1 if r.get("system_name", "none") == sy]
        hc = [r for r in s2 if r.get("variant") in HAS_CORRECT and r.get("system_name", "none") == sy]
        nc = [r for r in s2 if r.get("variant") in NO_CORRECT and r.get("system_name", "none") == sy]
        print(f"   [{sy}]  open-ended {pct(sum(bool(r.get('truncated')) for r in oe), len(oe))}"
              f"   MCQ w/ correct {pct(sum(bool(r.get('truncated')) for r in hc), len(hc))}"
              f"   MCQ NO correct {pct(sum(bool(r.get('truncated')) for r in nc), len(nc))}")

    # ---- 5. within-problem ----
    print("\n" + "=" * 78)
    print("5. WITHIN-PROBLEM  (system=none; same problem across variants)")
    by = collections.defaultdict(dict)
    for r in s2:
        if r.get("system_name", "none") == "none" and r["mode"] == "cot":
            by[r["source_idx"]][r.get("variant")] = r
    trip = [d for d in by.values() if {"aligned", "catch", "decoy"} <= set(d)]
    print(f"   problems with aligned+catch+decoy all present: {len(trip)}")
    if trip:
        a_ok = sum(not d["aligned"].get("truncated") and d["aligned"]["correct"] for d in trip)
        d_ok = sum(not d["decoy"].get("truncated") and d["decoy"]["correct"] for d in trip)
        c_loop = sum(bool(d["catch"].get("truncated")) for d in trip)
        c_loop_but_solvable = sum(
            bool(d["catch"].get("truncated"))
            and not d["aligned"].get("truncated") and d["aligned"]["correct"]
            for d in trip)
        print(f"   aligned solved & terminated           : {a_ok}/{len(trip)}")
        print(f"   decoy   solved & terminated           : {d_ok}/{len(trip)}")
        print(f"   catch   looped (truncated)            : {c_loop}/{len(trip)}")
        print(f"   catch looped ON A PROBLEM IT SOLVES   : {c_loop_but_solvable}/{len(trip)}"
              f"   <- clean loop-vs-terminate control set")

    # ---- 6. repetition signature ----
    print("\n" + "=" * 78)
    print("6. REPETITION in truncated NO-CORRECT traces (loop fingerprint)")
    tnc = [r for r in s2 if r.get("variant") in NO_CORRECT and r.get("truncated")]
    reps = [max_line_repeat(r.get("reasoning") or "") for r in tnc]
    if reps:
        consec = sorted(x[0] for x in reps)
        topfreq = sorted(x[1] for x in reps)
        print(f"   truncated no-correct traces: {len(tnc)}")
        print(f"   longest run of identical consecutive sentences: "
              f"median {consec[len(consec)//2]}, max {consec[-1]}")
        print(f"   most-repeated sentence's count:                 "
              f"median {topfreq[len(topfreq)//2]}, max {topfreq[-1]}")
        loopy = sum(1 for c, f, _ in reps if f >= 5)
        print(f"   traces with a sentence repeated >=5x:           {loopy}/{len(tnc)}  "
              f"({pct(loopy, len(tnc)).strip()})")


if __name__ == "__main__":
    main()
