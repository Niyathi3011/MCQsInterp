"""
Why do certain questions loop, and why only in some runs?  Two analyses on stored runs
(no generation), per model:

  A. question level (AQuA): what the model answers with NO options (aqua_forms/open)
     decides whether the multiple-choice version loops?  Questions are grouped by the
     model's own open-ended answer: matches the correct option / matches a wrong option /
     matches no option / never answered; loop rate of the 5-option version (all seeds).
  B. run level (matched pairs: same question looped in one seed, finished in another):
     does the FIRST answer the run states already fit?  AQuA: is the first stated number
     one of the options?  LogiQA: is the first stated letter the key?  MATH: is the
     first stated result the gold answer?  Loop run vs finished run.

    python looping_mechanism/why_loop.py --model-tag r1-distill-qwen-7b
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
sys.path.insert(0, HERE)
from bench_switch import open_claims  # noqa: E402
from build_pairs import STRONG, claims  # noqa: E402
from generate import is_correct  # noqa: E402

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def num(s):
    m = NUM.search((s or "").replace(",", ""))
    return float(m.group(0)) if m else None


def same(a, b):
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1, abs(b))


def option_hit(value, options):
    """index of the option whose number equals value, or None"""
    v = num(value)
    for i, o in enumerate(options):
        if same(v, num(o)):
            return i
    return None


def pct(k, n):
    return f"{100 * k / n:.0f}%" if n else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="r1-distill-qwen-7b")
    a = ap.parse_args()
    NAT = os.path.join(ROOT, "natural_looping", "results", a.model_tag, "sampled_t0.6_max12k")
    CTL = os.path.join(ROOT, "controlled_looping", "results", a.model_tag, "sampled_t0.6_max12k")
    out = os.path.join(HERE, "results", a.model_tag, "why_loop")
    os.makedirs(out, exist_ok=True)
    items = {}
    for f in ("aqua_test_254.jsonl", "logiqa_test_500.jsonl", "math500_test_500.jsonl"):
        for r in map(json.loads, open(os.path.join(ROOT, "natural_looping", "data", f))):
            items[r["id"]] = r
    L = [f"# Why do certain questions loop? ({a.model_tag}, T=0.6)", ""]

    # ---- A: AQuA, the model's own open-ended answer vs the options
    opn = collections.defaultdict(list)
    for p in glob.glob(os.path.join(CTL, "aqua_forms", "seed*", "open.jsonl")):
        for r in map(json.loads, open(p)):
            if not r["truncated"] and r.get("pred"):
                opn[f"aqua_{r['idx']}"].append(r["pred"])
    loops = collections.defaultdict(list)
    for p in glob.glob(os.path.join(NAT, "seed*", "aqua.jsonl")):
        for r in map(json.loads, open(p)):
            loops[r["id"]].append(r["truncated"])
    keep = {f"aqua_{json.loads(l)['idx']}" for l in open(os.path.join(ROOT, "controlled_looping", "data",
                                                                       "aqua_forms", "minus_gold.jsonl"))}
    groups = collections.defaultdict(list)
    rows = []
    for q in sorted(keep):
        it = items[q]
        preds = opn.get(q, [])
        if not preds:
            g = "never answers (open form always loops)"
        else:
            top = collections.Counter(preds).most_common(1)[0][0]
            hit = option_hit(top, it["options"])
            g = ("own answer = correct option" if hit is not None and it["letters"][hit] == it["gold"]
                 else "own answer = a wrong option" if hit is not None else "own answer matches NO option")
        groups[g].append(loops[q])
        rows.append(dict(id=q, group=g, open_preds=preds, mcq_loops=loops[q]))
    L += ["## A. AQuA: the model's own answer (no options shown) vs looping on the 5-option question", "",
          "| own open-ended answer (majority over 4 seeds) | questions | 5-option runs that loop | "
          "questions that loop in >=1 seed |", "|---|---|---|---|"]
    for g in ["own answer = correct option", "own answer = a wrong option", "own answer matches NO option",
              "never answers (open form always loops)"]:
        v = groups.get(g, [])
        runs = [x for q in v for x in q]
        L.append(f"| {g} | {len(v)} | {pct(sum(runs), len(runs))} ({sum(runs)}/{len(runs)}) | "
                 f"{pct(sum(any(q) for q in v), len(v))} |")
    with open(os.path.join(out, "aqua_own_answer.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- B: matched pairs, does the first stated answer fit?
    P = [json.loads(l) for l in open(os.path.join(NAT, "matched_pairs.jsonl"))]
    L += ["", "## B. Same question, looped in one seed and finished in another: does the first stated "
          "answer already fit?", "",
          "AQuA: first stated number is one of the options. LogiQA: first stated letter is the key. "
          "MATH: first stated result equals the gold answer. (Runs with no detectable claim excluded.)", "",
          "| dataset | pairs | first answer fits: loop run | first answer fits: finished run | "
          "pairs where only the finished run's first answer fits | only the loop run's |",
          "|---|---|---|---|---|---|"]
    prow = []
    for d in ("aqua", "logiqa", "math500"):
        fl, ff, only_f, only_l, n = 0, 0, 0, 0, 0
        for p in [x for x in P if x["dataset"] == d]:
            it = items[p["id"]]
            res = []
            for side in ("loop", "finish"):
                t = p[side]["reasoning"]
                if d == "aqua":
                    cl = claims(t, False, STRONG)
                    cl = [c for c in cl if not re.fullmatch(r"[A-E]", c[1])]
                    res.append(None if not cl else option_hit(cl[0][1], it["options"]) is not None)
                elif d == "logiqa":
                    cl = [c for c in claims(t, True, STRONG) if re.fullmatch(r"[A-D]", c[1])]
                    res.append(None if not cl else cl[0][1] == it["gold"])
                else:
                    cl = open_claims(t)
                    res.append(None if not cl else bool(is_correct(cl[0][1], it)))
            prow.append(dict(id=p["id"], dataset=d, loop_first_fits=res[0], finish_first_fits=res[1]))
            if None in res:
                continue
            n += 1
            fl += res[0]
            ff += res[1]
            only_f += res[1] and not res[0]
            only_l += res[0] and not res[1]
        L.append(f"| {d} | {n} | {pct(fl, n)} | {pct(ff, n)} | {only_f} | {only_l} |")
    with open(os.path.join(out, "pairs_first_answer.jsonl"), "w") as f:
        for r in prow:
            f.write(json.dumps(r) + "\n")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
