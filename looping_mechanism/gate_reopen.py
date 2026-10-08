"""
Does the stop gate reopen once the conflict can be resolved?

Same no-valid-option items, read at every place the model says its answer is not among
the options ("... isn't one of the options."), at the token ending that sentence:
  catch loops          normal prompt, the run loops                (conflict, no way out)
  allow-None, None     "you may answer None": the run answers None  (conflict resolved)
  allow-None, loops    "you may answer None": the run still loops
and, for runs that answer None, at the real stop (token before </think>).

Measured: logit(</think>), whether </think> is the top choice, and the direct
contribution of the model's stop MLPs (doubt_timecourse.read_points).

    LOOP_MODEL=Qwen/Qwen3-4B-Thinking-2507 python looping_mechanism/gate_reopen.py
"""
import argparse
import glob
import json
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
sys.path.insert(0, HERE)
from model_cfg import CFG, MODEL, TAG  # noqa: E402
import bench_switch  # noqa: E402
from generate import last_boxed  # noqa: E402

CONFLICT = re.compile(r"not (among|one of|in|listed in) the (given )?(options|choices|answer choices)|"
                      r"isn['’]?t (an|one of the|among the) (options?|choices?)|not an option|"
                      r"none of the (options|choices) (is|are|match)|doesn['’]?t match any (of the )?(options|choices)",
                      re.I)
SETS = {"math_catch": "prompt_boxed/seed{}/catch", "aqua_minus_gold": "aqua_forms/seed{}/minus_gold",
        "gsm8k_catch": "gsm8k_ctrl/seed{}/catch", "logiqa_minus_gold": "logiqa_forms/seed{}/minus_gold"}
OUT = os.path.join(HERE, "results", TAG, "gate_reopen")


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def conflict_ends(text, k):
    """char offsets of the punctuation ending the first k sentences that state the conflict"""
    out = []
    for m in CONFLICT.finditer(text):
        e = re.compile(r"[.!?\n]").search(text, m.end())
        if e and (not out or e.start() > out[-1]):
            out.append(e.start())
        if len(out) >= k:
            break
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=60)
    ap.add_argument("--points", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    C = os.path.join(ROOT, "controlled_looping", "results", TAG, "sampled_t0.6_max12k")
    groups = {"catch loops": [], "allow-None, answers None": [], "allow-None, still loops": []}
    for name, base in SETS.items():
        for p in glob.glob(os.path.join(C, "allow_none", "seed*", f"{name}.jsonl")):
            s = int(p.split("seed")[-1].split("/")[0])
            for r in jl(p):
                none = not r["truncated"] and re.search(r"none", last_boxed(r["completion"] or "") or "", re.I)
                g = "allow-None, still loops" if r["truncated"] else "allow-None, answers None" if none else None
                if g and conflict_ends(r["reasoning"], 1):
                    groups[g].append(dict(r, set=name))
            for r in jl(os.path.join(C, base.format(s) + ".jsonl")):
                if r["truncated"] and conflict_ends(r["reasoning"], 1):
                    groups["catch loops"].append(dict(r, set=name))
    rng = random.Random(0)
    for g in groups:
        rng.shuffle(groups[g])
        groups[g] = groups[g][:a.per_group]
        print(f"  {g}: {len(groups[g])} runs")

    bench_switch.MODEL = MODEL
    from doubt_timecourse import read_points
    from stop_signal import chat_prefix, encode, token_at
    model, tid, u = bench_switch.load_model()
    rows = []
    for g, runs in groups.items():
        for i, r in enumerate(runs, 1):
            pre = chat_prefix(model, r["prompt"])
            fin = g == "allow-None, answers None"
            toks, offs = encode(model, pre + r["reasoning"] + ("</think>" if fin else ""))
            ends = conflict_ends(r["reasoning"], a.points)
            pos = [token_at(offs, len(pre) + c) for c in ends]
            stop = token_at(offs, len(pre) + len(r["reasoning"])) - 1 if fin else None
            want = [q for q in pos if q is not None] + ([stop] if stop else [])
            res = read_points(model, toks, want, u, tid, CFG["stop"])
            rows.append(dict(group=g, set=r["set"], id=r["id"], seed=r["seed"],
                             conflict=[res[q] for q in pos if q in res],
                             stop=res.get(stop) if stop else None))
            if i % 20 == 0:
                print(f"  {g}: {i}/{len(runs)}")
    with open(os.path.join(OUT, "points.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    L = [f"# Does the stop gate reopen once the conflict can be resolved? ({TAG})", "",
         "Read at the end of each sentence where the model says its answer is not among the options "
         "(first 5 per run). stop-MLP push = direct contribution of the model's stop MLPs "
         f"{CFG['stop']} to logit(</think>).", "",
         "| group | runs | logit(</think>) at conflict statements: first / max over the run | "
         "</think> top-1 at any conflict statement | stop-MLP push (first) | at the real stop |",
         "|---|---|---|---|---|---|"]
    for g in groups:
        R = [r for r in rows if r["group"] == g and r["conflict"]]
        if not R:
            continue
        first = [r["conflict"][0]["logit"] for r in R]
        mx = [max(c["logit"] for c in r["conflict"]) for r in R]
        top = [any(c["top1"] for c in r["conflict"]) for r in R]
        push = [r["conflict"][0]["comp_dla"] for r in R]
        st = [r["stop"]["logit"] for r in R if r["stop"]]
        L.append(f"| {g} | {len(R)} | {np.mean(first):+.1f} / {np.mean(mx):+.1f} | {100 * np.mean(top):.0f}% | "
                 f"{np.mean(push):+.1f} | {f'{np.mean(st):+.1f}' if st else '-'} |")
    L += ["", "By set (max logit at conflict statements):", "", "| set | " + " | ".join(groups) + " |",
          "|---|" + "---|" * len(groups)]
    for s in SETS:
        cells = []
        for g in groups:
            R = [r for r in rows if r["group"] == g and r["set"] == s and r["conflict"]]
            cells.append(f"{np.mean([max(c['logit'] for c in r['conflict']) for r in R]):+.1f} (n={len(R)})" if R else "-")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
