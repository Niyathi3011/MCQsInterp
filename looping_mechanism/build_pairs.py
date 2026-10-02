"""
Matched loop/finish pairs for the stop-signal analysis, with a COMMIT point in
every looping trace.  One pair per problem (no problem counted twice).

Groups (all sampled T=0.6, boxed prompt, unless noted):
  ctrl_cross      catch LOOPED vs aligned FINISHED correctly, same problem, same
                  seed (the original catch-vs-aligned design, now sampled)
  ctrl_same       catch looped in one seed, finished in another (identical prompt)
  nat_aqua / nat_logiqa / nat_math500
                  natural: same question looped in one seed, finished in another
                  (natural_looping/.../matched_pairs.jsonl)
  greedy_ref      the ORIGINAL greedy catch-vs-aligned pairs (phase2/loop_pairs.jsonl,
                  answerline prompt) -- reference for "does the greedy result replicate"

Commit point = where the looping run states the answer it keeps returning to and
then starts doubting it: the first backtrack phrase ("Wait", "But hold on", ...)
whose preceding sentence CLAIMS that value (a conclusion cue + the value). The
value is the gold answer for catch problems when the model derives it, else the
value the looping run claims most often (its own answer). Stored as a character
offset into the looping run's reasoning; the analysis uses the token right before
it (the end of the committing sentence -- a boundary, like the token before
</think> in the finished run).

    python looping_mechanism/build_pairs.py     # -> looping_mechanism/data/pairs.jsonl
"""
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "phase2"))
from prep_loop import BACKTRACK  # noqa: E402

CTRL = os.path.join(ROOT, "controlled_looping", "results", "r1-distill-qwen-7b",
                    "sampled_t0.6_max12k", "prompt_boxed")
NAT = os.path.join(ROOT, "natural_looping", "results", "r1-distill-qwen-7b",
                   "sampled_t0.6_max12k", "matched_pairs.jsonl")
GREEDY = os.path.join(ROOT, "phase2", "loop_pairs.jsonl")

# weak cue: any stated result -- enough when we look for a KNOWN value (the gold)
CUE = re.compile(r"\b(so|therefore|thus|hence|answer|which is|we get|get|gives|equals|"
                 r"result|total|must be|should be)\b|=|\\boxed", re.I)
# strong cue: an actual conclusion -- required when the value is the model's own
STRONG = re.compile(r"\banswer\b|\\boxed|\bfinal\b|\boption\b|\bchoice\b|not an option|"
                    r"isn['’]?t (an|one of the|among the) options?|not among|"
                    r"\b(so|therefore|thus|hence),? the (value|result|sum|product|number|total|"
                    r"probability|area|length|answer|solution|remainder|minimum|maximum|ratio)\b|"
                    r"\bthe (solution|result) is\b|\bwe conclude\b", re.I)
VALUE = re.compile(r"\\frac\{[^{}]*\}\{[^{}]*\}|(?<![\w.^_{])-?\d+(?:\.\d+)?(?:/\d+)?(?![\w}^])")
LETTER = re.compile(r"(?:answer|option|choice)\s*(?:is|:|would be|should be)?\s*\(?([A-E])\)?\b"
                    r"|\\boxed\{\\?(?:text\{)?\(?([A-E])\)?\}?\}", re.I)
SENT_START = re.compile(r"(^|[.!?:)\]]\s*|\n\s*)$")


def norm(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except ValueError:
        return v.upper().replace(" ", "")


def claims(text, mcq, cue, value_at_end=False):
    """[(backtrack_char_offset, claimed_value)] for every backtrack phrase that
    STARTS a sentence and whose preceding sentence states a value with `cue`."""
    out = []
    for m in BACKTRACK.finditer(text):
        if not SENT_START.search(text[max(0, m.start() - 3):m.start()] if m.start() else ""):
            continue                                    # mid-sentence "... isn't an option"
        before = text[max(0, m.start() - 300):m.start()].rstrip()
        sent = re.split(r"(?<=[.!?\n])\s+", before)[-1] if before else ""
        if not sent or not cue.search(sent):
            continue
        if mcq:
            ls = [a or b for a, b in LETTER.findall(sent)]
            if ls:
                out.append((m.start(), ls[-1].upper()))
                continue
        vs = list(VALUE.finditer(sent))
        if vs and (not value_at_end or len(sent) - vs[-1].end() <= 20):
            out.append((m.start(), norm(vs[-1].group(0))))
    return out


def commit_point(text, gold=None, mcq=False, own_ok=True):
    """(char_offset, value, method) or None.  With a gold value, the first
    sentence-initial doubt right after the model derives it; otherwise (own_ok)
    the first doubt after a STRONG claim of the value it claims most often."""
    if gold is not None:
        g = norm(str(gold))
        for off, v in claims(text, mcq, CUE, value_at_end=True):   # "... so P(5) = 15."
            if v == g:
                return off, v, "gold"
    if not own_ok:
        return None
    cl = claims(text, mcq, STRONG)
    if not cl:
        return None
    cnt = collections.Counter(v for _, v in cl)
    top = max(cnt, key=lambda v: (cnt[v], -next(o for o, x in cl if x == v)))
    off = next(o for o, v in cl if v == top)
    return off, top, f"own_top(x{cnt[top]})"


def load_seeds(cond):
    out = {}
    for sd in sorted(glob.glob(os.path.join(CTRL, "seed*"))):
        out[os.path.basename(sd)] = {r["idx"]: r for r in
                                     map(json.loads, open(os.path.join(sd, f"{cond}.jsonl")))}
    return out


def piece(r, **kw):
    return dict(prompt=r["prompt"], reasoning=r["reasoning"],
                completion=r.get("completion") or "", **kw)


def main():
    rng = random.Random(0)
    pairs, stats = [], collections.Counter()
    catch, aligned, opn = load_seeds("catch"), load_seeds("aligned"), load_seeds("open")
    gold_value = {d["idx"]: d["gold_value"] for d in map(json.loads, open(os.path.join(
        ROOT, "controlled_looping", "data", "boxed", "catch.jsonl")))}
    seeds = sorted(catch)
    idxs = sorted(catch[seeds[0]])

    def add(group, pid, loop_r, fin_r, gold, mcq, meta):
        own_ok = not (group.startswith("ctrl") or group == "greedy_ref")   # catch: gold only
        cp = commit_point(loop_r["reasoning"], gold, mcq, own_ok)
        if cp is None:
            stats[(group, "no commit point")] += 1
            return False
        off, val, how = cp
        pairs.append(dict(group=group, id=pid, gold=gold, **meta,
                          loop=piece(loop_r, commit_off=off, commit_value=val, commit_how=how),
                          finish=piece(fin_r, correct=fin_r.get("correct"))))
        stats[(group, "pairs")] += 1
        return True

    for i in idxs:
        # ctrl_cross: first seed where catch looped, aligned finished correctly, open solved
        for s in seeds:
            c, a, o = catch[s][i], aligned[s][i], opn[s][i]
            if c["truncated"] and not a["truncated"] and a["correct"] and o["correct"]:
                if add("ctrl_cross", f"ctrl_{i}_{s}", c, a, gold_value[i], False,
                       dict(seed_loop=s, seed_finish=s)):
                    break
        # ctrl_same: catch looped in one seed, finished in another
        lo = [s for s in seeds if catch[s][i]["truncated"]]
        fi = [s for s in seeds if not catch[s][i]["truncated"]]
        if lo and fi:
            add("ctrl_same", f"ctrlsame_{i}", catch[lo[0]][i], catch[fi[0]][i],
                gold_value[i], False, dict(seed_loop=lo[0], seed_finish=fi[0]))

    for p in map(json.loads, open(NAT)):
        mcq = p["format"] == "mcq"
        g = f"nat_{p['dataset']}"
        lr = dict(prompt=p["prompt"], reasoning=p["loop"]["reasoning"])
        fr = dict(prompt=p["prompt"], reasoning=p["finish"]["reasoning"],
                  completion=p["finish"]["completion"], correct=p["finish"]["correct"])
        # natural: the model's OWN committed value (for open questions the gold is
        # tried first, as for catch; MCQ gold is a letter it may never "claim")
        add(g, p["id"], lr, fr, None if mcq else p["gold"], mcq,
            dict(seed_loop=p["loop"]["seed"], seed_finish=p["finish"]["seed"]))

    for rec in map(json.loads, open(GREEDY)):
        c, a = rec["catch"], rec["aligned"]
        tmpl = ("{q}\n(A) {a}\n(B) {b}\n\nEnd with a line formatted exactly as: "
                "Answer: (X)  where X is A or B.")
        lr = dict(prompt=tmpl.format(q=c["question"], a=c["option_A"], b=c["option_B"]),
                  reasoning=c["reasoning"])
        fr = dict(prompt=tmpl.format(q=a["question"], a=a["option_A"], b=a["option_B"]),
                  reasoning=a["reasoning"], completion=a["completion"], correct=a["correct"])
        add("greedy_ref", f"greedy_{rec['source_idx']}", lr, fr, c["gold_value"], False,
            dict(seed_loop="greedy", seed_finish="greedy"))

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "pairs.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    groups = sorted({g for g, _ in stats})
    print(f"{'group':12s} {'pairs':>6s} {'no commit point':>16s}   commit via gold / own value")
    for g in groups:
        how = collections.Counter(p["loop"]["commit_how"].split("(")[0] for p in pairs if p["group"] == g)
        print(f"{g:12s} {stats[(g, 'pairs')]:6d} {stats[(g, 'no commit point')]:16d}   {dict(how)}")
    print(f"\n{len(pairs)} pairs -> looping_mechanism/data/pairs.jsonl")


if __name__ == "__main__":
    main()
