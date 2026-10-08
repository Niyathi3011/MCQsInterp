"""
Per-benchmark stop switch: learn the stop-signal MLPs and their values from a
benchmark's OWN catch-vs-aligned pairs, then test them on that benchmark's natural
loops, which are held out (never used for learning).

One folder per benchmark and temperature holds everything (inputs, settings, outputs):
    looping_mechanism/results/r1-distill-qwen-7b/bench_switch/<bench>_<t0|t0.6>/

Stages (run in order):
  pairs    (CPU) held-out set = questions that loop naturally at this temperature
           (any seed).  Learning pairs = catch LOOPED vs aligned FINISHED correctly,
           same question and seed, on the other questions only.  Commit points as in
           build_pairs.py.  Also the replay test loops (one natural loop per held-out
           question) and the held-out rows for the live test.
             -> config.json heldout.json learn_pairs.jsonl test_loops.jsonl live_items.jsonl
  signal   (GPU) on the learning pairs: logit(</think>) at the finished run's stop vs
           the loop's commit point, per-MLP contribution to the gap, this benchmark's
           top-6 stop MLPs, and the donor (mean MLP output right before </think> in the
           finished aligned runs).  Patching check on the learning pairs.
             -> signal_pairs.jsonl components.json donor.pt
  replay   (GPU) Option A: each held-out natural loop is fed up to its commit point,
           then: base / force </think> / bias / switch (own MLPs + own donor) /
           switch_math (the MATH six + MATH donor) / ctrl (own donor on MLP 15);
           the answer is generated fresh and scored.          -> replay.jsonl
  live-gen (vLLM) Option B, step 1: the held-out questions generated FROM SCRATCH with
           new seeds (no recorded run reused); this is the "no intervention" outcome.
             -> live_runs/seed<s>.jsonl
  trigger  (CPU) the stop rule, using only text written so far: no </think> after
           --trigger-tokens thinking tokens, then the first sentence-initial doubt after
           a stated answer (STRONG cue; no answer key), else the first paragraph break
           1000 tokens later.  Runs that finish first are untouched.  -> live_triggers.jsonl
  live     (GPU) Option B, step 2: every triggered run is continued from its trigger
           point under force / bias / switch / switch_math / ctrl.  -> live_branches.jsonl
  summary  -> summary.md

    python looping_mechanism/bench_switch.py --bench aqua --temp 0.6 --stage pairs
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
sys.path.insert(0, HERE)            # first: looping_mechanism/stop_signal.py, not natural_looping's
from build_pairs import CUE, STRONG, claims, commit_point, norm  # noqa: E402
from generate import DATA, extract, is_correct  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TAG = "r1-distill-qwen-7b"
NATR = os.path.join(ROOT, "natural_looping", "results", TAG)
CTLR = os.path.join(ROOT, "controlled_looping", "results", TAG)
CTLD = os.path.join(ROOT, "controlled_looping", "data")
MATH_SIX = [27, 26, 25, 24, 22, 20]
CONTROL = [15]
# per model: results tag, its MATH switch (stop MLPs from the MATH catch-vs-aligned pairs),
# control MLP, and the pairs file its MATH donor comes from
MODELS = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
          dict(tag="r1-distill-qwen-7b", ref=[27, 26, 25, 24, 22, 20], control=[15], ref_pairs="pairs.jsonl"),
          "Qwen/Qwen3-4B-Thinking-2507":
          dict(tag="qwen3-4b-thinking-2507", ref=[28, 31, 32, 33, 34, 35], control=[18],
               ref_pairs="pairs_qwen3-4b-thinking-2507.jsonl")}


def tag_of(model):
    return MODELS[model]["tag"]
NAT = {"gsm8k": "gsm8k", "aqua": "aqua", "logiqa": "logiqa", "math": "math500"}
# (catch file, aligned file or None = the natural run of the same question)
FORMS = {"gsm8k": ("gsm8k_ctrl/{}catch.jsonl", "gsm8k_ctrl/{}aligned.jsonl"),
         "aqua": ("aqua_forms/{}minus_gold.jsonl", None),
         "logiqa": ("logiqa_forms/{}minus_gold.jsonl", None),
         "math": ("prompt_boxed/{}catch.jsonl", "prompt_boxed/{}aligned.jsonl")}
CATCH_DATA = {"gsm8k": "gsm8k_ctrl/catch.jsonl", "aqua": "aqua_forms/minus_gold.jsonl",
              "logiqa": "logiqa_forms/minus_gold.jsonl", "math": "boxed/catch.jsonl"}


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def wjl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def seeds_of(temp):
    return ["greedy"] if temp == 0 else [0, 1, 2, 3]


def natural_runs(bench, temp):
    """{seed: {natural id: run}}"""
    d = NAT[bench]
    if temp == 0:
        return {"greedy": {r["id"]: r for r in jl(os.path.join(NATR, "greedy_max12k", f"{d}.jsonl"))}}
    return {s: {r["id"]: r for r in jl(os.path.join(NATR, "sampled_t0.6_max12k", f"seed{s}", f"{d}.jsonl"))}
            for s in seeds_of(temp)}


def form_runs(bench, temp, which):
    """{seed: {natural id: run}} for the catch (which=0) or aligned (1) form"""
    pat = FORMS[bench][which]
    if pat is None:
        return natural_runs(bench, temp)
    nid = nid_of(bench)
    out = {}
    for s in seeds_of(temp):
        if temp == 0:
            p = os.path.join(CTLR, "greedy_max12k", pat.format(""))
        else:
            sub, name = pat.split("/{}")
            p = os.path.join(CTLR, "sampled_t0.6_max12k", sub, f"seed{s}", name)
            if bench == "math":
                p = os.path.join(CTLR, "sampled_t0.6_max12k", "prompt_boxed", f"seed{s}", name)
        out[s] = {nid[r["idx"]]: r for r in jl(p)}
    return out


def nid_of(bench):
    """controlled row idx -> natural question id"""
    d = NAT[bench]
    if bench != "math":
        return collections.defaultdict(lambda: None, {i: f"{d}_{i}" for i in range(2000)})
    q2id = {r["question"].strip(): r["id"] for r in jl(os.path.join(ROOT, "natural_looping", "data", DATA[d]))}
    return {r["idx"]: q2id[r["question"].strip()] for r in jl(os.path.join(CTLD, "boxed", "catch.jsonl"))}


def items_of(bench):
    return {r["id"]: r for r in jl(os.path.join(ROOT, "natural_looping", "data", DATA[NAT[bench]]))}


def first_number(s):
    m = re.search(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return m.group(0) if m else None


# open-ended stated result: a conclusion word, the value at the end of the sentence, and
# not a condition / check / given value ("if", "check", "given", "<", "between", "?")
OPEN_CUE = re.compile(r"\b(so|therefore|thus|hence|answer|final|result|total|altogether|which is|"
                      r"equals|we get|get|gives)\b|=", re.I)
NOT_RESULT = re.compile(r"\b(if|check|given|between|less than|greater than|at least|at most|suppose|"
                        r"assume|whether|maybe|perhaps)\b|[<>?≤≥]", re.I)


def open_claims(text):
    """[(doubt offset, value)] where the preceding sentence states a result (open-ended)"""
    out = []
    for o, v in claims(text, False, OPEN_CUE, value_at_end=True):
        before = text[max(0, o - 300):o].rstrip()
        sent = re.split(r"(?<=[.!?\n])\s+", before)[-1] if before else ""
        if not NOT_RESULT.search(sent):
            out.append((o, v))
    return out


def open_commit(text):
    """key-free commit point for an open-ended loop: the first sentence-initial doubt
    after a sentence ending in the value the loop states most often"""
    cl = open_claims(text)
    if not cl:
        return None
    cnt = collections.Counter(v for _, v in cl)
    top = max(cnt, key=lambda v: (cnt[v], -next(o for o, x in cl if x == v)))
    return next(o for o, v in cl if v == top), top, f"own_end(x{cnt[top]})"


# --------------------------------------------------------------------- stage: pairs
def stage_pairs(a, out):
    nat, items = natural_runs(a.bench, a.temp), items_of(a.bench)
    catch, aligned = form_runs(a.bench, a.temp, 0), form_runs(a.bench, a.temp, 1)
    cdata = {r["idx"]: r for r in jl(os.path.join(CTLD, CATCH_DATA[a.bench]))}
    nid = nid_of(a.bench)
    mcq = items[next(iter(items))]["format"] == "mcq"
    held = sorted({i for s in nat for i, r in nat[s].items() if r["truncated"]})
    stats = collections.Counter()

    # learning pairs: catch looped vs aligned finished correctly, held-out questions excluded
    learn = []
    for idx, crow in sorted(cdata.items()):
        q = nid[idx]
        if q in held:
            stats["catch questions skipped (held out)"] += 1
            continue
        for s in seeds_of(a.temp):
            c, al = catch[s].get(q), aligned[s].get(q)
            if not (c and al and c["truncated"] and not al["truncated"] and al["correct"]):
                continue
            if a.bench in ("gsm8k", "math"):        # the model derives the gold number
                cp = commit_point(c["reasoning"], crow["gold_value"], False, own_ok=False)
            else:                                   # removed option: its number, else own letter
                g = first_number(crow.get("removed_gold"))
                cp = (commit_point(c["reasoning"], g, False, own_ok=False) if g else None) \
                    or commit_point(c["reasoning"], None, True, own_ok=True)
            if cp is None:
                stats["learning: no commit point"] += 1
                continue
            off, val, how = cp
            learn.append(dict(id=f"{q}_s{s}", question=q, seed=s,
                              loop=dict(prompt=c["prompt"], reasoning=c["reasoning"], commit_off=off,
                                        commit_value=val, commit_how=how),
                              finish=dict(prompt=al["prompt"], reasoning=al["reasoning"],
                                          completion=al.get("completion") or "", correct=al["correct"])))
            break

    # replay / live test: one natural loop per held-out question (first looping seed);
    # --max-heldout tests a random subset (all held-out questions stay out of learning)
    tq = sorted(random.Random(0).sample(held, a.max_heldout)) if 0 < a.max_heldout < len(held) else held
    test = []
    for q in tq:
        s = next(s for s in seeds_of(a.temp) if q in nat[s] and nat[s][q]["truncated"])
        r, it = nat[s][q], items[q]
        # no answer key: MCQ -> the letter/value it claims most often (build_pairs rule);
        # open-ended -> the value its sentences most often END in, at its first such claim
        cp = commit_point(r["reasoning"], None, True, own_ok=True) if mcq else open_commit(r["reasoning"])
        if cp is None:
            stats["test loops: no commit point"] += 1
        test.append(dict(id=f"{q}_s{s}", question=q, seed=s, prompt=r["prompt"], reasoning=r["reasoning"],
                         commit_off=cp[0] if cp else None, commit_value=cp[1] if cp else None,
                         commit_how=cp[2] if cp else None,
                         n_seeds_looped=sum(nat[x][q]["truncated"] for x in nat if q in nat[x])))
    wjl(os.path.join(out, "learn_pairs.jsonl"), learn)
    wjl(os.path.join(out, "test_loops.jsonl"), test)
    wjl(os.path.join(out, "live_items.jsonl"), [items[q] for q in tq])
    json.dump(dict(heldout=held), open(os.path.join(out, "heldout.json"), "w"), indent=1)
    stats.update({"questions": len(items), "held-out questions (loop naturally)": len(held),
                  "held-out questions tested": len(tq),
                  "learning pairs": len(learn), "replay test loops": len(test),
                  "replay test loops with a commit point": sum(t["commit_off"] is not None for t in test)})
    json.dump(dict(stats), open(os.path.join(out, "pairs_stats.json"), "w"), indent=1)
    for k, v in stats.items():
        print(f"  {k}: {v}")


# --------------------------------------------------------------------- GPU helpers
def load_model():
    import loop_probe
    from loop_probe import get_model, think_dir
    import rq3_resolve
    rq3_resolve.prefill.__defaults__ = (256,)
    loop_probe.RAW_THINK = False
    model = get_model(MODEL, "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tid, u = think_dir(model)
    return model, tid, u.float()


def math_ref_donor(model, tid, u, bench_dir):
    """the MATH donor used so far (sampled catch-vs-aligned finished runs, all 128
    ctrl_cross pairs), cached once for all benchmarks"""
    import torch
    from stop_signal import measure, positions
    p = os.path.join(bench_dir, "math_ref_donor.pt")
    if os.path.exists(p):
        return torch.load(p)
    vecs = []
    for q in [x for x in jl(os.path.join(HERE, "data", MODELS[MODEL]["ref_pairs"])) if x["group"] == "ctrl_cross"]:
        pos = positions(model, q, tid)
        if pos:
            vecs.append(measure(model, pos[0], pos[1], u, tid)[1])
    d = {L: torch.stack([v[L] for v in vecs]).mean(0).cpu() for L in range(model.cfg.n_layers)}
    torch.save(dict(donor=d, n=len(vecs)), p)
    print(f"  MATH reference donor: mean over {len(vecs)} finished runs -> {p}")
    return dict(donor=d, n=len(vecs))


# --------------------------------------------------------------------- stage: signal
def stage_signal(a, out):
    import torch
    from stop_signal import measure, positions
    model, tid, u = load_model()
    nL = model.cfg.n_layers
    P = jl(os.path.join(out, "learn_pairs.jsonl"))
    rows, fvecs, t0 = [], [], time.time()
    for i, p in enumerate(P, 1):
        pos = positions(model, p, tid)
        if pos is None:
            continue
        of, fv = measure(model, pos[0], pos[1], u, tid)
        ol, _ = measure(model, pos[2], pos[3], u, tid)
        fvecs.append(fv)
        rows.append(dict(id=p["id"], finish_logit=of["logit"], loop_logit=ol["logit"],
                         finish_top1=of["top1"], loop_top1=ol["top1"], loop_next=ol["next_tok"],
                         mlp_gap=(of["mlp"] - ol["mlp"]).tolist(), attn_gap=(of["attn"] - ol["attn"]).tolist()))
        if i % 10 == 0 or i == len(P):
            print(f"  signal {i}/{len(P)}  {time.time() - t0:5.0f}s")
    ref = math_ref_donor(model, tid, u, os.path.dirname(out))["donor"]
    if len(rows) >= 10:
        gap = np.array([r["mlp_gap"] for r in rows]).mean(0)
        own = [int(L) for L in np.argsort(-gap)[:6]]
        donor = {L: torch.stack([v[L] for v in fvecs]).mean(0).cpu() for L in range(nL)}
        note = ""
    else:   # too few learning pairs (the benchmark's catch form barely loops): use the MATH switch
        own, donor = list(MATH_SIX), ref
        note = f"only {len(rows)} learning pairs: own switch = the MATH switch"
        print(f"  {note}")
    torch.save(dict(donor=donor, n=len(fvecs)), os.path.join(out, "donor.pt"))
    json.dump(dict(own=own, math=MATH_SIX, control=CONTROL, n_pairs=len(rows), note=note),
              open(os.path.join(out, "components.json"), "w"))
    print(f"  own stop MLPs (top-6 gap contribution, {len(rows)} pairs): {own}   MATH six: {MATH_SIX}")
    # patching check on the learning pairs: % of the gap recovered at the commit point
    patches = {"own": {L: donor[L].cuda() for L in own}, "math": {L: ref[L].cuda() for L in MATH_SIX},
               "control": {L: donor[L].cuda() for L in CONTROL}}
    for r, p in zip(rows, [p for p in P if positions(model, p, tid)]):
        pos = positions(model, p, tid)
        o, _ = measure(model, pos[2], pos[3], u, tid, patches=patches)
        g = r["finish_logit"] - r["loop_logit"]
        for k in patches:
            r[f"recovered_{k}"] = (o[f"patch_{k}"] - r["loop_logit"]) / g if g else None
    wjl(os.path.join(out, "signal_pairs.jsonl"), rows)
    if rows:
        print("  gap recovered by patching: " + ", ".join(
        f"{k} {100 * np.mean([r[f'recovered_{k}'] for r in rows]):.0f}%" for k in patches))


# --------------------------------------------------------------------- generation
def conds_and_donors(comp, donor, ref):
    from rq3_resolve import Cond
    batch_own = [Cond("x_base", "x"), Cond("x_force", "x", "force"), Cond("x_bias", "x", "bias"),
                 Cond("x_mlp@1", "x", "interp", 1.0, comp["own"], once=True),
                 Cond("x_ctrl@1", "x", "interp", 1.0, comp["control"], once=True)]
    names_own = ["base", "force", "bias", "switch", "ctrl"]
    batch_math = [Cond("x_mlp@1", "x", "interp", 1.0, comp["math"], once=True)]
    d_own = {L: donor[L].cuda() for L in set(comp["own"]) | set(comp["control"])}
    d_math = {L: ref[L].cuda() for L in comp["math"]}
    return [(batch_own, names_own, d_own), (batch_math, ["switch_math"], d_math)]


def run_conditions(model, prefix, batches, tid, temp, rng, think_cap, post_cap):
    import torch
    from rq3_resolve import generate_batch
    uhat = torch.zeros(model.cfg.d_model, device="cuda")
    res = {}
    for conds, names, donor in batches:
        outs = generate_batch(model, prefix, conds, tid, think_cap, post_cap, donor, uhat, {},
                              temperature=temp, rng=rng)
        res.update(zip(names, outs))
    return res


def score_row(model, o, item):
    post = model.tokenizer.decode(o["post_ids"]) if o["stop"] is not None else None
    pred = extract(post, item) if post is not None else None
    return post, pred, (is_correct(pred, item) if post is not None else False)


def prefix_ids(model, prompt, reasoning, off):
    from stop_signal import chat_prefix, encode, token_at
    pre = chat_prefix(model, prompt)
    toks, offs = encode(model, pre + reasoning)
    k = token_at(offs, len(pre) + off)
    return None if k is None or k < 2 else toks[:k]


# --------------------------------------------------------------------- stage: replay
def stage_replay(a, out):
    import torch
    model, tid, u = load_model()
    comp = json.load(open(os.path.join(out, "components.json")))
    donor = torch.load(os.path.join(out, "donor.pt"))["donor"]
    ref = math_ref_donor(model, tid, u, os.path.dirname(out))["donor"]
    batches = conds_and_donors(comp, donor, ref)
    if a.lite:          # force + the two switches only (bias / ctrl / base known from the full runs)
        keep = [(c, n) for c, n in zip(*batches[0][:2]) if n in ("force", "switch")]
        batches[0] = ([c for c, _ in keep], [n for _, n in keep], batches[0][2])
    items = items_of(a.bench)
    path = os.path.join(out, "replay.jsonl")
    done = {r["id"] for r in jl(path)}
    T = [t for t in jl(os.path.join(out, "test_loops.jsonl")) if t["commit_off"] is not None and t["id"] not in done]
    rng = torch.Generator(device="cuda").manual_seed(0)
    print(f"  replay: {len(T)} held-out loops to run; conditions: base force bias switch ctrl switch_math")
    with open(path, "a") as f:
        for i, t in enumerate(T, 1):
            prefix = prefix_ids(model, t["prompt"], t["reasoning"], t["commit_off"])
            if prefix is None:
                continue
            t0 = time.time()
            try:
                res = run_conditions(model, prefix, batches, tid, a.temp, rng, a.think_cap, a.post_cap)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [{i}/{len(T)}] {t['id']}: out of GPU memory ({len(prefix)} tokens), skipped")
                continue
            for name, o in res.items():
                post, pred, ok = score_row(model, o, items[t["question"]])
                f.write(json.dumps(dict(id=t["id"], question=t["question"], condition=name,
                                        prefix_tokens=len(prefix), stop=o["stop"], think_new=len(o["think_ids"]),
                                        post_tokens=len(o["post_ids"]), bias=o["bias"], pred=pred, correct=ok,
                                        think=model.tokenizer.decode(o["think_ids"]), post=post),
                                   ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(T)}] {t['id']:20s} {time.time() - t0:5.0f}s  " + " ".join(
                f"{n}={'stop' if o['stop'] is not None else '-'}/{'ok' if score_row(model, o, items[t['question']])[2] else 'x'}"
                for n, o in res.items()))


# --------------------------------------------------------------------- stage: trigger
def find_trigger(tok, reasoning, mcq, n_tokens, fallback=1000):
    """(char offset of the doubt phrase / next paragraph, kind) or None: the rule uses
    only the text up to that point and no answer key"""
    offs = tok(reasoning, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    if len(offs) <= n_tokens:
        return None
    start = offs[n_tokens][0]
    # MCQ: a conclusion cue ("answer", "option", "not among"...); open-ended: a sentence
    # that ENDS in a stated value ("... so x = 42."), since maths loops rarely say "answer"
    cl = claims(reasoning, True, STRONG) if mcq else open_claims(reasoning)
    for o, _ in cl:
        if o >= start:
            if len(tok(reasoning[start:o], add_special_tokens=False)["input_ids"]) <= fallback:
                return o, "commit"
            break
    if len(offs) <= n_tokens + fallback:
        return None
    m = re.compile(r"\n\n(?=\S)").search(reasoning, offs[n_tokens + fallback][0])
    return (m.end(), "paragraph") if m else None


def stage_trigger(a, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    items = items_of(a.bench)
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "live_runs", "seed*.jsonl"))):
        for r in jl(p):
            mcq = items[r["id"]]["format"] == "mcq"
            tr = find_trigger(tok, r["reasoning"], mcq, a.trigger_tokens)
            rows.append(dict(id=r["id"], seed=r["seed"], truncated=r["truncated"], correct=r["correct"],
                             completion_tokens=r["completion_tokens"],
                             trigger_off=tr[0] if tr else None, trigger_kind=tr[1] if tr else None))
    wjl(os.path.join(out, "live_triggers.jsonl"), rows)
    fired = [r for r in rows if r["trigger_off"] is not None]
    print(f"  live runs {len(rows)}: loops {sum(r['truncated'] for r in rows)}; trigger fires on {len(fired)} "
          f"({sum(r['truncated'] for r in fired)} loops, {sum(not r['truncated'] for r in fired)} would finish; "
          f"kinds {dict(collections.Counter(r['trigger_kind'] for r in fired))})")


# --------------------------------------------------------------------- stage: live
def stage_live(a, out):
    import torch
    model, tid, u = load_model()
    comp = json.load(open(os.path.join(out, "components.json")))
    donor = torch.load(os.path.join(out, "donor.pt"))["donor"]
    ref = math_ref_donor(model, tid, u, os.path.dirname(out))["donor"]
    batches = conds_and_donors(comp, donor, ref)
    # live test: force + the two switches only (bias and ctrl behave as in the replay)
    keep = [(c, n) for c, n in zip(*batches[0][:2]) if n in ("force", "switch")]
    batches[0] = ([c for c, _ in keep], [n for _, n in keep], batches[0][2])
    items = items_of(a.bench)
    runs = {(r["id"], r["seed"]): r for p in glob.glob(os.path.join(out, "live_runs", "seed*.jsonl")) for r in jl(p)}
    path = os.path.join(out, "live_branches.jsonl")
    done = {(r["id"], r["seed"]) for r in jl(path)}
    F = [t for t in jl(os.path.join(out, "live_triggers.jsonl"))
         if t["trigger_off"] is not None and (t["id"], t["seed"]) not in done]
    rng = torch.Generator(device="cuda").manual_seed(1)
    print(f"  live: {len(F)} triggered runs to continue")
    with open(path, "a") as f:
        for i, t in enumerate(F, 1):
            r = runs[(t["id"], t["seed"])]
            prefix = prefix_ids(model, r["prompt"], r["reasoning"], t["trigger_off"])
            if prefix is None:
                continue
            t0 = time.time()
            try:
                res = run_conditions(model, prefix, batches, tid, a.temp, rng, a.think_cap, a.post_cap)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [{i}/{len(F)}] {t['id']}: out of GPU memory, skipped")
                continue
            for name, o in res.items():
                if name == "base":
                    continue
                post, pred, ok = score_row(model, o, items[t["id"]])
                f.write(json.dumps(dict(id=t["id"], seed=t["seed"], condition=name, prefix_tokens=len(prefix),
                                        trigger_kind=t["trigger_kind"], stop=o["stop"],
                                        think_new=len(o["think_ids"]), post_tokens=len(o["post_ids"]),
                                        pred=pred, correct=ok, think=model.tokenizer.decode(o["think_ids"]),
                                        post=post), ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(F)}] {t['id']} seed{t['seed']} ({'loop' if t['truncated'] else 'would finish'}"
                  f"{', right' if t['correct'] else ''}) {time.time() - t0:4.0f}s  " + " ".join(
                      f"{n}={'ok' if score_row(model, o, items[t['id']])[2] else 'x'}"
                      for n, o in res.items() if n != "base"))


# --------------------------------------------------------------------- stage: summary
def pct(x):
    return f"{100 * np.mean(x):.0f}%" if len(x) else "-"


def boot(v, n=2000):
    v = np.array(v, float)
    if not len(v):
        return "-"
    rng = np.random.default_rng(0)
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return f"[{100 * np.percentile(b, 2.5):.0f}-{100 * np.percentile(b, 97.5):.0f}]"


def stage_summary(a, out):
    cfg = json.load(open(os.path.join(out, "config.json")))
    L = [f"# Per-benchmark stop switch: {a.bench}, T={a.temp}", "",
         "Learned from this benchmark's catch-vs-aligned pairs; tested on its natural loops "
         "(held out). Settings: `config.json`.", "", "## Data", ""]
    for k, v in json.load(open(os.path.join(out, "pairs_stats.json"))).items():
        L.append(f"- {k}: {v}")
    S = jl(os.path.join(out, "signal_pairs.jsonl"))
    if S:
        comp = json.load(open(os.path.join(out, "components.json")))
        g = [r["finish_logit"] - r["loop_logit"] for r in S]
        mg = np.array([r["mlp_gap"] for r in S]).mean(0)
        L += ["", "## Learning: the stop signal on this benchmark's catch-vs-aligned pairs", "",
              f"- pairs: {len(S)}",
              f"- logit(</think>): finished stop {np.mean([r['finish_logit'] for r in S]):+.1f}, "
              f"loop commit {np.mean([r['loop_logit'] for r in S]):+.1f}, gap {np.mean(g):+.1f} "
              f"(gap > 0 in {pct([x > 0 for x in g])})",
              f"- MLP share of the gap: {100 * mg.sum() / np.mean(g):.0f}%",
              f"- own stop MLPs (top-6): {comp['own']}; MATH six: {comp['math']}; "
              f"share of gap in own six: {100 * mg[comp['own']].sum() / np.mean(g):.0f}%, "
              f"in MATH six: {100 * mg[comp['math']].sum() / np.mean(g):.0f}%",
              "- gap recovered by patching at the commit point: " + ", ".join(
                  f"{k} {100 * np.mean([r[f'recovered_{k}'] for r in S]):.0f}%" for k in ("own", "math", "control"))]
    order = ["base", "force", "bias", "switch", "switch_math", "ctrl"]
    R = jl(os.path.join(out, "replay.jsonl"))
    if R:
        L += ["", "## Option A: replay on the held-out natural loops", "",
              f"One loop per held-out question, fed up to its commit point; answer generated fresh "
              f"(T={a.temp}, answer cap {cfg['post_cap']}, thinking cap {cfg['think_cap']}).", "",
              "| condition | loops | stops | correct [95% CI] | correct (of stopped) | answer hits the cap |",
              "|---|---|---|---|---|---|"]
        for c in order:
            Rc = [r for r in R if r["condition"] == c]
            if not Rc:
                continue
            st = [r for r in Rc if r["stop"] is not None]
            L.append(f"| {c} | {len(Rc)} | {pct([r['stop'] is not None for r in Rc])} | "
                     f"{pct([r['correct'] for r in Rc])} {boot([r['correct'] for r in Rc])} | "
                     f"{pct([r['correct'] for r in st])} | {pct([r['post_tokens'] >= cfg['post_cap'] for r in st])} |")
    T = jl(os.path.join(out, "live_triggers.jsonl"))
    B = jl(os.path.join(out, "live_branches.jsonl"))
    if T:
        fired = {(t["id"], t["seed"]) for t in T if t["trigger_off"] is not None}
        L += ["", "## Option B: live test on the held-out questions (fresh runs)", "",
              f"- fresh runs: {len(T)} ({len({t['id'] for t in T})} questions x seeds "
              f"{sorted({t['seed'] for t in T})}); loops {sum(t['truncated'] for t in T)}; "
              f"correct with no intervention {pct([bool(t['correct']) for t in T])}",
              f"- rule fires on {len(fired)} runs: {sum(t['truncated'] for t in T if (t['id'], t['seed']) in fired)} loops, "
              f"{sum(not t['truncated'] for t in T if (t['id'], t['seed']) in fired)} that would have finished "
              f"({sum(bool(t['correct']) for t in T if (t['id'], t['seed']) in fired)} of them correctly)", ""]
        if B:
            br = {(r["id"], r["seed"], r["condition"]): r for r in B}
            have = {(r["id"], r["seed"]) for r in B}
            L += ["Accuracy over ALL fresh runs. Runs the rule did not fire on keep their own outcome; so do "
                  f"triggered runs whose branch did not stop within {cfg['think_cap']} tokens (the intervention "
                  "did not take effect: a loop stays a loop, a run that would finish keeps its answer). "
                  "Only runs whose branches are done are counted.", "",
                  "| condition | runs | correct [95% CI] | loops left | of triggered: loops rescued | "
                  "would-finish runs hurt (right -> wrong) |", "|---|---|---|---|---|---|"]
            TT = [t for t in T if (t["id"], t["seed"]) not in fired or (t["id"], t["seed"]) in have]
            for c in ["none"] + [x for x in order if x not in ("base",)]:
                cor, left, resc, hurt = [], 0, 0, 0
                for t in TT:
                    k = (t["id"], t["seed"])
                    if c == "none" or k not in fired:
                        cor.append(bool(t["correct"]))
                        left += t["truncated"]
                        continue
                    b = br.get((t["id"], t["seed"], c))
                    if b is None:
                        break
                    ok = bool(b["correct"]) if b["stop"] is not None else bool(t["correct"])
                    cor.append(ok)
                    left += b["stop"] is None and t["truncated"]
                    resc += t["truncated"] and ok
                    hurt += bool(t["correct"]) and not ok
                else:
                    L.append(f"| {c} | {len(cor)} | {pct(cor)} {boot(cor)} | {left} | "
                             + ("- | - |" if c == "none" else f"{resc} | {hurt} |"))
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    global MODEL, TAG, NATR, CTLR, MATH_SIX, CONTROL
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, choices=list(MODELS))
    ap.add_argument("--bench", required=True, choices=list(NAT))
    ap.add_argument("--temp", type=float, required=True, choices=[0.0, 0.6])
    ap.add_argument("--stage", required=True,
                    choices=["pairs", "signal", "replay", "trigger", "live", "summary"])
    ap.add_argument("--trigger-tokens", type=int, default=6000,
                    help="fixed in advance (not tuned): no </think> after this many thinking tokens")
    ap.add_argument("--max-heldout", type=int, default=0,
                    help="test a random subset of this many held-out questions (0 = all)")
    ap.add_argument("--lite", action="store_true", help="replay: force + switches only")
    ap.add_argument("--think-cap", type=int, default=256)
    ap.add_argument("--post-cap", type=int, default=1200)
    a = ap.parse_args()
    a.temp = 0 if a.temp == 0 else 0.6
    MODEL, m = a.model, MODELS[a.model]
    TAG, MATH_SIX, CONTROL = m["tag"], m["ref"], m["control"]
    NATR = os.path.join(ROOT, "natural_looping", "results", TAG)
    CTLR = os.path.join(ROOT, "controlled_looping", "results", TAG)
    out = os.path.join(HERE, "results", TAG, "bench_switch", f"{a.bench}_t{a.temp:g}")
    os.makedirs(out, exist_ok=True)
    cfg_path = os.path.join(out, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    cfg.update(model=MODEL, bench=a.bench, temperature=a.temp, trigger_tokens=a.trigger_tokens,
               think_cap=a.think_cap, post_cap=a.post_cap, max_heldout=a.max_heldout, math_six=MATH_SIX, control=CONTROL,
               live_seeds=cfg.get("live_seeds"), **{f"ran_{a.stage}": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())})
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    print(f"[{a.stage}] {a.bench} T={a.temp} -> {os.path.relpath(out, ROOT)}")
    dict(pairs=stage_pairs, signal=stage_signal, replay=stage_replay, trigger=stage_trigger,
         live=stage_live, summary=stage_summary)[a.stage](a, out)


if __name__ == "__main__":
    main()
