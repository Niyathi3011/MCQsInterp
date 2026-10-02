"""
RQ3 / E3c -- does restoring the late-MLP stop signal RESOLVE the catch loop, or
does it only force the </think> token?

E3a/E3b showed the late MLPs (27,26,25,24,22,20) are sufficient/necessary for
logit(</think>) at ka.  But E3b clamps at the token right before </think> and the
model stops at step 0: that is compatible with a pure "stop gate" (H1) as well as
with the MLPs carrying the decision "stop reconciling, the answer is not listed"
(H2).  This script separates the two.

Differences from rq3_patch.py --experiment e3b:
  * KV-cached, batched decoding: all catch conditions of a problem run as one
    batch (each row with its own edit), then the aligned ones.
  * --start decide (default): cut the catch trace right BEFORE the first
    backtrack phrase that follows the model's first statement of the gold value
    -- the point where an aligned trace would conclude and catch starts doubting.
    --start ka reproduces the E3b prefix.
  * edits are PERSISTENT: applied at the start token and at every generated
    position while the model thinks (those positions stay in the KV cache as
    edited).  "...once" conditions edit the start token only.  Overwriting whole
    MLP outputs at every position wrecks generation (smoke test: orth -> "Wait
    Wait Wait", zero-lesion -> "X X X"), so full-vector edits other than the
    graded interp run once, and the persistent lesion removes only the
    </think> component.
  * catch conditions
      base           no edit
      force          append </think> immediately (budget forcing)
      bias           add b to logit(</think>) every thinking step; b = this
                     problem's start-token logit increase under mlp@1
      mlp@a          v + a*(donor - v), a in --alphas         (graded)
      proj@1         only the </think>-unembed-direction part of the edit
      orth@1once     everything EXCEPT that direction, start token only
      random@1       same size as the full edit, random fixed direction
      ctrl@1         full edit on --control-layers
  * aligned conditions (prefix ends at its own pre-</think> token)
      base, lesionproj (remove the </think> component, persistent),
      lesion0once (zero the components at the start token only)

Per generation it records the thinking text and the unclamped text after
</think>, and scores: stop step, backtrack phrases / 100 words, duplicate
sentence fraction, "not among the options" phrases, and in the post-</think>
text: states gold value / commits to a letter / flags the mismatch.

  python phase2/rq3_resolve.py --pairs phase2/loop_pairs_raw.jsonl --raw-think \
      --n 2 --gen-tokens 256 --post-tokens 128 --out phase2/raw/e3c_smoke.jsonl
  python phase2/rq3_resolve.py ... --out phase2/raw/e3c.jsonl        # resumable
  python phase2/rq3_resolve.py --summary-only --out phase2/raw/e3c.jsonl

GPU; stop any vLLM server first.
"""
import argparse
import collections
import json
import os
import re
import sys
import time

import numpy as np
import torch

import loop_probe
from loop_probe import build_full, c2t, get_model, think_dir, toks_of
from prep_loop import BACKTRACK
from check_gold_in_rescue import parse_letter, states_value
from rq3_patch import hook_names, ka_of, layer_of, parse_layers
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from run_model import letter_from_number, num_equal, parse_number  # noqa: E402
from run_model import parse_letter as rm_parse_letter  # noqa: E402

MISMATCH = re.compile(
    r"(not (among|one of|in|listed in|part of) (the )?(given |provided |answer )?(options|choices)"
    r"|none of the (given |provided |answer )?(options|choices)|neither (option|choice|of the)"
    r"|(isn['’]?t|is not|aren['’]?t|are not) (an? )?(valid )?(option|choice)s?"
    r"|no (correct|valid|matching) (option|choice|answer))",
    re.I,
)


# --------------------------------------------------------------------- prefixes
def decide_prefix(model, piece):
    """catch tokens up to (not incl.) the first backtrack phrase after the gold
    value is first stated.  None if either marker is missing."""
    r = piece["reasoning"]
    if piece.get("ans_off") is None:
        return None
    m = BACKTRACK.search(r, piece["ans_off"] + len(str(piece["gold_value"])))
    if not m:
        return None
    full, _, rcs, _ = build_full(model, piece)
    idx = c2t(model, full, rcs + m.start())
    toks = toks_of(model, full)
    return toks[:idx] if 0 < idx < len(toks) else None


def prefixes(model, rec, start):
    """(aligned_prefix ending at ka, catch_prefix) -- aligned always ends at its
    own pre-</think> token; catch at `start`."""
    got = ka_of(model, rec)
    if got is None:
        return None
    A, C, _ = got
    if start == "decide":
        C = decide_prefix(model, rec["catch"])
        if C is None:
            return None
    return A, C


# --------------------------------------------------------------------- conditions
class Cond:
    """One generation row.  kind: none | force | bias | interp | proj | orth |
    random | projzero | zero.  once=True edits only the start token (the last
    prefix token); otherwise every position while the model is thinking."""
    def __init__(self, name, trace, kind="none", alpha=1.0, layers=(), once=False):
        self.name, self.trace, self.kind = name, trace, kind
        self.alpha, self.layers, self.once = alpha, set(layers), once


def conditions(args, comp, ctrl):
    cs = [Cond("catch_base", "catch"),
          Cond("catch_force", "catch", "force"),
          Cond("catch_bias", "catch", "bias")]
    for a in sorted(parse_floats(args.alphas), reverse=True):
        cs.append(Cond(f"catch_mlp@{a:g}", "catch", "interp", a, comp))
    cs += [Cond("catch_proj@1", "catch", "proj", 1.0, comp),
           Cond("catch_orth@1once", "catch", "orth", 1.0, comp, once=True),
           Cond("catch_random@1", "catch", "random", 1.0, comp)]
    if ctrl:
        cs.append(Cond("catch_ctrl@1", "catch", "interp", 1.0, ctrl))
    cs += [Cond("aligned_base", "aligned"),
           Cond("aligned_lesionproj", "aligned", "projzero", 1.0, comp),
           Cond("aligned_lesion0once", "aligned", "zero", 1.0, comp, once=True)]
    if args.loops:                      # natural loops: only the catch-style edits,
        cs = [c for c in cs if c.trace == "catch"]      # applied to the loop trace
        for c in cs:
            c.name, c.trace = c.name.replace("catch_", "loop_", 1), "loop"
    if args.conditions:
        keep = set(args.conditions.split(","))
        cs = [c for c in cs if c.name in keep]
    if any(c.kind == "bias" for c in cs):
        have = {c.name.split("_", 1)[1] for c in cs}
        assert {"base", "mlp@1"} <= have, "bias needs the base and mlp@1 rows in the same run"
    return cs


def edit(v, c, d, uhat, r):
    """v: mlp_out at the edited position (fp32).  d: donor for this layer.
       e = alpha*(d - v)
      interp   v + e                  graded version of rq3_patch 'replace'
      proj     v + (e.u)u             only the </think>-direction part of e
      orth     v + e - (e.u)u         everything except that part
      random   v + |e| r              same size, fixed random direction
      projzero v - (v.u)u             remove the </think> component (surgical lesion)
      zero     0                      full lesion"""
    if c.kind == "zero":
        return torch.zeros_like(v)
    if c.kind == "projzero":
        return v - (v @ uhat) * uhat
    e = c.alpha * (d - v)
    if c.kind == "interp":
        return v + e
    if c.kind == "proj":
        return v + (e @ uhat) * uhat
    if c.kind == "orth":
        return v + e - (e @ uhat) * uhat
    if c.kind == "random":
        return v + e.norm() * r
    raise ValueError(c.kind)


# --------------------------------------------------------------------- decode
def prefill(model, toks, chunk=512):
    """KV cache holding toks[:-1], fed in chunks: a single forward over a long
    prefix builds (heads, L, L) fp32 attention patterns that OOM a 24 GB card
    at a few thousand tokens.  The caller runs toks[-1] itself, under hooks."""
    from transformer_lens import TransformerLensKeyValueCache
    cache = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, 1)
    body = toks[:-1]
    for i in range(0, len(body), chunk):
        model(body[i:i + chunk].unsqueeze(0), past_kv_cache=cache)
    return cache


def expand_cache(cache, B):
    for e in cache.entries:
        e.past_keys = e.past_keys.expand(B, -1, -1, -1).contiguous()
        e.past_values = e.past_values.expand(B, -1, -1, -1).contiguous()
    cache.previous_attention_mask = cache.previous_attention_mask.expand(B, -1).contiguous()
    return cache


@torch.inference_mode()
def generate_batch(model, prefix, conds, tid, think_cap, post_cap, donor, uhat, rdir,
                   temperature=0.0, rng=None):
    """All `conds` share `prefix`; decode them as one batch, row b under its own
    edit.  KV cache: the prefix is prefilled once (unedited) and copied per row;
    the start token and every generated token are then run batched.

    Row b is edited at the last position of each forward while it is thinking
    (only at the start token if once=True).  When it emits </think> its edit is
    switched off before </think> is fed back, and the answer is generated
    unedited until EOS, a second </think>, or post_cap.  catch_bias gets
    b = logit(</think>)[catch_mlp@1] - logit(</think>)[catch_base] at the start
    token, i.e. the same first-step push as the full MLP edit, but on the
    logit only."""
    B = len(conds)
    eos = model.tokenizer.eos_token_id
    dev = prefix.device
    cache = expand_cache(prefill(model, prefix), B)
    step = [0]
    active = [c.kind not in ("none", "force", "bias") for c in conds]

    def hook(act, hook):
        L = layer_of(hook.name)
        for b, c in enumerate(conds):
            if active[b] and L in c.layers and not (c.once and step[0] > 0):
                act[b, -1] = edit(act[b, -1].float(), c, donor.get(L), uhat,
                                  rdir.get(L)).to(act.dtype)
        return act

    layers = sorted(set().union(*[c.layers for c in conds]))
    hooks = [(n, hook) for n in hook_names(layers)]

    def pick(row):
        if temperature > 0:
            return int(torch.multinomial(torch.softmax(row / temperature, -1), 1,
                                         generator=rng))
        return int(row.argmax())

    phase = ["think"] * B
    think, post = [[] for _ in range(B)], [[] for _ in range(B)]
    stop, eos_in_think = [None] * B, [False] * B
    names = [c.name for c in conds]
    with model.hooks(fwd_hooks=hooks):
        logits = model(prefix[-1:].expand(B, 1), past_kv_cache=cache)[:, -1].float()
        first = logits[:, tid].tolist()
        bias = None
        if any(c.kind == "bias" for c in conds):
            short = [n.split("_", 1)[1] for n in names]
            bias = first[short.index("mlp@1")] - first[short.index("base")]
        while True:
            nxt = []
            for b, c in enumerate(conds):
                row = logits[b]
                if phase[b] == "think":
                    if c.kind == "force":
                        tok = tid
                    else:
                        if c.kind == "bias":
                            row = row.clone()
                            row[tid] += bias
                        tok = pick(row)
                    think[b].append(tok)
                    if tok == tid:
                        stop[b] = len(think[b]) - 1
                        active[b] = False
                        phase[b] = "post" if post_cap else "done"
                    elif tok == eos:
                        eos_in_think[b], phase[b] = True, "done"
                    elif len(think[b]) >= think_cap:
                        phase[b] = "done"
                elif phase[b] == "post":
                    tok = pick(row)
                    post[b].append(tok)
                    if tok in (eos, tid) or len(post[b]) >= post_cap:
                        phase[b] = "done"
                else:
                    tok = eos                       # finished row: filler, ignored
                nxt.append(tok)
            if all(p == "done" for p in phase):
                break
            step[0] += 1
            logits = model(torch.tensor(nxt, device=dev)[:, None],
                           past_kv_cache=cache)[:, -1].float()
    del cache
    torch.cuda.empty_cache()
    return [dict(stop=stop[b], eos_in_think=eos_in_think[b], first_logit=first[b],
                 bias=bias if conds[b].kind == "bias" else None,
                 think_ids=think[b], post_ids=post[b]) for b in range(B)]


@torch.inference_mode()
def selftest(model, prefix, n=24):
    """Teacher-forced check of the cached, batched path: decode n greedy tokens
    (two identical unedited rows, so the batch path is exercised), then run ONE
    uncached forward over prefix+those tokens and ask whether its argmax at each
    step reproduces the cached choice.  (Comparing two free-running greedy
    sequences is useless: a single bf16 near-tie flips one token and every later
    token differs.)"""
    tid, _ = think_dir(model)
    out = generate_batch(model, prefix, [Cond("a", "catch"), Cond("b", "catch")], tid,
                         n, 0, {}, None, {})
    cached = out[0]["think_ids"]
    same_rows = cached == out[1]["think_ids"]
    seq = torch.cat([prefix, prefix.new_tensor(cached)])
    logits = model(seq.unsqueeze(0))[0, len(prefix) - 1:-1].float()
    plain = logits.argmax(-1).tolist()
    agree = sum(a == b for a, b in zip(cached, plain))
    top2 = logits.topk(2, -1).values
    ties = int(((top2[:, 0] - top2[:, 1]) < 0.5).sum())
    print(f"  selftest (teacher-forced): cached greedy == uncached argmax on "
          f"{agree}/{len(plain)} steps; {ties} steps have a top-2 margin < 0.5;"
          f" batch rows identical: {same_rows}"
          f"  ({'OK' if agree >= len(plain) - ties and same_rows else 'CHECK'})")


# --------------------------------------------------------------------- scoring
def sentences(text):
    return [re.sub(r"\s+", " ", s).strip().lower()
            for s in re.split(r"(?<=[.?!\n])", text) if len(s.strip()) > 15]


def answer_correct(post, opts):
    """Is the answer after </think> right?  opts = the trace's option_A/B and
    gold_letter (None for open-ended).  Same parsers as run_model.py's scoring;
    None when there is no correct answer to get (catch)."""
    if opts is None:
        return None
    if opts.get("option_A") is None:                                  # open-ended
        return num_equal(parse_number(post), opts["gold"]) if post.strip() else False
    if not opts.get("gold_letter"):
        return None
    pred = rm_parse_letter(post) or letter_from_number(
        post, opts["option_A"], opts["option_B"], opts["gold"], opts["gold_letter"])
    return pred == opts["gold_letter"]


def score(think, post, gold, stopped, opts=None):
    """think: text decoded up to and including </think> (if emitted).
    opts: option_A/B, gold_letter, gold of the trace, to score correctness."""
    body = think.replace("</think>", "")
    words = max(len(body.split()), 1)
    sents = sentences(think)
    dup = 1 - len(set(sents)) / len(sents) if sents else 0.0
    d = dict(
        think_words=len(body.split()),
        backtrack_per100w=100 * len(BACKTRACK.findall(body)) / words,
        mismatch_think=bool(MISMATCH.search(body)),
        dup_sent_frac=dup,
        # an answer written INSIDE the thinking block (e.g. a lesioned trace
        # that concludes but never emits </think>)
        think_letter=parse_letter(body) if body.strip() else None,
    )
    if stopped:
        # natural stop = </think> right after a sentence/paragraph end (or at
        # once); a forced gate tends to cut mid-sentence ("...and they didn</think>")
        d["stop_clean"] = (not body.strip()) or bool(re.search(r"[.!?:)\]}*]\s*$|\n\s*$", body))
        letter = parse_letter(post) if post.strip() else None
        d.update(post_gold=states_value(gold, post), post_letter=letter,
                 post_mismatch=bool(MISMATCH.search(post)))
        # gold & no letter: gives the right (unlisted) value, doesn't pick an option
        d["gold_noletter"] = d["post_gold"] and letter is None
        d["post_other"] = not d["post_gold"] and letter is None
    # a trace that never stops never answers: counts as not correct
    ok = answer_correct(post, opts) if stopped else (False if opts else None)
    d["correct"] = ok
    return d


# --------------------------------------------------------------------- run
def parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def donor_mean(model, recs, layers):
    """mean aligned mlp_out@ka over problems (same donor as rq3_patch e3b)."""
    want = set(hook_names(layers))
    tot, k = None, 0
    for rec in recs:
        got = ka_of(model, rec)
        if got is None:
            continue
        A, _, _ = got
        with torch.inference_mode():
            kv = prefill(model, A)
            _, cache = model.run_with_cache(A[-1:].unsqueeze(0), past_kv_cache=kv,
                                            names_filter=lambda n: n in want)
        d = {L: cache[f"blocks.{L}.hook_mlp_out"][0, -1].float().clone() for L in layers}
        del cache, kv
        tot = d if tot is None else {L: tot[L] + d[L] for L in layers}
        k += 1
    return {L: tot[L] / k for L in layers}, k


def run(args):
    comp, ctrl = parse_layers(args.components), parse_layers(args.control_layers)
    recs = [json.loads(l) for l in open(args.pairs)]
    if args.n:
        recs = recs[: args.n]
    print(f"{len(recs)} pairs; loading {args.model} ...")
    model = get_model(args.model, args.device, args.dtype)
    model.eval()
    model.requires_grad_(False)
    tid, u = think_dir(model)
    uhat = (u / u.norm()).float()

    allL = sorted(set(comp) | set(ctrl))
    donor, k = donor_mean(model, [json.loads(l) for l in open(args.pairs)], allL)
    print(f"  donor = mean aligned mlp_out@ka over {k} problems ({args.pairs})")
    if args.loops:
        recs = [json.loads(l) for l in open(args.loops)]
        if args.n:
            recs = recs[: args.n]
        print(f"  --loops: {len(recs)} natural loop traces from {args.loops}")
    g = torch.Generator().manual_seed(args.seed)
    rdir = {}
    for L in allL:
        r = torch.randn(model.cfg.d_model, generator=g)
        rdir[L] = (r / r.norm()).to(uhat.device)
    rng = (torch.Generator(device=args.device).manual_seed(args.seed)
           if args.temperature > 0 else None)

    conds = conditions(args, comp, ctrl)
    rows = [json.loads(l) for l in open(args.out)] if os.path.exists(args.out) else []
    done = {(r["source_idx"], r.get("variant")) for r in rows}
    if done:
        print(f"  resuming: {len(done)} problems already in {args.out}")
    print(f"  start={args.start}  gen_tokens={args.gen_tokens}  post_tokens={args.post_tokens}"
          f"  temperature={args.temperature}")
    print(f"  conditions: {', '.join(c.name for c in conds)}\n")

    if args.selftest:
        if args.loops:
            p = next(pp for pp in (decide_prefix(model, r["loop"]) for r in recs) if pp is not None)
        else:
            p = next(pp for pp in (prefixes(model, r, args.start) for r in recs)
                     if pp is not None)[1]
        selftest(model, p)

    f = open(args.out, "a")
    skipped = 0
    for i, rec in enumerate(recs):
        sidx, var = rec["source_idx"], rec.get("variant")
        if (sidx, var) in done:
            continue
        if args.loops:
            lp = decide_prefix(model, rec["loop"])
            pref = None if lp is None else (None, lp)
            piece = rec["loop"]
        else:
            pref = prefixes(model, rec, args.start)
            piece = rec["catch"] if pref else None
        if pref is None:
            skipped += 1
            print(f"  [{i + 1}/{len(recs)}] source_idx {sidx}: no {args.start} point, skipped")
            continue
        gold = str(piece["gold_value"])
        opts = dict(option_A=piece.get("option_A"), option_B=piece.get("option_B"),
                    gold_letter=piece.get("gold_letter"), gold=gold)
        traces = (("loop", pref[1]),) if args.loops else (("catch", pref[1]), ("aligned", pref[0]))
        for trace, prefix in traces:
            group = [c for c in conds if c.trace == trace]
            if not group:
                continue
            t0 = time.time()
            outs = generate_batch(model, prefix, group, tid, args.gen_tokens,
                                  args.post_tokens, donor, uhat, rdir,
                                  temperature=args.temperature, rng=rng)
            for c, o in zip(group, outs):
                think = model.tokenizer.decode(o["think_ids"])
                post = model.tokenizer.decode(o["post_ids"])
                o_opts = opts if trace in ("loop", "catch") else dict(
                    opts, gold_letter=rec["aligned"].get("gold_letter"),
                    option_A=rec["aligned"].get("option_A"),
                    option_B=rec["aligned"].get("option_B"))
                row = dict(source_idx=sidx, variant=var, condition=c.name, start=args.start,
                           prefix_len=len(prefix), gold=gold, opts=o_opts, bias=o["bias"],
                           stop=o["stop"], eos_in_think=o["eos_in_think"],
                           first_logit=o["first_logit"], think=think, post=post,
                           **score(think, post, gold, o["stop"] is not None, o_opts))
                f.write(json.dumps(row) + "\n")
                rows.append(row)
            f.flush()
            print(f"  [{i + 1}/{len(recs)}] source_idx {sidx} {var or ''} {trace:<7s} prefix={len(prefix):5d}t"
                  f"  {time.time() - t0:6.1f}s   stop@: "
                  + " ".join(f"{c.name.split('_', 1)[1]}={o['stop']}" for c, o in zip(group, outs)))
    f.close()
    if skipped:
        print(f"\n  {skipped} problems had no '{args.start}' point and were skipped")
    summarise(rows, [c.name for c in conds])


# --------------------------------------------------------------------- summary
def summarise(rows, order=None):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    order = order or sorted(by)
    # only problems that have EVERY listed condition, so rows are comparable
    key = lambda r: (r["source_idx"], r.get("variant"))                   # noqa: E731
    common = set.intersection(*[{key(r) for r in by[c]} for c in order if by[c]])
    print(f"\nE3c summary   (n={len(common)} problems with all conditions)\n")
    hdr = (f"  {'condition':<19s} {'stop%':>6s} {'med@':>5s} {'clean':>6s} {'ltrInThk':>8s}"
           f" {'bt/100w':>8s} {'dup':>5s}"
           f" {'mism.think':>10s} | {'gold':>5s} {'gold&noLtr':>10s} {'letter':>6s} {'mism.post':>9s} {'other':>6s}"
           f" | {'CORRECT':>7s}")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    pct = lambda xs: f"{100 * np.mean(xs):5.0f}%" if xs else "    -"
    for c in order:
        rs = [r for r in by[c] if key(r) in common]
        if not rs:
            continue
        st = [r for r in rs if r["stop"] is not None]
        med = int(np.median([r["stop"] for r in st])) if st else "-"
        print(f"  {c:<19s} {pct([r['stop'] is not None for r in rs]):>6s} {str(med):>5s}"
              f" {pct([r['stop_clean'] for r in st]):>6s}"
              f" {pct([r.get('think_letter') is not None for r in rs if r['stop'] is None]):>8s}"
              f" {np.mean([r['backtrack_per100w'] for r in rs]):8.2f}"
              f" {np.mean([r['dup_sent_frac'] for r in rs]):5.2f}"
              f" {pct([r['mismatch_think'] for r in rs]):>10s} |"
              f" {pct([r['post_gold'] for r in st]):>5s}"
              f" {pct([r['gold_noletter'] for r in st]):>10s}"
              f" {pct([r['post_letter'] is not None for r in st]):>6s}"
              f" {pct([r['post_mismatch'] for r in st]):>9s}"
              f" {pct([r['post_other'] for r in st]):>6s}"
              f" | {pct([r['correct'] for r in rs if r.get('correct') is not None]):>7s}")
    print("\n  stop%      emitted </think> within the thinking budget"
          "\n  clean      of stopped: </think> came at a sentence/paragraph boundary"
          "\n  ltrInThk   of NOT stopped: committed to (A)/(B) inside the thinking text"
          "\n  bt/100w    backtrack phrases per 100 words while thinking (loop proxy)"
          "\n  dup        fraction of thinking sentences that repeat an earlier one"
          "\n  mism.think thinking text says the answer is not among the options"
          "\n  right of | : over STOPPED generations only, text after </think>:"
          "\n    gold = states the correct value, gold&noLtr = states it and picks no option,"
          "\n    letter = commits to (A)/(B) (always wrong on catch; right on aligned),"
          "\n    mism.post = says no option matches, other = neither gold nor a letter"
          "\n  CORRECT    over ALL problems (no stop = no answer = wrong); only where a"
          "\n             correct answer exists (natural loops, aligned)")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="phase2/loop_pairs_raw.jsonl")
    ap.add_argument("--raw-think", action="store_true")
    ap.add_argument("--loops", default=None,
                    help="natural-loop mode: jsonl from prep_solvable_loops.py; edits run on "
                         "each looping trace from its decide point; --pairs only supplies "
                         "the donor")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=0, help="limit problems (0 = all)")
    ap.add_argument("--start", default="decide", choices=["decide", "ka"])
    ap.add_argument("--components", default="27,26,25,24,22,20")
    ap.add_argument("--control-layers", default="15")
    ap.add_argument("--alphas", default="0.25,0.5,1")
    ap.add_argument("--conditions", default=None, help="comma subset of condition names")
    ap.add_argument("--gen-tokens", type=int, default=1024, help="thinking-phase cap")
    ap.add_argument("--post-tokens", type=int, default=400, help="answer cap after </think>")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="phase2/raw/e3c.jsonl")
    ap.add_argument("--selftest", action="store_true",
                    help="check KV-cached greedy == uncached greedy before running")
    ap.add_argument("--summary-only", action="store_true",
                    help="no model: re-score --out from its stored text and print the table")
    args = ap.parse_args()
    loop_probe.RAW_THINK = args.raw_think
    if args.summary_only:                    # re-scored from stored text: metric
        rows = [json.loads(l) for l in open(args.out)]   # changes need no rerun
        summarise([dict(r, **score(r["think"], r["post"], r["gold"], r["stop"] is not None,
                                   r.get("opts"))) for r in rows])
        return
    run(args)


if __name__ == "__main__":
    main()
