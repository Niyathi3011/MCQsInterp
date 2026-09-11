"""
RQ3 -- causal test of the RQ2 / E2c result.  See phase2/RQ3_DESIGN.md.

E2c (correlational): the aligned-vs-catch gap in logit(</think>) at the
pre-</think> token `ka` is written almost entirely by late-layer MLPs, dominated
by MLP 27 (~48% of a ~+11.5 gap); on catch those MLPs go quiet.

RQ3 forces / removes that write and sees if the outcome moves:

  RESCUE  (catch  <- aligned) : set the stop-writing MLP output(s) at ka to their
          terminated-trace value.   predict logit(</think>) -> aligned; loop ends.
  LESION  (aligned -> ablate) : zero the same MLP output(s).
          predict logit(</think>) -> catch; termination breaks.

  E3a  static patch at ka, measure logit(</think>)          (fast, deterministic)
  E3b  --generate: clamp during greedy decode from ka,       (slow, behavioural)
       measure whether </think> is emitted / the text loops
  CONTROL: same patch on a late MLP NOT in the E2c set (--control-layers).

  E3b credibility checks (all optional flags on the same command):
    --save-transcripts FILE   dump every generated continuation, to confirm
                               RESCUE produces a coherent conclusion, not garbage
    --surgical                edit ONLY the </think>-direction component of the
                               patched MLPs' output (proj_set/proj_zero), not the
                               whole vector -- is it that specific write, or any
                               edit to these layers?
    --donor-mode catch        RESCUE donor = catch's own (already-quiet) value --
                               sanity check that the donor's CONTENT matters
    --donor-mode random       RESCUE donor = a magnitude-matched random direction
                               -- isolates direction from perturbation size
    --post-think-tokens N     after </think> fires, drop the clamp and generate N
                               more tokens UNCLAMPED -- without this, "100% emitted
                               </think>" only proves the stop TOKEN appeared, not
                               that a real answer follows it

  python phase2/rq3_patch.py --pairs phase2/loop_pairs.jsonl \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --n 12
  python phase2/rq3_patch.py ... --components 27,26,25,24,22,20
  python phase2/rq3_patch.py ... --experiment e3b --n 20 --gen-tokens 200

Reuses helpers from loop_probe.py (same dir).  GPU; stop any vLLM server first.
"""
import argparse
import json
import re

import numpy as np
import torch

import loop_probe
from loop_probe import build_full, c2t, get_model, think_dir, toks_of


def parse_layers(s):
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def hook_names(layers):
    return [f"blocks.{L}.hook_mlp_out" for L in layers]


def layer_of(name):
    return int(name.split(".")[1])


# --------------------------------------------------------------------- forward
def think_logit(model, toks, pos, tid, fwd_hooks=()):
    with torch.no_grad():
        logits = model.run_with_hooks(
            toks.unsqueeze(0), fwd_hooks=list(fwd_hooks), return_type="logits")
    v = float(logits[0, pos, tid])
    del logits
    return v


def cache_mlp_at(model, toks, pos, layers):
    want = set(hook_names(layers))
    with torch.no_grad():
        _, cache = model.run_with_cache(
            toks.unsqueeze(0), names_filter=lambda n: n in want)
    out = {L: cache[f"blocks.{L}.hook_mlp_out"][0, pos].detach().float().clone()
           for L in layers}
    del cache
    return out


# --------------------------------------------------------------------- patches
def patch_pos(pos, mode, donor=None, uhat=None):
    """fwd hook: rewrite blocks.L.hook_mlp_out at ONE token index `pos`.
       replace / add / zero / proj_set (match donor only along uhat) / proj_zero.
    """
    def hook(act, hook):
        L = layer_of(hook.name)
        v = act[0, pos].float()
        if mode == "replace":
            new = donor[L]
        elif mode == "add":
            new = v + donor[L]
        elif mode == "zero":
            new = torch.zeros_like(v)
        elif mode == "proj_set":
            new = v + ((donor[L] @ uhat) - (v @ uhat)) * uhat
        elif mode == "proj_zero":
            new = v - (v @ uhat) * uhat
        else:
            raise ValueError(mode)
        act[0, pos] = new.to(act.dtype)
        return act
    return hook


def clamp_last(mode, donor=None, uhat=None):
    """fwd hook for generation: same edit but always at the final position.
    proj_set / proj_zero touch ONLY the </think>-direction component of the
    output, leaving the rest of the vector untouched -- the surgical variants
    used by --surgical to test whether the *specific* stop-direction write
    (not a side effect of overwriting the whole MLP output) drives the flip."""
    def hook(act, hook):
        L = layer_of(hook.name)
        v = act[0, -1].float()
        if mode == "replace":
            new = donor[L]
        elif mode == "zero":
            new = torch.zeros_like(v)
        elif mode == "proj_set":
            d = donor[L]
            new = v + ((d @ uhat) - (v @ uhat)) * uhat
        elif mode == "proj_zero":
            new = v - (v @ uhat) * uhat
        else:
            raise ValueError(mode)
        act[0, -1] = new.to(act.dtype)
        return act
    return hook


def ka_of(model, rec):
    """(aligned_toks[:ka+1], catch_toks[:ka+1], ka) or None."""
    fa, _, _, ca = build_full(model, rec["aligned"])
    if ca is None:
        return None
    ta = toks_of(model, fa)
    ka = c2t(model, fa, ca) - 1
    if not (0 < ka < len(ta)):
        return None
    fc, _, _, _ = build_full(model, rec["catch"])
    tc = toks_of(model, fc)
    if ka >= len(tc):
        return None
    return ta[:ka + 1], tc[:ka + 1], ka


# --------------------------------------------------------------------- E3a
def run_e3a(model, recs, args):
    from scipy.stats import wilcoxon
    tid, u = think_dir(model)
    uhat = (u / u.norm()).float()
    comp, ctrl = parse_layers(args.components), parse_layers(args.control_layers)
    allL = sorted(set(comp) | set(ctrl))

    keys = ("la", "lc", "rescue_replace", "rescue_add", "rescue_proj",
            "lesion_zero", "lesion_proj", "ctrl_replace")
    acc = {k: [] for k in keys}
    used = 0
    for i, rec in enumerate(recs):
        got = ka_of(model, rec)
        if got is None:
            continue
        A, C, ka = got
        d_al = cache_mlp_at(model, A, ka, allL)
        d_ca = cache_mlp_at(model, C, ka, allL)
        delta = {L: d_al[L] - d_ca[L] for L in allL}

        acc["la"].append(think_logit(model, A, ka, tid))
        acc["lc"].append(think_logit(model, C, ka, tid))
        acc["rescue_replace"].append(think_logit(model, C, ka, tid,
            [(n, patch_pos(ka, "replace", d_al)) for n in hook_names(comp)]))
        acc["rescue_add"].append(think_logit(model, C, ka, tid,
            [(n, patch_pos(ka, "add", delta)) for n in hook_names(comp)]))
        acc["rescue_proj"].append(think_logit(model, C, ka, tid,
            [(n, patch_pos(ka, "proj_set", d_al, uhat)) for n in hook_names(comp)]))
        acc["lesion_zero"].append(think_logit(model, A, ka, tid,
            [(n, patch_pos(ka, "zero")) for n in hook_names(comp)]))
        acc["lesion_proj"].append(think_logit(model, A, ka, tid,
            [(n, patch_pos(ka, "proj_zero", uhat=uhat)) for n in hook_names(comp)]))
        if ctrl:
            acc["ctrl_replace"].append(think_logit(model, C, ka, tid,
                [(n, patch_pos(ka, "replace", d_al)) for n in hook_names(ctrl)]))
        used += 1
        if used % 10 == 0:
            print(f"  e3a {used}/{len(recs)}")

    a = {k: np.array(v) for k, v in acc.items() if v}
    n = len(a["la"])
    gap = float((a["la"] - a["lc"]).mean())
    print(f"\nE3a  static patch at ka   (n={n})   components = MLP {comp}"
          f"{'   control = MLP ' + str(ctrl) if ctrl else ''}\n")
    print(f"  logit(</think>)  aligned      {a['la'].mean():+7.3f}")
    print(f"  logit(</think>)  catch        {a['lc'].mean():+7.3f}")
    print(f"  aligned - catch gap          {gap:+7.3f}\n")

    def frac_recov(k):
        return 100 * (a[k].mean() - a["lc"].mean()) / gap

    def frac_remov(k):
        return 100 * (a["la"].mean() - a[k].mean()) / gap

    print(f"  RESCUE  catch + aligned MLP (replace) {a['rescue_replace'].mean():+7.3f}"
          f"   {frac_recov('rescue_replace'):5.0f}% of gap recovered")
    print(f"  RESCUE  catch + (aligned-catch) delta {a['rescue_add'].mean():+7.3f}"
          f"   {frac_recov('rescue_add'):5.0f}%")
    print(f"  RESCUE  catch + aligned MLP, proj-u   {a['rescue_proj'].mean():+7.3f}"
          f"   {frac_recov('rescue_proj'):5.0f}%")
    print(f"  LESION  aligned - MLP (zero)          {a['lesion_zero'].mean():+7.3f}"
          f"   {frac_remov('lesion_zero'):5.0f}% of gap removed")
    print(f"  LESION  aligned - MLP, proj-u only    {a['lesion_proj'].mean():+7.3f}"
          f"   {frac_remov('lesion_proj'):5.0f}%")
    if "ctrl_replace" in a:
        print(f"  CONTROL catch + MLP {ctrl} (replace)  {a['ctrl_replace'].mean():+7.3f}"
              f"   {frac_recov('ctrl_replace'):5.0f}%   (expect ~0)")

    def w(x, y):
        try:
            r = wilcoxon(x, y)
            return f"W={float(r.statistic):.1f}  p={float(r.pvalue):.2e}"
        except Exception as e:  # noqa: BLE001
            return f"(wilcoxon: {e})"
    print(f"\n  rescue_replace vs catch  : {w(a['rescue_replace'], a['lc'])}  (n={n})")
    print(f"  lesion_zero    vs aligned: {w(a['lesion_zero'], a['la'])}  (n={n})")


# --------------------------------------------------------------------- E3b
def donor_mean(model, recs, layers, which="aligned"):
    """Mean mlp_out@ka over problems.  which="aligned" (default) -> the normal
    rescue donor (terminated-trace value).  which="catch" -> catch's OWN value
    at the same index -- clamping catch to this should be a near no-op, the
    sanity control for --donor-mode catch/random (does the donor's CONTENT
    matter, or does any clamp do something)."""
    tot, k = {L: None for L in layers}, 0
    for rec in recs:
        got = ka_of(model, rec)
        if got is None:
            continue
        A, C, ka = got
        src = A if which == "aligned" else C
        d = cache_mlp_at(model, src, ka, layers)
        for L in layers:
            tot[L] = d[L] if tot[L] is None else tot[L] + d[L]
        k += 1
    return {L: tot[L] / k for L in layers}, k


def random_donor(dm_aligned, dm_catch, layers, seed=0):
    """catch-mean + a RANDOM-direction vector matched in magnitude to the real
    (aligned - catch) delta per layer.  Isolates direction: same-size nudge,
    wrong direction.  If this ALSO flips the outcome, the effect is "any
    sufficiently large perturbation", not this specific signal."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for L in layers:
        delta = dm_aligned[L] - dm_catch[L]
        norm = delta.norm()
        direction = torch.randn(delta.shape, generator=g).to(delta)
        direction = direction / direction.norm()
        out[L] = dm_catch[L] + norm * direction
    return out


def max_run(text):
    parts = [re.sub(r"\s+", " ", s).strip().lower()
             for s in re.split(r"(?<=[.\n])", text) if len(s.strip()) > 15]
    best = run = 1
    for x, y in zip(parts, parts[1:]):
        run = run + 1 if x == y else 1
        best = max(best, run)
    return best


def generate(model, prefix, tid, max_new, fwd_hooks=(), post_think_tokens=0,
             temperature=0.0, rng=None):
    """Decode, no KV cache -- full forward pass each step on a growing
    sequence.  Every step is a different (batch, seq_len, vocab=~152k) logits
    tensor, so the CUDA allocator can't reuse the previous step's block; left
    unchecked this fragments badly over many steps x conditions x problems.
    Free explicitly and periodically empty_cache to keep that bounded.

    IMPORTANT: the clamp only ever ran up to and including the </think> token
    (`stop` marks that offset). Emitting </think> is not the same as producing
    an answer -- with post_think_tokens > 0, once </think> fires the fwd_hooks
    are dropped and generation continues *unclamped* for a few more tokens, so
    you can read whether a real answer follows, not just that the stop token did.

    temperature=0 (default): greedy argmax, deterministic -- same result every
    run, as used throughout this experiment so far. temperature>0: sample
    instead of taking the argmax at every step (both the clamped phase and the
    post-think tail) -- results become stochastic, so pass `rng` (a
    torch.Generator seeded once by the caller) to make a given run
    reproducible; it does NOT make one run representative of the distribution."""
    def pick(row):
        if temperature and temperature > 0:
            probs = torch.softmax(row.float() / temperature, dim=-1)
            return int(torch.multinomial(probs, 1, generator=rng))
        return int(row.argmax())

    out = prefix.clone()
    stop = None
    with model.hooks(fwd_hooks=list(fwd_hooks)):
        for i in range(max_new):
            with torch.no_grad():
                logits = model(out.unsqueeze(0))
            nxt = pick(logits[0, -1])
            out = torch.cat([out, out.new_tensor([nxt])])
            del logits
            if nxt == tid:
                stop = i
                break
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
    # only if </think> actually fired (stop is not None) -- generate the natural,
    # UNCLAMPED continuation so it can be read as an actual answer, not just a
    # stop token.  If the budget ran out with no </think>, there's nothing to
    # continue past.
    if stop is not None and post_think_tokens:
        eos_id = model.tokenizer.eos_token_id
        for j in range(post_think_tokens):
            with torch.no_grad():
                logits = model(out.unsqueeze(0))
            nxt = pick(logits[0, -1])
            out = torch.cat([out, out.new_tensor([nxt])])
            del logits
            if eos_id is not None and nxt == eos_id:
                break              # real end of turn -- do NOT run past it into a new one
            if (j + 1) % 10 == 0:
                torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    txt = model.tokenizer.decode(out[len(prefix):])
    return stop, max_run(txt), txt


def summarise(tag, rows):
    if not rows:
        print(f"  {tag:34s}  (no usable problems)")
        return
    stop = [r for r in rows if r[0] is not None]
    offs = sorted(r[0] for r in stop)
    reps = sorted(r[1] for r in rows)
    med = lambda z: z[len(z) // 2] if z else float("nan")
    print(f"  {tag:34s}  n={len(rows):3d}  </think> emitted {len(stop):3d}"
          f" ({100*len(stop)/len(rows):3.0f}%)   median stop@{med(offs) if offs else '-':>4}"
          f"   median max-repeat {med(reps)}")


def run_e3b(model, recs, args):
    tid, u = think_dir(model)
    uhat = (u / u.norm()).float()
    comp, ctrl = parse_layers(args.components), parse_layers(args.control_layers)
    allL = sorted(set(comp) | set(ctrl))
    rescue_mode, lesion_mode = ("proj_set", "proj_zero") if args.surgical else ("replace", "zero")

    print(f"E3b  clamped generation from ka   (gen_tokens={args.gen_tokens}, "
          f"post_think_tokens={args.post_think_tokens})")
    print(f"     rescue/lesion components = MLP {comp}"
          f"{'   control = MLP ' + str(ctrl) if ctrl else ''}")
    print(f"     donor-mode={args.donor_mode}   surgical={args.surgical}"
          f"  (rescue={rescue_mode}, lesion={lesion_mode})")
    rng = None
    if args.gen_temperature > 0:
        rng = torch.Generator(device=args.device).manual_seed(args.seed)
        print(f"     gen_temperature={args.gen_temperature}  seed={args.seed}  "
              f"(STOCHASTIC -- this run is one sample, not a rate; reproducible "
              f"only because of the fixed seed)")
    print()

    # CONTROL always uses the standard aligned donor, regardless of --donor-mode,
    # so it stays a stable baseline across donor-mode experiments.
    dm_al, k = donor_mean(model, recs, allL, which="aligned")
    print(f"  aligned donor = mean aligned mlp_out@ka over {k} problems")
    dm = dm_al
    if args.donor_mode != "aligned":
        dm_ca, _ = donor_mean(model, recs, allL, which="catch")
        if args.donor_mode == "catch":
            dm = dm_ca
            print("  RESCUE donor = catch's OWN mean mlp_out@ka (sanity control: "
                  "clamping to the already-quiet value should be close to a no-op)")
        elif args.donor_mode == "random":
            dm = random_donor(dm_al, dm_ca, allL, seed=args.seed)
            print("  RESCUE donor = catch-mean + a RANDOM-direction vector, magnitude-matched "
                  "to the real (aligned-catch) delta (isolates direction from magnitude)")
    print()

    rows = {"base_c": [], "resc": [], "ctlc": [], "base_a": [], "lesa": []}
    transcripts = []
    for i, rec in enumerate(recs):
        got = ka_of(model, rec)
        if got is None:
            continue
        A, C, ka = got
        conds = [
            ("base_c", "catch_unpatched", C, ()),
            ("resc", "catch_rescue", C, [(n, clamp_last(rescue_mode, dm, uhat)) for n in hook_names(comp)]),
        ]
        if ctrl:
            conds.append(("ctlc", "catch_control", C,
                          [(n, clamp_last("replace", dm_al)) for n in hook_names(ctrl)]))
        conds += [
            ("base_a", "aligned_unpatched", A, ()),
            ("lesa", "aligned_lesion", A, [(n, clamp_last(lesion_mode, uhat=uhat)) for n in hook_names(comp)]),
        ]
        for key, name, prefix, hooks in conds:
            stop, reps, txt = generate(model, prefix, tid, args.gen_tokens, hooks,
                                       post_think_tokens=args.post_think_tokens,
                                       temperature=args.gen_temperature, rng=rng)
            rows[key].append((stop, reps, txt))
            if args.save_transcripts:
                transcripts.append(dict(source_idx=rec["source_idx"], condition=name,
                                        stop=stop, max_repeat=reps, text=txt))
        torch.cuda.empty_cache()          # 5 growing-sequence generations/problem
        print(f"  e3b {i + 1}/{len(recs)}   (no KV cache -- each problem is 5 "
              f"full generations, can take minutes)")

    if args.save_transcripts:
        with open(args.save_transcripts, "w") as f:
            for t in transcripts:
                f.write(json.dumps(t) + "\n")
        print(f"\n  wrote {len(transcripts)} transcripts -> {args.save_transcripts}")

    print()
    summarise("catch  unpatched", rows["base_c"])
    summarise(f"catch  RESCUE clamp MLP {comp}", rows["resc"])
    if rows["ctlc"]:
        summarise(f"catch  CONTROL clamp MLP {ctrl}", rows["ctlc"])
    summarise("aligned unpatched", rows["base_a"])
    summarise(f"aligned LESION zero MLP {comp}", rows["lesa"])


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="phase2/loop_pairs.jsonl")
    ap.add_argument("--experiment", default="e3a", choices=["e3a", "e3b"])
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=0, help="limit problems (0 = all)")
    ap.add_argument("--components", default="27",
                    help="comma list of MLP layers to patch (E2c top set: 27,26,25,24,22,20)")
    ap.add_argument("--control-layers", default="15",
                    help="comma list of late MLP layers NOT in the E2c set (control)")
    ap.add_argument("--gen-tokens", type=int, default=200, help="E3b: max tokens to decode")
    ap.add_argument("--raw-think", action="store_true",
                    help="loop_pairs came from a run that kept literal <think> tags")
    ap.add_argument("--surgical", action="store_true",
                    help="E3b: rescue/lesion edit ONLY the </think>-direction component of the "
                         "patched MLPs' output, not the whole vector (proj_set / proj_zero)")
    ap.add_argument("--donor-mode", default="aligned", choices=["aligned", "catch", "random"],
                    help="E3b RESCUE donor: aligned (default, the real fix) | catch (sanity "
                         "control -- clamp to catch's own value, should be ~a no-op) | random "
                         "(magnitude-matched random direction -- isolates direction from size). "
                         "CONTROL always uses the aligned donor regardless of this flag.")
    ap.add_argument("--seed", type=int, default=0, help="--donor-mode random: RNG seed")
    ap.add_argument("--save-transcripts", default=None,
                    help="E3b: write every generated continuation to this JSONL path, for "
                         "reading whether RESCUE produces coherent text or just breaks generation")
    ap.add_argument("--gen-temperature", type=float, default=0.0,
                    help="E3b: >0 samples instead of greedy argmax at every decode step "
                         "(both the clamped phase and the post-think tail). 0 (default) = "
                         "greedy, deterministic, same behaviour as every run so far. Makes "
                         "results stochastic -- use --seed for a reproducible single sample, "
                         "it does not make one run a reliable rate.")
    ap.add_argument("--post-think-tokens", type=int, default=0,
                    help="E3b: once </think> fires, drop the clamp and generate this many more "
                         "tokens UNCLAMPED, so the saved transcript includes a real answer "
                         "attempt, not just the stop token. 0 (default) = old behaviour, stop "
                         "exactly at </think>. ~30-50 is enough to see an 'Answer: (X)' line.")
    args = ap.parse_args()
    loop_probe.RAW_THINK = args.raw_think

    recs = [json.loads(l) for l in open(args.pairs)]
    if args.n:
        recs = recs[: args.n]
    print(f"{len(recs)} pairs; loading {args.model} on {args.device} ...")
    model = get_model(args.model, args.device, args.dtype)
    model.eval()
    model.requires_grad_(False)

    print("\n" + "=" * 74 + f"\n{args.experiment.upper()}\n" + "=" * 74)
    (run_e3a if args.experiment == "e3a" else run_e3b)(model, recs, args)


if __name__ == "__main__":
    main()
