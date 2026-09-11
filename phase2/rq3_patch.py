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
    """fwd hook for generation: same edit but always at the final position."""
    def hook(act, hook):
        L = layer_of(hook.name)
        v = act[0, -1].float()
        if mode == "replace":
            new = donor[L]
        elif mode == "zero":
            new = torch.zeros_like(v)
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
def donor_mean(model, recs, layers):
    tot, k = {L: None for L in layers}, 0
    for rec in recs:
        got = ka_of(model, rec)
        if got is None:
            continue
        A, _, ka = got
        d = cache_mlp_at(model, A, ka, layers)
        for L in layers:
            tot[L] = d[L] if tot[L] is None else tot[L] + d[L]
        k += 1
    return {L: tot[L] / k for L in layers}, k


def max_run(text):
    parts = [re.sub(r"\s+", " ", s).strip().lower()
             for s in re.split(r"(?<=[.\n])", text) if len(s.strip()) > 15]
    best = run = 1
    for x, y in zip(parts, parts[1:]):
        run = run + 1 if x == y else 1
        best = max(best, run)
    return best


def generate(model, prefix, tid, max_new, fwd_hooks=()):
    """Greedy decode, no KV cache -- full forward pass each step on a growing
    sequence.  Every step is a different (batch, seq_len, vocab=~152k) logits
    tensor, so the CUDA allocator can't reuse the previous step's block; left
    unchecked this fragments badly over many steps x conditions x problems.
    Free explicitly and periodically empty_cache to keep that bounded."""
    out = prefix.clone()
    stop = None
    with model.hooks(fwd_hooks=list(fwd_hooks)):
        for i in range(max_new):
            with torch.no_grad():
                logits = model(out.unsqueeze(0))
            nxt = int(logits[0, -1].argmax())
            out = torch.cat([out, out.new_tensor([nxt])])
            del logits
            if nxt == tid:
                stop = i
                break
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    txt = model.tokenizer.decode(out[len(prefix):])
    return stop, max_run(txt)


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
    tid, _ = think_dir(model)
    comp, ctrl = parse_layers(args.components), parse_layers(args.control_layers)
    print(f"E3b  clamped generation from ka   (gen_tokens={args.gen_tokens})")
    print(f"     rescue/lesion components = MLP {comp}"
          f"{'   control = MLP ' + str(ctrl) if ctrl else ''}\n")

    dm, k = donor_mean(model, recs, sorted(set(comp) | set(ctrl)))
    print(f"  donor = mean aligned mlp_out@ka over {k} problems\n")

    base_c, resc, ctlc, base_a, lesa = [], [], [], [], []
    for i, rec in enumerate(recs):
        got = ka_of(model, rec)
        if got is None:
            continue
        A, C, ka = got
        base_c.append(generate(model, C, tid, args.gen_tokens))
        resc.append(generate(model, C, tid, args.gen_tokens,
            [(n, clamp_last("replace", dm)) for n in hook_names(comp)]))
        if ctrl:
            ctlc.append(generate(model, C, tid, args.gen_tokens,
                [(n, clamp_last("replace", dm)) for n in hook_names(ctrl)]))
        base_a.append(generate(model, A, tid, args.gen_tokens))
        lesa.append(generate(model, A, tid, args.gen_tokens,
            [(n, clamp_last("zero")) for n in hook_names(comp)]))
        torch.cuda.empty_cache()          # 5 growing-sequence generations/problem
        if (i + 1) % 5 == 0:
            print(f"  e3b {i + 1}/{len(recs)}")

    print()
    summarise("catch  unpatched", base_c)
    summarise(f"catch  RESCUE clamp MLP {comp}", resc)
    if ctlc:
        summarise(f"catch  CONTROL clamp MLP {ctrl}", ctlc)
    summarise("aligned unpatched", base_a)
    summarise(f"aligned LESION zero MLP {comp}", lesa)


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
