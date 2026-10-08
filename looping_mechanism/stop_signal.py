"""
Stop-signal mechanism on matched loop/finish pairs (looping_mechanism/data/pairs.jsonl):
is the looping run's </think> signal withheld at its commit point, by which
components, and does restoring the late MLPs bring it back?  Same analysis as the
greedy catch experiment (phase2 E2b / E2c / E3a), run identically on every group:
controlled (sampled), controlled (greedy reference) and natural (sampled).

Positions (exact, from the tokenizer's character offsets):
  finish@stop     token right before </think> in the FINISHED run
  loop@commit     token right before the doubt phrase that follows the looping
                  run's commit sentence ("... so P(5) = 15." | "Wait, ...")
Both are "the model just stated its answer; next token: stop or doubt?".

Measured at each position (R1-Distill-Qwen-7B, phase2 weight processing):
  lens    residual stream after every layer projected on the </think>
          unembedding (ln_final applied); last value = logit(</think>)
  DLA     direct logit attribution of each attention layer and MLP to </think>
  top1    is </think> the model's most likely next token here?
  patch   at loop@commit, replace MLPs --components with
            own          this pair's finish@stop values
            donor_greedy mean aligned mlp_out before </think>, greedy catch experiment
            donor_ctrl   mean finish@stop mlp_out over the sampled ctrl_cross pairs
            control      own values, but on MLP --control-layers
          -> % of the finish-loop logit gap recovered

"strict" = only pairs whose finish@stop has </think> as the top-1 token (a
confident stop).

    python looping_mechanism/stop_signal.py            # -> looping_mechanism/results/...
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "phase2"))
import loop_probe  # noqa: E402
from loop_probe import get_model, think_dir  # noqa: E402
from rq3_patch import hook_names, parse_layers  # noqa: E402
from rq3_resolve import donor_mean, prefill  # noqa: E402

GROUPS = ["greedy_ref", "ctrl_cross", "ctrl_same", "nat_aqua", "nat_logiqa", "nat_math500"]
LABEL = {"greedy_ref": "controlled, greedy (original pairs)",
         "ctrl_cross": "controlled: catch loop vs aligned finish",
         "ctrl_same": "controlled: catch loop vs catch finish",
         "nat_aqua": "natural AQuA", "nat_logiqa": "natural LogiQA",
         "nat_math500": "natural MATH-500"}
WANT = ("hook_resid_pre", "hook_resid_post", "hook_attn_out", "hook_mlp_out")


def chat_prefix(model, prompt):
    s = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    if s.rstrip().endswith("<think>"):
        return s if s.endswith("\n") else s + "\n"
    return s + "<think>\n"


def encode(model, text):
    enc = model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return torch.tensor(enc["input_ids"], device=model.cfg.device), enc["offset_mapping"]


def token_at(offsets, char):
    """index of the first token that starts at or after `char`"""
    return next((i for i, (a, b) in enumerate(offsets) if a >= char or b > char), None)


def positions(model, pair, tid):
    fin = pair["finish"]
    pre_f = chat_prefix(model, fin["prompt"])
    # vLLM's reasoning parser returns exactly the text between <think> and </think>,
    # and the answer after it, so the generated sequence is reasoning + "</think>" +
    # completion.  (Adding a newline here merges into e.g. ".\n\n" -- a token the
    # model never produced before stopping, after which it expects "Wait".)
    full_f = pre_f + fin["reasoning"] + "</think>" + (fin["completion"] or "")
    tf, of = encode(model, full_f)
    j = token_at(of, len(pre_f) + len(fin["reasoning"]))              # the </think> token
    if j is None or int(tf[j]) != tid:
        return None
    lp = pair["loop"]
    pre_l = chat_prefix(model, lp["prompt"])
    tl, ol = encode(model, pre_l + lp["reasoning"])
    k = token_at(ol, len(pre_l) + lp["commit_off"])                   # the doubt token ("Wait")
    if k is None or k < 2:
        return None
    return tf, j - 1, tl, k - 1


@torch.inference_mode()
def measure(model, toks, pos, u, tid, patches=None):
    kv = prefill(model, toks[:pos + 1])
    kv.freeze()
    last = toks[pos:pos + 1].unsqueeze(0)
    logits, cache = model.run_with_cache(
        last, past_kv_cache=kv,
        names_filter=lambda n: n.endswith(WANT) or n == "ln_final.hook_scale")
    nL = model.cfg.n_layers
    acc = cache.accumulated_resid(layer=-1, incl_mid=False, pos_slice=-1, apply_ln=True)
    acc = acc.float().reshape(acc.shape[0], -1, acc.shape[-1])[:, -1]
    scale = cache["ln_final.hook_scale"][0, -1].float()
    row = logits[0, -1].float()
    out = dict(lens=(acc @ u).cpu().numpy(),
               attn=np.array([float((cache["attn_out", L][0, -1].float() / scale) @ u) for L in range(nL)]),
               mlp=np.array([float((cache["mlp_out", L][0, -1].float() / scale) @ u) for L in range(nL)]),
               logit=float(row[tid]), logprob=float(torch.log_softmax(row, -1)[tid]),
               top1=int(row.argmax()) == tid,
               next_tok=model.tokenizer.decode([int(toks[pos + 1])]) if pos + 1 < len(toks) else "")
    vec = {L: cache["mlp_out", L][0, -1].float().clone() for L in range(nL)}
    del cache, logits
    for name, donor in (patches or {}).items():
        def hook(act, hook, donor=donor):
            act[0, -1] = donor[int(hook.name.split(".")[1])].to(act.dtype)
            return act
        lg = model.run_with_hooks(last, past_kv_cache=kv,
                                  fwd_hooks=[(n, hook) for n in hook_names(list(donor))])
        out[f"patch_{name}"] = float(lg[0, -1, tid])
    del kv
    torch.cuda.empty_cache()
    return out, vec


def summarise(rows, comp, ctrl, nL, model_name="DeepSeek-R1-Distill-Qwen-7B"):
    from scipy.stats import spearmanr, wilcoxon
    S = collections.OrderedDict()
    for g in GROUPS:
        for strict in (False, True):
            R = [r for r in rows if r["group"] == g and (r["finish"]["top1"] or not strict)]
            if len(R) < 5:
                continue
            fl = np.array([r["finish"]["logit"] for r in R])
            ll = np.array([r["loop"]["logit"] for r in R])
            gap = fl - ll
            dm = np.mean([r["finish"]["mlp"] - r["loop"]["mlp"] for r in R], 0)
            da = np.mean([r["finish"]["attn"] - r["loop"]["attn"] for r in R], 0)
            rec = {k: float(np.mean([r["loop"][f"patch_{k}"] - r["loop"]["logit"] for r in R]) / gap.mean())
                   for k in ("own", "donor_greedy", "donor_ctrl", "control") if f"patch_{k}" in R[0]["loop"]}
            try:
                p = float(wilcoxon(fl, ll).pvalue)
            except ValueError:
                p = float("nan")
            S[(g, strict)] = dict(
                n=len(R), finish_logit=float(fl.mean()), loop_logit=float(ll.mean()),
                gap=float(gap.mean()), gap_pos=float((gap > 0).mean()), p=p,
                finish_top1=float(np.mean([r["finish"]["top1"] for r in R])),
                loop_top1=float(np.mean([r["loop"]["top1"] for r in R])),
                mlp_delta=dm, attn_delta=da, mlp_share=float(dm.sum() / (dm.sum() + da.sum())),
                top_mlps=[int(x) for x in np.argsort(-dm)[:6]],
                comp_share=float(dm[comp].sum() / gap.mean()), recover=rec,
                lens_finish=np.mean([r["finish"]["lens"] for r in R], 0),
                lens_loop=np.mean([r["loop"]["lens"] for r in R], 0))
    ref = S.get(("greedy_ref", False)) or S.get(("ctrl_cross", False))
    ref_name = "greedy controlled pairs" if ("greedy_ref", False) in S else "controlled catch-vs-aligned pairs"
    for s in S.values():
        s["rho_vs_greedy"] = float(spearmanr(s["mlp_delta"], ref["mlp_delta"])[0]) if ref else float("nan")

    L = ["# The </think> stop signal: controlled vs natural loops", "",
         f"{model_name}. Each pair: one run that FINISHED and one that LOOPED.",
         "finish@stop = token before </think> in the finished run; loop@commit = token",
         "before the doubt phrase after the looping run commits to its answer. Controlled",
         "and natural pairs are sampled (T=0.6, boxed prompt) unless marked greedy.",
         "`strict` keeps pairs whose finished run had </think> as its top-1 token.", "",
         "## 1. Is </think> suppressed at the looping run's commit point?", "",
         "| group | pairs | logit(</think>) finish | loop | **gap** | pairs gap>0 | "
         "</think> top-1: finish / loop | Wilcoxon p |", "|---|---|---|---|---|---|---|---|"]
    for (g, st), s in S.items():
        L.append(f"| {LABEL[g]}{' (strict)' if st else ''} | {s['n']} | {s['finish_logit']:+.2f} | "
                 f"{s['loop_logit']:+.2f} | **{s['gap']:+.2f}** | {100 * s['gap_pos']:.0f}% | "
                 f"{100 * s['finish_top1']:.0f}% / {100 * s['loop_top1']:.0f}% | {s['p']:.1e} |")
    L += ["", "## 2. Which components carry the gap?", "",
          f"MLP share = MLP part of the gap (vs attention). Stop-signal MLPs: {comp}. rho = "
          f"Spearman correlation of the {nL} per-MLP gap contributions with the {ref_name}.", "",
          "| group | MLP share | top-6 MLPs by gap | share of gap in MLPs "
          f"{comp} | rho vs reference |", "|---|---|---|---|---|"]
    for (g, st), s in S.items():
        L.append(f"| {LABEL[g]}{' (strict)' if st else ''} | {100 * s['mlp_share']:.0f}% | "
                 f"{s['top_mlps']} | {100 * s['comp_share']:.0f}% | {s['rho_vs_greedy']:+.2f} |")
    L += ["", "## 3. Does restoring the late MLPs restore the stop logit?", "",
          f"% of the finish-loop gap recovered by replacing MLPs {comp} at loop@commit.", "",
          "| group | own finish values | sampled-controlled donor | "
          f"control: MLP {ctrl} |", "|---|---|---|---|"]
    for (g, st), s in S.items():
        r = s["recover"]
        L.append(f"| {LABEL[g]}{' (strict)' if st else ''} | {100 * r['own']:.0f}% | "
                 f"{100 * r['donor_ctrl']:.0f}% | {100 * r['control']:.0f}% |")
    return S, L


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(HERE, "data", "pairs.jsonl"))
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--components", default="27,26,25,24,22,20",
                    help="stop-signal MLPs to patch, or 'auto' = the 6 MLPs with the largest gap "
                         "contribution on the controlled catch-vs-aligned pairs (found first)")
    ap.add_argument("--control-layers", default="15", help="control MLP(s), or 'auto' = middle layer")
    ap.add_argument("--n", type=int, default=0, help="limit pairs per group (0 = all)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results", "r1-distill-qwen-7b"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    loop_probe.RAW_THINK = False

    model = get_model(args.model, "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tid, u = think_dir(model)
    u = u.float()
    nL = model.cfg.n_layers
    ctrl = [nL // 2] if args.control_layers == "auto" else parse_layers(args.control_layers)
    comp = None if args.components == "auto" else parse_layers(args.components)
    greedy_donor = None
    if "R1-Distill-Qwen-7B" in args.model and comp is not None:   # the original greedy pairs
        greedy_donor, k = donor_mean(model, [json.loads(l) for l in open(
            os.path.join(ROOT, "phase2", "loop_pairs.jsonl"))], sorted(set(comp) | set(ctrl)))
        print(f"greedy donor: mean aligned mlp_out before </think> over {k} greedy pairs")

    by = collections.defaultdict(list)
    for p in map(json.loads, open(args.pairs)):
        by[p["group"]].append(p)
    work = []
    for g in GROUPS:
        P = by.get(g, [])
        work += P[: args.n] if args.n else P

    # pass 1: every finished run at its stop point (also gives the sampled ctrl donor)
    fin, skipped = {}, collections.Counter()
    for i, p in enumerate(work):
        pos = positions(model, p, tid)
        if pos is None:
            skipped[p["group"]] += 1
            continue
        tf, ka, tl, kc = pos
        m, vec = measure(model, tf, ka, u, tid)
        fin[p["id"]] = (pos, m, {L: v.cpu() for L, v in vec.items()})
        if (i + 1) % 50 == 0:
            print(f"  pass 1 (finished runs): {i + 1}/{len(work)}")
    if comp is None:            # find the stop MLPs from the controlled catch-vs-aligned pairs
        deltas = []
        for p in work:
            if p["group"] == "ctrl_cross" and p["id"] in fin:
                (tf, ka, tl, kc), fm, _ = fin[p["id"]]
                deltas.append(fm["mlp"] - measure(model, tl, kc, u, tid)[0]["mlp"])
        dm = np.mean(deltas, 0)
        comp = sorted(int(x) for x in np.argsort(-dm)[:6])
        ctrl = [c for c in ctrl if c not in comp] or [nL // 2 - 1]
        print(f"auto stop MLPs (top-6 gap contribution on {len(deltas)} controlled pairs): {comp}; "
              f"control: {ctrl}")
    cc = [v[2] for pid, v in fin.items() if pid.startswith("ctrl_") and not pid.startswith("ctrlsame")]
    ctrl_donor = {L: torch.stack([c[L] for c in cc]).mean(0).cuda() for L in comp}
    print(f"sampled ctrl donor: mean over {len(cc)} ctrl_cross finished runs; "
          f"skipped (no clean </think>/commit position): {dict(skipped)}")

    # pass 2: every looping run at its commit point, with the patches
    rows = []
    for i, p in enumerate(work):
        if p["id"] not in fin:
            continue
        (tf, ka, tl, kc), fm, fvec = fin[p["id"]]
        patches = {"own": {L: fvec[L].cuda() for L in comp}, "donor_ctrl": ctrl_donor,
                   "control": {L: fvec[L].cuda() for L in ctrl}}
        if greedy_donor is not None:
            patches["donor_greedy"] = {L: greedy_donor[L] for L in comp}
        lm, _ = measure(model, tl, kc, u, tid, patches=patches)
        rows.append(dict(group=p["group"], id=p["id"], finish=fm, loop=lm))
        if (i + 1) % 50 == 0:
            print(f"  pass 2 (looping runs): {i + 1}/{len(work)}")

    S, L = summarise(rows, comp, ctrl, nL, args.model.split("/")[-1])
    json.dump(dict(components=comp, control=ctrl), open(os.path.join(args.out_dir, "components.json"), "w"))
    ser = lambda x: x.tolist() if isinstance(x, np.ndarray) else x  # noqa: E731
    json.dump({f"{g}{'/strict' if st else ''}": {k: ser(v) for k, v in s.items()}
               for (g, st), s in S.items()},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    with open(os.path.join(args.out_dir, "per_pair.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: ({kk: ser(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
                                for k, v in r.items()}) + "\n")
    open(os.path.join(args.out_dir, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out_dir}/summary.md, summary.json, per_pair.jsonl")


if __name__ == "__main__":
    main()
