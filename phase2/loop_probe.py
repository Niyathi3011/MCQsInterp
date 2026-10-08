"""
RQ2 mechanistic experiments on R1 non-termination.  See phase2/RQ2_DESIGN.md.

  E2a  linear probe for "no correct option exists"   (aligned+swap  vs  catch)
  E2b  logit lens on the </think> token              (catch  vs  matched aligned)
  E2c  direct logit attribution onto </think>        (which components suppress it)

  python phase2/loop_probe.py --pairs phase2/loop_pairs.jsonl --experiment all \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --n 12

Iterate with --n 12 first; then drop --n for the full set.  Needs a GPU
(--device cpu works for the 1.5B, slowly).  Stop any vLLM server first.
"""
import argparse
import collections
import json

import numpy as np
import torch

# Set True (via --raw-think) when loop_pairs.jsonl was built from a run that kept
# the literal <think>...</think> tags (run_model.py against a server with NO
# --reasoning-parser).  Then `reasoning` / `completion` already carry the exact
# whitespace around </think>, so build_full must NOT synthesise "\n</think>\n\n".
RAW_THINK = False


# ----------------------------------------------------------------------------- model
def get_model(name, device, dtype, process=True, n_ctx=None):
    """Equivalent of HookedTransformer.from_pretrained (default processing) for
    RMSNorm models, without its RAM peak.  TransformerLens 3.8 upcasts the whole
    state dict to fp32 and deep-copies it at every processing step: ~60 GB for a
    7B model, OOM-killed in a 62 GB container.  Instead: load unprocessed (its
    no-op fp32 round-trip skipped), then apply the same processing TL applies to
    RMSNorm models in place on the GPU, one tensor at a time in fp32
    (see _process_rms_inplace)."""
    from transformer_lens import HookedTransformer
    import transformer_lens.loading_from_pretrained as L
    from transformer_lens.weight_processing import ProcessWeights
    if name not in L.OFFICIAL_MODEL_NAMES:
        L.OFFICIAL_MODEL_NAMES.append(name)          # R1-distill isn't registered
    orig = ProcessWeights.__dict__["process_weights"]
    ProcessWeights.process_weights = staticmethod(lambda sd, *a, **kw: sd)
    try:
        # TransformerLens caps Qwen3's context at 2,048 tokens, and extending the rotary
        # table on the fly breaks under the KV cache; our traces run to ~12k tokens
        n_ctx = n_ctx or (32768 if "qwen3" in name.lower() else None)
        kw = {} if n_ctx is None else {"n_ctx": n_ctx}
        model = HookedTransformer.from_pretrained_no_processing(name, dtype=dtype,
                                                                device=device, **kw)
    finally:
        ProcessWeights.process_weights = orig
    assert model.cfg.normalization_type == "RMS", model.cfg.normalization_type
    if process:
        _process_rms_inplace(model)
    _mlp_weights_linear_layout(model)
    return model


@torch.no_grad()
def _mlp_weights_linear_layout(model):
    """GatedMLP.forward calls F.linear(x, W.T.contiguous()) on W_in/W_gate/W_out
    every forward (to match HF's nn.Linear numerics) -- three weight-sized copies
    per layer, ~85% of a KV-cached decode step.  Store each W as the transpose of
    a contiguous (out, in) buffer, so W.T.contiguous() is a no-op.  Same values,
    same shape, same F.linear call -> identical numerics, no extra memory."""
    for blk in model.blocks:
        for name in ("W_in", "W_gate", "W_out"):
            p = getattr(blk.mlp, name, None)
            if p is not None:
                p.data = p.data.T.contiguous().T


@torch.no_grad()
def _process_rms_inplace(model):
    """What ProcessWeights.process_weights(fold_ln, center_writing_weights,
    center_unembed, fold_value_biases) does for an RMSNorm model:
      fold_ln         ln1.w into W_Q/W_K/W_V, ln2.w into W_in/W_gate,
                      ln_final.w into W_U; then those norm weights = 1
      center_writing  skipped by TL for RMSNorm
      center_unembed  W_U -= mean over vocab, b_U -= mean
      fold_v_biases   b_O += sum_h b_V[h] @ W_O[h]; b_V = 0  (GQA: b_V repeated)
    The norm modules stay RMSNorm with weight 1 (TL swaps in RMSNormPre, which
    computes the same thing)."""
    def scale(p, w):
        p.copy_((p.float() * w).to(p.dtype))

    def attr(mod, *names):                   # GQA models store _W_K/_W_V/_b_V
        return next(getattr(mod, n) for n in names if getattr(mod, n, None) is not None)

    cfg = model.cfg
    rep = cfg.n_heads // (cfg.n_key_value_heads or cfg.n_heads)
    for blk in model.blocks:
        a = blk.attn
        w1 = blk.ln1.w.float()
        for W in (a.W_Q, attr(a, "_W_K", "W_K"), attr(a, "_W_V", "W_V")):
            scale(W, w1[None, :, None])
        blk.ln1.w.fill_(1)
        w2 = blk.ln2.w.float()
        for W in (blk.mlp.W_in, getattr(blk.mlp, "W_gate", None)):
            if W is not None:
                scale(W, w2[:, None])
        blk.ln2.w.fill_(1)
        b_V = attr(a, "_b_V", "b_V")
        bv = torch.repeat_interleave(b_V.float(), rep, dim=0) if b_V.shape[0] != cfg.n_heads \
            else b_V.float()
        a.b_O.copy_((a.b_O.float() + (bv[:, :, None] * a.W_O.float()).sum([0, 1]))
                    .to(a.b_O.dtype))
        b_V.zero_()
    W_U = model.unembed.W_U
    folded = W_U.float() * model.ln_final.w.float()[:, None]
    W_U.copy_((folded - folded.mean(-1, keepdim=True)).to(W_U.dtype))
    del folded
    model.ln_final.w.fill_(1)
    b_U = model.unembed.b_U
    b_U.copy_((b_U.float() - b_U.float().mean()).to(b_U.dtype))
    torch.cuda.empty_cache()


def user_prefix(model, piece):
    """chat-templated user turn, ending in '<think>\\n'."""
    if piece.get("option_A") is None:                  # open-ended (stage1) trace
        content = (f"{piece['question']}"
                   "\n\nEnd with a line formatted exactly as: Answer: <number>")
    else:
        content = (f"{piece['question']}\n(A) {piece['option_A']}\n(B) {piece['option_B']}"
                   "\n\nEnd with a line formatted exactly as: Answer: (X)  where X is A or B.")
    s = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    user_char_len = len(s)                            # end of the prompt proper
    if RAW_THINK:
        # keep the chat template's generation prompt verbatim (it ends "<think>\n"
        # for R1-distill); `reasoning` is exactly msg.content, i.e. what the model
        # produced after it.  Only synthesise "<think>" if the template omits it.
        if not s.rstrip().endswith("<think>"):
            s = s + "<think>"
        return s, user_char_len
    if s.rstrip().endswith("<think>"):
        s = s if s.endswith("\n") else s + "\n"
    else:
        s = s + "<think>\n"
    return s, user_char_len


def build_full(model, piece):
    """returns (full_text, user_char_len, reason_char_start, close_char|None)."""
    pre, user_char_len = user_prefix(model, piece)
    reason_char_start = len(pre)
    if piece["truncated"]:
        return pre + piece["reasoning"], user_char_len, reason_char_start, None
    sep = "</think>" if RAW_THINK else "\n</think>\n\n"
    full = pre + piece["reasoning"] + sep + piece["completion"]
    close_char = reason_char_start + len(piece["reasoning"]) + (0 if RAW_THINK else 1)
    return full, user_char_len, reason_char_start, close_char


def c2t(model, text, char_off):
    """token index that char offset `char_off` falls at."""
    return int(model.to_tokens(text[:char_off], prepend_bos=False).shape[1])


def toks_of(model, text):
    return model.to_tokens(text, prepend_bos=False)[0]


def catch_cap(model, full, reason_char_start, loop_onset_off, window):
    if loop_onset_off is None:
        return None
    return c2t(model, full, reason_char_start + loop_onset_off) + window


# ----------------------------------------------------------------------------- E2a
def positions_e2a(model, full, ucl, rcs, piece, close_char, n_tok):
    """Read-out positions (token indices into `full`):
      opt_end  - end of prompt (CONFOUND: catch/aligned option-A text differs here)
      mid      - reason start + 300 tokens (fixed-offset control, expect ~chance)
      concl    - last mention of the gold value inside the solving span
      pre_end  - the token right before the model loops (catch) / stops (aligned)
    """
    t_r0 = c2t(model, full, rcs)
    pos = {"opt_end": c2t(model, full, ucl) - 1, "mid": t_r0 + 300}
    reasoning = piece["reasoning"]
    if piece["truncated"] and piece.get("loop_onset_off") is not None:
        span_end = rcs + piece["loop_onset_off"]
    elif close_char is not None:
        span_end = close_char
    else:
        span_end = rcs + int(len(reasoning) * 0.8)
    pos["pre_end"] = c2t(model, full, span_end) - 1
    gv = str(piece.get("gold_value") or "")
    if gv:
        j = full[rcs:span_end].rfind(gv)
        if j >= 0:
            pos["concl"] = c2t(model, full, rcs + j)
    return {k: v for k, v in pos.items() if 0 <= v < n_tok}


def resid_at(model, toks, positions):
    with torch.no_grad():
        _, cache = model.run_with_cache(
            toks.unsqueeze(0), names_filter=lambda nm: nm.endswith("resid_post"))
    out = {}
    for name, p in positions.items():
        out[name] = torch.stack(
            [cache["resid_post", L][0, p] for L in range(model.cfg.n_layers)]
        ).float().cpu().numpy()                       # (n_layers, d_model)
    del cache
    torch.cuda.empty_cache()
    return out


def run_e2a(model, recs, args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score

    feats = collections.defaultdict(list)             # (layer, posname) -> [(vec, y, grp)]
    for i, rec in enumerate(recs):
        for var, y in (("aligned", 0), ("swap", 0), ("catch", 1)):
            p = rec[var]
            full, ucl, rcs, close = build_full(model, p)
            toks = toks_of(model, full)
            if p["truncated"]:
                cap = catch_cap(model, full, rcs, p["loop_onset_off"], rec["catch_window"])
                if cap:
                    toks = toks[:cap]
            pos = positions_e2a(model, full, ucl, rcs, p, close, len(toks))
            if not pos:
                continue
            for name, arr in resid_at(model, toks, pos).items():
                for L in range(model.cfg.n_layers):
                    feats[(L, name)].append((arr[L], y, rec["source_idx"]))
        if (i + 1) % 10 == 0:
            print(f"  e2a forward {i + 1}/{len(recs)}")

    names = sorted({k[1] for k in feats})
    print(f"\nE2a  held-out balanced accuracy   (rows=layer, cols=position)")
    print("layer  " + "  ".join(f"{n:>9}" for n in names))
    table = {}
    for L in range(model.cfg.n_layers):
        cells = []
        for name in names:
            items = feats.get((L, name), [])
            X = np.stack([v for v, _, _ in items]) if items else None
            if X is None or len({y for _, y, _ in items}) < 2 or len({g for *_, g in items}) < 5:
                cells.append("    -    ")
                continue
            yv = np.array([y for _, y, _ in items])
            gv = np.array([g for *_, g in items])
            accs = []
            for tr, te in GroupKFold(5).split(X, yv, gv):
                clf = LogisticRegression(class_weight="balanced", C=args.C, max_iter=2000)
                clf.fit(X[tr], yv[tr])
                accs.append(balanced_accuracy_score(yv[te], clf.predict(X[te])))
            m = float(np.mean(accs))
            table[(L, name)] = m
            cells.append(f"  {m:5.3f}  ")
        print(f"L{L:2d}    " + "".join(cells))

    # a real *computed* direction is near-chance in the raw embeddings.  Any position
    # already separable at layers 0-2 is reading local token identity, not d_noopt
    # (opt_end -> option-A text; pre_end -> "token before </think>" vs "before a
    # backtrack phrase" == the label).  Auto-exclude those.
    early = collections.defaultdict(list)
    for (L, n), v in table.items():
        if L <= 2:
            early[n].append(v)
    surface = {n for n, vs in early.items() if np.mean(vs) >= 0.70}
    print(f"\n[surface-confounded positions] (bal-acc >= 0.70 at layers 0-2, "
          f"EXCLUDED from d_noopt): {sorted(surface) or 'none'}")

    cand = {k: v for k, v in table.items() if k[1] not in surface and k[1] != "mid"}
    if not cand:
        print("no non-confounded position produced a probe result."); return
    (bl, bn), bm = max(cand.items(), key=lambda kv: kv[1])
    print(f"best non-confounded: layer {bl}, position '{bn}'  bal-acc {bm:.3f}")
    if bm < 0.75:
        print("  -> below 0.75: 'no correct option' is NOT a clean linear direction at\n"
              "     these positions. NOT saving d_noopt. Try the full 83 pairs, span-pooled\n"
              "     residuals, a non-linear probe, or whole-activation patching (RQ3).")
        return
    items = feats[(bl, bn)]
    X = np.stack([v for v, _, _ in items]); yv = np.array([y for _, y, _ in items])
    clf = LogisticRegression(class_weight="balanced", C=args.C, max_iter=2000).fit(X, yv)
    d = clf.coef_[0].astype(np.float32); d /= np.linalg.norm(d)
    np.save(args.dir + "/d_noopt.npy", d)
    json.dump({"layer": bl, "position": bn, "bal_acc": bm, "C": args.C},
              open(args.dir + "/d_noopt.meta.json", "w"), indent=2)
    print(f"saved d_noopt (layer {bl}, {bn}) -> {args.dir}/d_noopt.npy")


# ----------------------------------------------------------------------------- E2b
def think_dir(model):
    ids = model.to_tokens("</think>", prepend_bos=False)[0]
    tid = int(ids[-1])
    return tid, model.W_U[:, tid].detach().float()


def accum_proj(model, toks, pos, u):
    with torch.no_grad():
        _, cache = model.run_with_cache(
            toks.unsqueeze(0),
            names_filter=lambda nm: (nm.endswith("hook_resid_pre")
                                     or nm.endswith("hook_resid_post")
                                     or nm == "ln_final.hook_scale"))
    acc = cache.accumulated_resid(layer=-1, incl_mid=False, pos_slice=pos, apply_ln=True)
    out = (acc.float() @ u).detach().cpu().numpy()    # (n_layers + 1,)
    del cache
    torch.cuda.empty_cache()
    return out


def run_e2b(model, recs, args):
    from scipy.stats import wilcoxon
    tid, u = think_dir(model)
    print(f"E2b  </think> token id = {tid}")
    A, C1, C2 = [], [], []
    for i, rec in enumerate(recs):
        fa, _, ra, ca = build_full(model, rec["aligned"])
        if ca is None:
            continue
        ta = toks_of(model, fa)
        ka = c2t(model, fa, ca) - 1
        if not (0 < ka < len(ta)):
            continue
        fc, _, rc, _ = build_full(model, rec["catch"])
        tc = toks_of(model, fc)
        if ka + 400 >= len(tc):
            continue
        A.append(accum_proj(model, ta[:ka + 1], ka, u))
        C1.append(accum_proj(model, tc[:ka + 1], ka, u))
        C2.append(accum_proj(model, tc[:ka + 401], ka + 400, u))
        if (i + 1) % 10 == 0:
            print(f"  e2b {i + 1}/{len(recs)}  (usable {len(A)})")

    A, C1, C2 = map(lambda z: np.stack(z), (A, C1, C2))
    print(f"\nusable pairs: {len(A)}")
    print(f"{'layer':>5} {'aligned@ka':>12} {'catch@ka':>12} {'catch@ka+400':>14}")
    for L in range(A.shape[1]):
        print(f"{L:5d} {A[:,L].mean():12.3f} {C1[:,L].mean():12.3f} {C2[:,L].mean():14.3f}")
    try:
        w = wilcoxon(A[:, -1], C1[:, -1])
        print(f"\nfinal-layer </think> projection  aligned vs catch@ka:  "
              f"W={float(w.statistic):.1f}  p={float(w.pvalue):.2e}  (n={len(A)})")
    except Exception as e:  # noqa: BLE001
        print(f"(wilcoxon skipped: {e})")


# ----------------------------------------------------------------------------- E2c
def dla(model, toks, pos, u):
    """per-head / per-MLP contribution to the </think> logit at ONE position.
    Computed manually at a single position -- avoids stack_head_results, which
    materialises (n_layers, seq, n_heads, d_model) in fp32 (~22 GiB)."""
    with torch.no_grad():
        _, cache = model.run_with_cache(
            toks.unsqueeze(0),
            names_filter=lambda nm: (nm.endswith("hook_z") or nm.endswith("hook_mlp_out")
                                     or nm == "ln_final.hook_scale"))
    nL, nH = model.cfg.n_layers, model.cfg.n_heads
    scale = cache["ln_final.hook_scale"][0, pos].float()          # (1,)
    head = np.zeros((nL, nH), dtype=np.float32)
    for L in range(nL):
        z = cache["z", L][0, pos].float()                         # (n_heads, d_head)
        contrib = torch.einsum("hd,hdm->hm", z, model.W_O[L].float())   # (n_heads, d_model)
        head[L] = ((contrib / scale) @ u).detach().cpu().numpy()
    mlp = np.array([float((cache["mlp_out", L][0, pos].float() / scale) @ u)
                    for L in range(nL)], dtype=np.float32)
    del cache
    torch.cuda.empty_cache()
    return head, mlp


def run_e2c(model, recs, args):
    tid, u = think_dir(model)
    hd_a = hd_c = ml_a = ml_c = None
    used = 0
    for i, rec in enumerate(recs):
        fa, _, ra, ca = build_full(model, rec["aligned"])
        if ca is None:
            continue
        ta = toks_of(model, fa)
        ka = c2t(model, fa, ca) - 1
        if not (0 < ka < len(ta)):
            continue
        fc, _, rc, _ = build_full(model, rec["catch"])
        tc = toks_of(model, fc)
        if ka >= len(tc):
            continue
        ha, ma = dla(model, ta[:ka + 1], ka, u)
        hc, mc = dla(model, tc[:ka + 1], ka, u)
        hd_a = ha if hd_a is None else hd_a + ha
        hd_c = hc if hd_c is None else hd_c + hc
        ml_a = ma if ml_a is None else ml_a + ma
        ml_c = mc if ml_c is None else ml_c + mc
        used += 1
        if (i + 1) % 10 == 0:
            print(f"  e2c {i + 1}/{len(recs)}  (usable {used})")

    hd_a, hd_c, ml_a, ml_c = (x / used for x in (hd_a, hd_c, ml_a, ml_c))
    dhead = hd_a - hd_c
    dmlp = ml_a - ml_c
    print(f"\nE2c  (n={used})  positive Delta = supports </think> in aligned but not catch\n")
    flat = sorted(((float(dhead[L, H]), L, H) for L in range(model.cfg.n_layers)
                   for H in range(model.cfg.n_heads)), key=lambda t: -t[0])
    print("top heads by Delta:")
    for v, L, H in flat[:15]:
        print(f"  L{L:2d}H{H:2d}  Delta={v:+.3f}   aligned={hd_a[L,H]:+.3f}  catch={hd_c[L,H]:+.3f}")
    print("\ntop MLPs by Delta:")
    for L in np.argsort(-dmlp)[:8]:
        print(f"  L{L:2d}     Delta={dmlp[L]:+.3f}   aligned={ml_a[L]:+.3f}  catch={ml_c[L]:+.3f}")
    print(f"\nsum head Delta {dhead.sum():+.2f} + sum mlp Delta {dmlp.sum():+.2f} "
          f"~= logit(</think>) gap aligned-catch")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="phase2/loop_pairs.jsonl")
    ap.add_argument("--experiment", default="all", choices=["e2a", "e2b", "e2c", "all"])
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n", type=int, default=0, help="limit pairs (0 = all)")
    ap.add_argument("--C", type=float, default=0.03, help="probe L2 (smaller = stronger)")
    ap.add_argument("--dir", default="phase2")
    ap.add_argument("--raw-think", action="store_true",
                    help="loop_pairs came from a run that kept literal <think> tags "
                         "(no --reasoning-parser); use the real </think> boundary")
    args = ap.parse_args()

    global RAW_THINK
    RAW_THINK = args.raw_think

    recs = [json.loads(l) for l in open(args.pairs)]
    if args.n:
        recs = recs[: args.n]
    print(f"{len(recs)} pairs; loading {args.model} on {args.device} ...")
    model = get_model(args.model, args.device, args.dtype)
    model.eval()
    model.requires_grad_(False)          # no autograd anywhere in this script

    todo = ["e2a", "e2b", "e2c"] if args.experiment == "all" else [args.experiment]
    for ex in todo:
        print("\n" + "=" * 74 + f"\n{ex.upper()}\n" + "=" * 74)
        {"e2a": run_e2a, "e2b": run_e2b, "e2c": run_e2c}[ex](model, recs, args)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
