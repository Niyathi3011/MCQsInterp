"""
Why does the stop switch (6 late MLPs, set once to the catch-learned values) stop R1's
loops but never Qwen3's?  Where is Qwen3's stop signal written, and what would a patch
need to include to make the model actually stop?

  1. locate  On the model's MATH catch-vs-aligned pairs (data/pairs_<tag>.jsonl,
             ctrl_cross): direct contribution of EVERY attention layer and MLP to the
             logit(</think>) gap between the finished run's stop and the loop's commit.
  2. patch   At the loop's commit token, replace a set of components with (a) this pair's
             own finished-run values (upper bound) and (b) the mean over pairs (the
             switch). Sets: the six stop MLPs; the top-12 MLPs; every late MLP; the
             top-6 attention layers; MLPs + attention; the six MLPs pushed harder
             (alpha = 2, 3); the whole residual stream at one layer (localisation).
             Measured: % of the gap recovered, and whether </think> becomes the
             model's top choice (what generation needs to stop).
  3. generate  On held-out natural loops (bench_switch test loops, AQuA + LogiQA): do the
             promising MLP sets actually make the model write </think>?

    LOOP_MODEL=Qwen/Qwen3-4B-Thinking-2507 python looping_mechanism/switch_layers.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
sys.path.insert(0, HERE)
from model_cfg import CFG, MODEL, TAG  # noqa: E402
import bench_switch  # noqa: E402

OUT = os.path.join(HERE, "results", TAG, "switch_layers")


def capture(model, toks, pos, names):
    """logits row and {hook name: activation} at position pos (last token run on a
    frozen KV cache, which is returned for further patched runs)"""
    from rq3_resolve import prefill
    kv = prefill(model, toks[:pos + 1])
    kv.freeze()
    last = toks[pos:pos + 1].unsqueeze(0)
    logits, cache = model.run_with_cache(last, past_kv_cache=kv, names_filter=lambda n: n in names)
    acts = {n: cache[n][0, -1].float().clone() for n in names}
    row = logits[0, -1].float()
    del cache, logits
    return row, acts, kv, last


def patched(model, kv, last, repl, tid):
    """repl: {hook name: (target vector, alpha)} -> (logit, prob, top1) of </think>"""
    def hook(act, hook):
        tgt, alpha = repl[hook.name]
        v = act[0, -1].float()
        act[0, -1] = (v + alpha * (tgt.to(v.device) - v)).to(act.dtype)
        return act
    with torch.inference_mode():
        lg = model.run_with_hooks(last, past_kv_cache=kv, fwd_hooks=[(n, hook) for n in repl])
    row = lg[0, -1].float()
    return float(row[tid]), float(torch.softmax(row, -1)[tid]), int(row.argmax()) == tid


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=46)
    ap.add_argument("--gen-loops", type=int, default=20)
    ap.add_argument("--post-cap", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    bench_switch.MODEL = MODEL
    from stop_signal import positions
    model, tid, u = bench_switch.load_model()
    nL = model.cfg.n_layers
    M = lambda L: f"blocks.{L}.hook_mlp_out"      # noqa: E731
    A = lambda L: f"blocks.{L}.hook_attn_out"     # noqa: E731
    R = lambda L: f"blocks.{L}.hook_resid_post"   # noqa: E731
    names = {f(L) for L in range(nL) for f in (M, A, R)} | {"ln_final.hook_scale"}
    pairs = [p for p in map(json.loads, open(os.path.join(HERE, "data", CFG["ref_pairs"])))
             if p["group"] == "ctrl_cross"][:a.max_pairs]

    # ---- 1. locate: per-component contribution to the gap
    fin, rows, t0 = [], [], time.time()
    for i, p in enumerate(pairs, 1):
        pos = positions(model, p, tid)
        if pos is None:
            continue
        rf, af, kv, _ = capture(model, pos[0], pos[1], names)
        del kv
        rl, al, kv, _ = capture(model, pos[2], pos[3], names)
        del kv
        torch.cuda.empty_cache()
        sf, sl = af["ln_final.hook_scale"], al["ln_final.hook_scale"]
        dla = lambda acts, s, n: float((acts[n] / s) @ u)  # noqa: E731
        rows.append(dict(id=p["id"], finish=float(rf[tid]), loop=float(rl[tid]),
                         mlp=[dla(af, sf, M(L)) - dla(al, sl, M(L)) for L in range(nL)],
                         attn=[dla(af, sf, A(L)) - dla(al, sl, A(L)) for L in range(nL)]))
        fin.append({n: v.cpu() for n, v in af.items() if n != "ln_final.hook_scale"})
        if i % 10 == 0:
            print(f"  locate {i}/{len(pairs)}  {time.time() - t0:4.0f}s")
    gap = np.mean([r["finish"] - r["loop"] for r in rows])
    mg = np.array([r["mlp"] for r in rows]).mean(0)
    ag = np.array([r["attn"] for r in rows]).mean(0)
    top_m = [int(L) for L in np.argsort(-mg)]
    top_a = [int(L) for L in np.argsort(-ag)]
    L = [f"# Where is the stop signal written, and what makes {TAG} stop?", "",
         f"{len(rows)} MATH catch-vs-aligned pairs. logit(</think>): finished stop "
         f"{np.mean([r['finish'] for r in rows]):+.1f}, loop commit {np.mean([r['loop'] for r in rows]):+.1f}, "
         f"gap {gap:+.1f}.", "",
         "## 1. Direct contribution to the gap (share of the gap)", "",
         f"- all MLPs {100 * mg.sum() / gap:.0f}%, all attention {100 * ag.sum() / gap:.0f}% "
         "(the rest: embeddings / indirect)",
         f"- top MLPs: " + ", ".join(f"{L_} ({100 * mg[L_] / gap:.0f}%)" for L_ in top_m[:12]),
         f"- top attention: " + ", ".join(f"{L_} ({100 * ag[L_] / gap:.0f}%)" for L_ in top_a[:8]),
         f"- the six stop MLPs {CFG['stop']}: {100 * mg[CFG['stop']].sum() / gap:.0f}%", ""]
    print("\n".join(L))

    # ---- 2. patch sets at the loop's commit token
    late = list(range(nL // 2, nL))
    sets = {"6 stop MLPs": ([M(x) for x in CFG["stop"]], 1),
            "6 stop MLPs, alpha 2": ([M(x) for x in CFG["stop"]], 2),
            "6 stop MLPs, alpha 3": ([M(x) for x in CFG["stop"]], 3),
            "top-12 MLPs": ([M(x) for x in top_m[:12]], 1),
            f"all late MLPs ({late[0]}-{nL - 1})": ([M(x) for x in late], 1),
            "top-6 attention": ([A(x) for x in top_a[:6]], 1),
            "6 MLPs + top-6 attention": ([M(x) for x in CFG["stop"]] + [A(x) for x in top_a[:6]], 1),
            "all late MLPs + attention": ([M(x) for x in late] + [A(x) for x in late], 1)}
    for x in range(nL // 2, nL, 3):
        sets[f"residual stream at layer {x}"] = ([R(x)], 1)
    mean = {n: torch.stack([f[n] for f in fin]).mean(0) for n in fin[0]}
    res = {k: {"own": [], "mean": []} for k in sets}
    t0 = time.time()
    for i, (p, r, f) in enumerate(zip([p for p in pairs if positions(model, p, tid)], rows, fin), 1):
        pos = positions(model, p, tid)
        from rq3_resolve import prefill
        kv = prefill(model, pos[2][:pos[3] + 1])
        kv.freeze()
        last = pos[2][pos[3]:pos[3] + 1].unsqueeze(0)
        g = r["finish"] - r["loop"]
        for k, (hn, alpha) in sets.items():
            for src, vals in (("own", f), ("mean", mean)):
                lg, pr, t1 = patched(model, kv, last, {n: (vals[n].cuda(), alpha) for n in hn}, tid)
                res[k][src].append(dict(rec=(lg - r["loop"]) / g, prob=pr, top1=t1))
        del kv
        torch.cuda.empty_cache()
        if i % 10 == 0:
            print(f"  patch {i}/{len(rows)}  {time.time() - t0:4.0f}s")
    L += ["## 2. Patching at the loop's commit token", "",
          "own = this pair's finished-run values (upper bound); mean = the average over pairs "
          "(what a switch can use). top-1 = </think> becomes the most likely token (needed to stop).", "",
          "| patched | gap recovered: own / mean | </think> top-1: own / mean | P(</think>): own / mean |",
          "|---|---|---|---|"]
    for k in sets:
        o, m_ = res[k]["own"], res[k]["mean"]
        f_ = lambda xs, key: np.mean([x[key] for x in xs])  # noqa: E731
        L.append(f"| {k} | {100 * f_(o, 'rec'):.0f}% / {100 * f_(m_, 'rec'):.0f}% | "
                 f"{100 * f_(o, 'top1'):.0f}% / {100 * f_(m_, 'top1'):.0f}% | "
                 f"{f_(o, 'prob'):.2f} / {f_(m_, 'prob'):.2f} |")
    print("\n".join(L[-(len(sets) + 5):]))
    json.dump(dict(rows=rows, patch=res, top_mlp=top_m, top_attn=top_a),
              open(os.path.join(OUT, "locate_patch.json"), "w"))
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")

    # ---- 3. generation on held-out natural loops: MLP sets that might make it stop
    from rq3_resolve import Cond, generate_batch
    from generate import extract, is_correct
    donor = {x: mean[M(x)].cuda() for x in range(nL)}
    gsets = [("force", Cond("x_force", "x", "force")),
             ("6 stop MLPs", Cond("x_a", "x", "interp", 1.0, CFG["stop"], once=True)),
             ("6 stop MLPs, alpha 3", Cond("x_b", "x", "interp", 3.0, CFG["stop"], once=True)),
             ("top-12 MLPs", Cond("x_c", "x", "interp", 1.0, top_m[:12], once=True)),
             ("all late MLPs", Cond("x_d", "x", "interp", 1.0, late, once=True)),
             ("all late MLPs, alpha 2", Cond("x_e", "x", "interp", 2.0, late, once=True)),
             ("6 stop MLPs, every step", Cond("x_f", "x", "interp", 1.0, CFG["stop"], once=False))]
    loops = []
    for b in ("aqua", "logiqa"):
        d = os.path.join(HERE, "results", TAG, "bench_switch", f"{b}_t0.6")
        items = bench_switch.items_of(b)
        for t in map(json.loads, open(os.path.join(d, "test_loops.jsonl"))):
            if t["commit_off"] is not None:
                loops.append((t, items[t["question"]]))
    loops = loops[:a.gen_loops]
    rng = torch.Generator(device="cuda").manual_seed(0)
    gen = {k: [] for k, _ in gsets}
    uhat = torch.zeros(model.cfg.d_model, device="cuda")
    t0 = time.time()
    with open(os.path.join(OUT, "generate.jsonl"), "w") as fo:
        for i, (t, it) in enumerate(loops, 1):
            prefix = bench_switch.prefix_ids(model, t["prompt"], t["reasoning"], t["commit_off"])
            outs = generate_batch(model, prefix, [c for _, c in gsets], tid, 256, a.post_cap, donor, uhat, {},
                                  temperature=0.6, rng=rng)
            for (k, _), o in zip(gsets, outs):
                post = model.tokenizer.decode(o["post_ids"]) if o["stop"] is not None else None
                ok = bool(is_correct(extract(post, it), it)) if post is not None else False
                gen[k].append(dict(stop=o["stop"] is not None, correct=ok))
                fo.write(json.dumps(dict(id=t["id"], cond=k, stop=o["stop"], correct=ok,
                                         think=model.tokenizer.decode(o["think_ids"]), post=post)) + "\n")
            print(f"  generate {i}/{len(loops)} {time.time() - t0:4.0f}s  " +
                  " ".join(f"[{k}]={'stop' if g[-1]['stop'] else '-'}" for k, g in gen.items()))
    L += ["", f"## 3. Generation on {len(loops)} held-out natural loops (AQuA + LogiQA), from the commit point",
          "", "| intervention | stops (writes </think> within 256 tokens) | correct |", "|---|---|---|"]
    for k, g in gen.items():
        L.append(f"| {k} | {100 * np.mean([x['stop'] for x in g]):.0f}% | {100 * np.mean([x['correct'] for x in g]):.0f}% |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-(len(gsets) + 4):]))


if __name__ == "__main__":
    main()
