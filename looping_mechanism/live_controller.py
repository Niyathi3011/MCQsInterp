"""
A stop controller for looping runs: SENSOR = the answer-correctness probe (layer-24
residual stream, answer_probe.py), SWITCH = the catch-learned stop MLPs (or, for
comparison, simply forcing </think>).

Each looping run is replayed; at every "states an answer, then doubts it" point (in
order, using only the text up to that point) the probe gives P(stated answer is
correct). The controller stops at the FIRST point with P > --threshold; the answer
is then generated for real and scored. The probe for a run's question is always one
trained WITHOUT that question (the fold it was held out of).

  --stage fit     (CPU) refit the 5 by-question folds of the probe; save them
  --stage scan    (TL)  probe probabilities at every claim point of every loop
  --stage force   (vLLM) force </think> at the controller's stopping point, 2 seeds
  --stage switch  (TL)  paste the catch-learned stop-MLP output there instead
  --stage summary

    python looping_mechanism/live_controller.py --stage fit
    python looping_mechanism/live_controller.py --stage scan
    python looping_mechanism/live_controller.py --stage force      # vLLM serving
    python looping_mechanism/live_controller.py --stage switch
    python looping_mechanism/live_controller.py --stage summary
"""
import argparse
import collections
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, "results", "r1-distill-qwen-7b")
PROBE = os.path.join(RES, "answer_probe")
MSP = os.path.join(RES, "model_stop_points")
OUT = os.path.join(RES, "live_controller")
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
sys.path.insert(0, HERE)
from generate import extract, is_correct  # noqa: E402
from model_stop_points import chat_prefix, load_loops  # noqa: E402

LAYER = 24


# --------------------------------------------------------------------- fit
def stage_fit(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    meta = [json.loads(l) for l in open(os.path.join(PROBE, "points.jsonl"))]
    X = np.load(os.path.join(PROBE, "features.npz"))[f"L{LAYER}"].astype(np.float32)
    y = np.array([m["label"] for m in meta], dtype=int)
    g = np.array([m["qid"] for m in meta])
    folds, qfold = [], {}
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        folds.append(make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
                     .fit(X[tr], y[tr]))
        for q in set(g[te]):
            qfold[q] = k
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(dict(folds=folds, qfold=qfold, layer=LAYER), open(os.path.join(OUT, "probe_folds.pkl"), "wb"))
    print(f"saved 5 probe folds (layer {LAYER}); {len(qfold)} questions assigned to held-out folds")


# --------------------------------------------------------------------- scan
def stage_scan(args):
    import torch
    from build_pairs import CUE, STRONG, claims
    from loop_probe import get_model
    from transformer_lens import TransformerLensKeyValueCache
    P = pickle.load(open(os.path.join(OUT, "probe_folds.pkl"), "rb"))
    model = get_model(MODEL, "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tok = model.tokenizer
    buf = {}

    def grab(act, hook):
        buf["h"] = act[0]
        return act
    path = os.path.join(OUT, "scan.jsonl")
    done = {json.loads(l)["uid"] for l in open(path)} if os.path.exists(path) else set()
    loops = load_loops()
    print(f"scan: {len(loops)} loops, {len(done)} done")
    with open(path, "a") as f:
        for i, lp in enumerate(loops, 1):
            if lp["uid"] in done:
                continue
            it = lp["item"]
            qid = lp["uid"].split("|")[0]
            mcq = it["format"] == "mcq"
            cl = claims(lp["reasoning"], mcq, STRONG)
            if not mcq:
                cl += claims(lp["reasoning"], False, CUE, value_at_end=True)
            cl = sorted(set(cl))
            pre = chat_prefix(tok, lp["prompt"])
            enc = tok(pre + lp["reasoning"], add_special_tokens=False, return_offsets_mapping=True)
            om = enc["offset_mapping"]
            pts = []
            for off, val in cl:
                c = len(pre) + off
                k = next((j for j, (a, b) in enumerate(om) if a >= c or b > c), None)
                if k is not None and k > 1:
                    pts.append(dict(pos=k - 1, char=off, value=val))
            rec = dict(uid=lp["uid"], group=lp["group"], qid=qid, n_tokens=len(om), points=[])
            if pts and qid in P["qfold"]:
                ids = torch.tensor(enc["input_ids"][: max(p["pos"] for p in pts) + 1], device="cuda")
                want = {p["pos"] for p in pts}
                got = {}
                kv = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, 1)
                with torch.inference_mode(), model.hooks(
                        fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", grab)]):
                    for s in range(0, len(ids), args.chunk):
                        model(ids[s:s + args.chunk].unsqueeze(0), past_kv_cache=kv)
                        for p in want:
                            if s <= p < s + args.chunk:
                                got[p] = buf["h"][p - s].float().cpu().numpy()
                del kv
                torch.cuda.empty_cache()
                clf = P["folds"][P["qfold"][qid]]
                Xp = np.stack([got[p["pos"]] for p in pts])
                probs = clf.predict_proba(Xp)[:, 1]
                for p, pr in zip(pts, probs):
                    rec["points"].append(dict(p, prob=float(pr)))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(loops)}")


def decision(rec, t):
    """index of the controller's stopping point (first claim with P > t), or None"""
    return next((j for j, p in enumerate(rec["points"]) if p["prob"] > t), None)


# --------------------------------------------------------------------- force (vLLM)
def stage_force(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tid = tok.convert_tokens_to_ids("</think>")
    loops = {lp["uid"]: lp for lp in load_loops()}
    scan = [json.loads(l) for l in open(os.path.join(OUT, "scan.jsonl"))]
    path = os.path.join(OUT, "force.jsonl")
    done = {(r["uid"], r["seed"]) for r in map(json.loads, open(path))} if os.path.exists(path) else set()
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    jobs = []
    for rec in scan:
        j = decision(rec, args.threshold)
        if j is None:
            continue
        for seed in range(args.seeds):
            if (rec["uid"], seed) not in done:
                jobs.append((rec, rec["points"][j]["pos"], seed))
    print(f"force: {len(jobs)} generations")

    def work(job):
        rec, pos, seed = job
        lp = loops[rec["uid"]]
        ids = tok(chat_prefix(tok, lp["prompt"]) + lp["reasoning"],
                  add_special_tokens=False)["input_ids"][:pos + 1] + [tid]
        r = client.completions.create(model=MODEL, prompt=ids, max_tokens=args.post_cap,
                                      temperature=0.6, top_p=0.95, seed=seed)
        return dict(uid=rec["uid"], group=rec["group"], seed=seed, pos=pos,
                    answer=r.choices[0].text, answer_tokens=r.usage.completion_tokens)

    with open(path, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        for i, fu in enumerate(as_completed([ex.submit(work, j) for j in jobs]), 1):
            try:
                f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"  error: {type(e).__name__}: {e}")
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}")


# --------------------------------------------------------------------- switch (TL)
def stage_switch(args):
    import torch
    import rq3_resolve
    from loop_probe import get_model, think_dir
    from rq3_resolve import Cond, generate_batch
    rq3_resolve.prefill.__defaults__ = (256,)
    comp = [27, 26, 25, 24, 22, 20]
    model = get_model(MODEL, "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tid, u = think_dir(model)
    uhat = (u / u.norm()).float()
    donor = {L: v.to("cuda") for L, v in torch.load(os.path.join(MSP, "catch_donor.pt")).items()}
    loops = {lp["uid"]: lp for lp in load_loops()}
    scan = [json.loads(l) for l in open(os.path.join(OUT, "scan.jsonl"))]
    path = os.path.join(OUT, "switch.jsonl")
    done = {json.loads(l)["uid"] for l in open(path)} if os.path.exists(path) else set()
    conds = [Cond("x_mlp@1", "x", "interp", 1.0, comp, once=True)]
    rng = torch.Generator(device="cuda").manual_seed(0)
    todo = [r for r in scan if decision(r, args.threshold) is not None and r["uid"] not in done]
    print(f"switch: {len(todo)} loops")
    with open(path, "a") as f:
        for i, rec in enumerate(todo, 1):
            pos = rec["points"][decision(rec, args.threshold)]["pos"]
            lp = loops[rec["uid"]]
            ids = model.tokenizer(chat_prefix(model.tokenizer, lp["prompt"]) + lp["reasoning"],
                                  add_special_tokens=False)["input_ids"][:pos + 1]
            try:
                o = generate_batch(model, torch.tensor(ids, device="cuda"), conds, tid, 256,
                                   args.post_cap, donor, uhat, {}, temperature=0.6, rng=rng)[0]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {rec['uid']}: out of GPU memory, skipped")
                continue
            f.write(json.dumps(dict(uid=rec["uid"], group=rec["group"], pos=pos, stop=o["stop"],
                                    answer=model.tokenizer.decode(o["post_ids"]) if o["stop"] is not None else None,
                                    answer_tokens=len(o["post_ids"])), ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")


# --------------------------------------------------------------------- summary
def stage_summary(args):
    loops = {lp["uid"]: lp for lp in load_loops()}
    scan = {json.loads(l)["uid"]: json.loads(l) for l in open(os.path.join(OUT, "scan.jsonl"))}

    def ok(uid, ans):
        it = loops[uid]["item"]
        return bool(is_correct(extract(ans, it), it)) if ans is not None else False

    def stated_ok(uid, val):
        from answer_probe import value_correct
        return value_correct(val, loops[uid]["item"])

    groups = ["aqua", "logiqa", "math500", "own150_open", "own150_aligned", "own150_swap"]
    t = args.threshold
    F = [json.loads(l) for l in open(os.path.join(OUT, "force.jsonl"))] if os.path.exists(
        os.path.join(OUT, "force.jsonl")) else []
    W = [json.loads(l) for l in open(os.path.join(OUT, "switch.jsonl"))] if os.path.exists(
        os.path.join(OUT, "switch.jsonl")) else []
    B = [json.loads(l) for l in open(os.path.join(MSP, "stop.jsonl"))]       # baselines
    L = ["# Live stop controller: probe as sensor, stop-MLPs as switch", "",
         f"Each looping run is replayed; the controller stops at the first 'states an answer, then",
         f"doubts it' point where the held-out probe gives P(correct) > {t}. Answers are generated",
         "for real after the stop (T=0.6) and scored. Loops that never trigger keep looping (no",
         "answer). Baselines: stop at the first commit point / at the model's closest-to-stop",
         "point (force </think>, same settings).", "",
         "| group | loops | controller stops | correct (of stopped): switch / force | correct over ALL "
         "loops: controller (switch) / first commit / model point / never stop |",
         "|---|---|---|---|---|"]
    tot = collections.Counter()
    for g in groups:
        U = [u for u, r in scan.items() if r["group"] == g]
        if not U:
            continue
        fired = [u for u in U if decision(scan[u], t) is not None]
        sw = [ok(r["uid"], r["answer"]) for r in W if r["group"] == g]
        fo = [ok(r["uid"], r["answer"]) for r in F if r["group"] == g]
        bt = [ok(r["uid"], r["answer"]) for r in B if r["group"] == g and r["point"] == "text"]
        bm = [ok(r["uid"], r["answer"]) for r in B if r["group"] == g and r["point"] == "model"]
        n_text = len({r["uid"] for r in B if r["group"] == g and r["point"] == "text"})
        all_sw = sum(sw) / len(U) if sw else float("nan")
        all_bt = (np.mean(bt) * n_text / len(U)) if bt else float("nan")
        all_bm = np.mean(bm) if bm else float("nan")
        L.append(f"| {g} | {len(U)} | {100 * len(fired) / len(U):.0f}% | "
                 f"{100 * np.mean(sw) if sw else float('nan'):.0f}% / {100 * np.mean(fo) if fo else float('nan'):.0f}% | "
                 f"{100 * all_sw:.0f}% / {100 * all_bt:.0f}% / {100 * all_bm:.0f}% / 0% |")
    L += ["", "## Threshold sweep (stated answer at the stopping point; no generation)", "",
          "| threshold | loops stopped | stated answer correct (of stopped) |", "|---|---|---|"]
    for tt in (0.3, 0.5, 0.7, 0.8, 0.9):
        st = [stated_ok(u, r["points"][decision(r, tt)]["value"]) for u, r in scan.items()
              if decision(r, tt) is not None]
        L.append(f"| {tt} | {100 * len(st) / len(scan):.0f}% | {100 * np.mean(st) if st else float('nan'):.0f}% |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "scan", "force", "switch", "summary"])
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--post-cap", type=int, default=1200)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()
    {"fit": stage_fit, "scan": stage_scan, "force": stage_force, "switch": stage_switch,
     "summary": stage_summary}[args.stage](args)


if __name__ == "__main__":
    main()
