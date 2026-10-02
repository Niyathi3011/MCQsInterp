"""
A detector for "if the model stops here, will its answer be right?"

Surface repetition does not predict it (selective_stopping.md); this tests whether
the model's hidden state does, as Zhang et al. (2025) found for intermediate answers.

  --stage extract (GPU)  points = every "states an answer, then doubts it" place
                 (build_pairs.claims) and the real stop of finished runs, in all
                 sampled runs with a correct answer (natural AQuA / LogiQA / MATH-500,
                 controlled open / aligned / swap; 4 seeds): every looping run +
                 --n-finished random finished runs. Label = is the stated answer
                 correct (answer key). Features = residual stream after --layers at
                 the token right before the doubt (or before </think>).
  --stage train  (CPU)   logistic regression per layer; 5-fold split BY QUESTION (no
                 question in both train and test); AUC vs simple baselines; and a
                 stopping rule on the looping runs: stop at the first point where the
                 probe says P(correct) > t, vs stopping at the first commit point.

    python looping_mechanism/answer_probe.py --stage extract
    python looping_mechanism/answer_probe.py --stage train
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "r1-distill-qwen-7b", "answer_probe")
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
sys.path.insert(0, HERE)
from generate import DATA, extract, is_correct, letter_of_stated_number  # noqa: E402

NAT = os.path.join(ROOT, "natural_looping", "results", "r1-distill-qwen-7b", "sampled_t0.6_max12k")
CTRL = os.path.join(ROOT, "controlled_looping", "results", "r1-distill-qwen-7b",
                    "sampled_t0.6_max12k", "prompt_boxed")


def load_runs():
    items = {}
    for f in DATA.values():
        for r in map(json.loads, open(os.path.join(ROOT, "natural_looping", "data", f))):
            items[r["id"]] = r
    for c in ("open", "aligned", "swap"):
        for r in map(json.loads, open(os.path.join(ROOT, "controlled_looping", "data", "boxed",
                                                   f"{c}.jsonl"))):
            items[r["id"]] = r
    runs = []
    for f in sorted(glob.glob(os.path.join(NAT, "seed*", "*.jsonl"))):
        if os.path.basename(f) == "summary.json":
            continue
        for r in map(json.loads, open(f)):
            runs.append(dict(r, group=r["dataset"], item=items[r["id"]]))
    for c in ("open", "aligned", "swap"):
        for f in sorted(glob.glob(os.path.join(CTRL, "seed*", f"{c}.jsonl"))):
            for r in map(json.loads, open(f)):
                runs.append(dict(r, group=f"own150_{c}", item=items[r["id"]]))
    return runs


def value_correct(value, item):
    """is a stated value (letter, number or LaTeX) the correct answer?"""
    if item["format"] == "mcq":
        if value in item["letters"]:
            return value == item["gold"]
        L = letter_of_stated_number(f"\\boxed{{{value}}}", item)
        return L == item["gold"] if L else False
    return bool(is_correct(value, item))


def chat_prefix(tok, prompt):
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                add_generation_prompt=True)
    return (s if s.endswith("\n") else s + "\n") if s.rstrip().endswith("<think>") else s + "<think>\n"


# --------------------------------------------------------------------- extract
def stage_extract(args):
    import torch
    from build_pairs import CUE, STRONG, claims
    from loop_probe import get_model
    from transformer_lens import TransformerLensKeyValueCache
    layers = [int(x) for x in args.layers.split(",")]
    runs = load_runs()
    loops = [r for r in runs if r["truncated"]]
    fins = [r for r in runs if not r["truncated"]]
    random.Random(0).shuffle(fins)
    runs = loops + fins[: args.n_finished]
    print(f"extract: {len(loops)} looping + {min(len(fins), args.n_finished)} finished runs")
    model = get_model(MODEL, "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tok = model.tokenizer
    os.makedirs(OUT, exist_ok=True)
    meta, feats = [], {L: [] for L in layers}
    buf = {}

    def grab(act, hook):
        buf[hook.name] = act[0]
        return act
    hooks = [(f"blocks.{L}.hook_resid_post", grab) for L in layers]
    for i, r in enumerate(runs, 1):
        it = r["item"]
        mcq = it["format"] == "mcq"
        text = r["reasoning"]
        cl = claims(text, mcq, STRONG)
        if not mcq:
            cl += claims(text, False, CUE, value_at_end=True)
        cl = sorted(set(cl))
        if len(cl) > args.max_points:                           # evenly spaced subset
            idx = np.linspace(0, len(cl) - 1, args.max_points).round().astype(int)
            cl = [cl[j] for j in sorted(set(idx))]
        pre = chat_prefix(tok, r["prompt"])
        full = pre + text + ("" if r["truncated"] else "</think>")
        enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
        om = enc["offset_mapping"]

        def tok_before(char):
            k = next((j for j, (a, b) in enumerate(om) if a >= char or b > char), None)
            return None if k is None else k - 1
        pts = []
        for off, val in cl:
            k = tok_before(len(pre) + off)
            if k is not None and k > 0:
                pts.append(dict(pos=k, kind="doubt", value=val,
                                label=value_correct(val, it), frac=off / max(len(text), 1)))
        if not r["truncated"]:
            k = tok_before(len(pre) + len(text))
            pts.append(dict(pos=k, kind="stop", value=r.get("pred"),
                            label=bool(r.get("correct")), frac=1.0))
        if not pts:
            continue
        ids = torch.tensor(enc["input_ids"][: max(p["pos"] for p in pts) + 1], device="cuda")
        want = {p["pos"] for p in pts}
        got = {}
        kv = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, 1)
        with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
            for s in range(0, len(ids), args.chunk):
                model(ids[s:s + args.chunk].unsqueeze(0), past_kv_cache=kv)
                for p in want:
                    if s <= p < s + args.chunk:
                        got[p] = {L: buf[f"blocks.{L}.hook_resid_post"][p - s].float().cpu().numpy()
                                  for L in layers}
        del kv
        torch.cuda.empty_cache()
        for n, p in enumerate(pts):
            if p["pos"] not in got:
                continue
            for L in layers:
                feats[L].append(got[p["pos"]][L].astype(np.float16))
            meta.append(dict(run=f"{r['id']}|s{r.get('seed')}", qid=r["id"], group=r["group"],
                             looping=r["truncated"], kind=p["kind"], k=n, value=p["value"],
                             label=p["label"], frac=p["frac"]))
        if i % 100 == 0:
            print(f"  {i}/{len(runs)} runs, {len(meta)} points")
    np.savez_compressed(os.path.join(OUT, "features.npz"),
                        **{f"L{L}": np.stack(v) for L, v in feats.items()})
    with open(os.path.join(OUT, "points.jsonl"), "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"saved {len(meta)} points x layers {layers} -> {OUT}")


# --------------------------------------------------------------------- train
def stage_train(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    meta = [json.loads(l) for l in open(os.path.join(OUT, "points.jsonl"))]
    F = np.load(os.path.join(OUT, "features.npz"))
    y = np.array([m["label"] for m in meta], dtype=int)
    groups = np.array([m["qid"] for m in meta])
    L = ["# Answer-correctness probe: can the hidden state tell whether stopping here is right?", "",
         f"{len(meta)} points ({int(y.sum())} correct) from {len({m['run'] for m in meta})} runs; "
         f"{sum(m['looping'] for m in meta)} points in looping runs. Split: 5 folds by question.", "",
         "| features | AUC (all points) | AUC (loop points only) |", "|---|---|---|"]
    gkf = GroupKFold(n_splits=5)
    loopmask = np.array([m["looping"] for m in meta])
    oof_best, best_auc = None, -1
    # baselines
    for name, x in (("position in trace", np.array([[m["frac"]] for m in meta])),
                    ("index of the claim in the run", np.array([[m["k"]] for m in meta]))):
        oof = np.zeros(len(y))
        for tr, te in gkf.split(x, y, groups):
            clf = LogisticRegression(max_iter=1000).fit(x[tr], y[tr])
            oof[te] = clf.predict_proba(x[te])[:, 1]
        L.append(f"| baseline: {name} | {roc_auc_score(y, oof):.3f} | "
                 f"{roc_auc_score(y[loopmask], oof[loopmask]):.3f} |")
    for key in sorted(F.files, key=lambda k: int(k[1:])):
        X = F[key].astype(np.float32)
        oof = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=2000))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        auc, aucl = roc_auc_score(y, oof), roc_auc_score(y[loopmask], oof[loopmask])
        L.append(f"| probe on layer {key[1:]} | {auc:.3f} | {aucl:.3f} |")
        if aucl > best_auc:
            best_auc, oof_best, best_key = aucl, oof, key
    # stopping rule on looping runs: first doubt point with P(correct) > t
    runs = collections.defaultdict(list)
    for m, p in zip(meta, oof_best):
        if m["looping"] and m["kind"] == "doubt":
            runs[m["run"]].append((m["k"], p, m["label"], m["group"]))
    L += ["", f"## Stopping rule on looping runs (probe on layer {best_key[1:]}, out-of-fold)", "",
          "Stop at the first commit point where P(correct) > t; if none, let it loop (no answer).",
          "Accuracy = the stated answer at the stopping point is correct (the forced-stop answer",
          "tracks it: ~55-70% correct when it is right, ~4-21% when wrong).", "",
          "| rule | loops stopped | correct, of stopped | correct, of all loops |", "|---|---|---|---|"]
    R = [sorted(v) for v in runs.values()]
    first = [v[0][2] for v in R]
    L.append(f"| stop at the first commit point | 100% | {100 * np.mean(first):.0f}% | "
             f"{100 * np.mean(first):.0f}% |")
    for t in (0.3, 0.5, 0.7, 0.8, 0.9):
        stops = [next((lab for _, p, lab, _ in v if p > t), None) for v in R]
        st = [s for s in stops if s is not None]
        L.append(f"| probe, t = {t} | {100 * len(st) / len(R):.0f}% | "
                 f"{100 * np.mean(st) if st else float('nan'):.0f}% | {100 * sum(st) / len(R):.0f}% |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["extract", "train"])
    ap.add_argument("--layers", default="14,20,24,27")
    ap.add_argument("--n-finished", type=int, default=1200)
    ap.add_argument("--max-points", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()
    {"extract": stage_extract, "train": stage_train}[args.stage](args)


if __name__ == "__main__":
    main()
