"""
Reproduce the reasoning-state analysis of Yu et al. 2026, "Can We Break LLMs Out of
Self-Loops?" (arXiv 2607.18100, SOPHIA), Sec. 2 / 4.1 / App. B, on our traces.

Their pipeline, as described in the paper:
  1. segment each reasoning trace at sentence-initial discourse markers
  2. embed each step = mean of the LAST-layer hidden states of a base model
     (Qwen3-4B-Base) over the step's tokens, from ONE forward pass over
     question || T1 || ... || Tn
  3. z-score per dataset, K-means K=5, seed 42
  4. order clusters by mean step index -> phases P1..P5
  5. report: phase-transition matrix, dwell / forward / backward mass,
     P(P5|P5), correct vs incorrect token length, #clusters visited,
     longest mono-cluster run; a "self-loop" is z_i == z_{i-1}

Not specified in the paper (our choices, flagged in the output):
  * the exact discourse-marker list (MARKERS below)
  * the max sequence length for the embedding pass (--max-tokens)
  * which of their traces count as "incorrect" when truncated -- we report
    correct / wrong-answer / truncated (no answer) separately

We add one cross-tab they do not have: their self-loop rate vs OUR loop (the
trace hit the token cap without answering).

  python phase2/reproduce_sophia.py --raw results/raw_numpair_r1.jsonl \
      --raw results/raw_r1.jsonl --tag r1 --out phase2/raw/sophia_r1.json

GPU for the embedding pass (~8 GB for Qwen3-4B-Base in bf16); stop vLLM first.
Embeddings are cached to <out>.npz, so re-clustering needs no GPU.
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np

MARKERS = (r"Wait|But|So|Hmm+|Alternatively|Let me|Let's|Now|Okay|OK|Therefore|Thus|"
           r"Hence|Maybe|Perhaps|Actually|First|Then|Next|Hold on|Alright|However|"
           r"Also|Another|Similarly|Finally|In summary|To confirm|Double-check|Check")
SPLIT = re.compile(rf"(?:(?<=[.!?:])\s+|\n+)(?=(?:{MARKERS})\b)")


def segment(text):
    """Steps = spans starting at a sentence-initial discourse marker."""
    parts = [p for p in SPLIT.split(text) if p and p.strip()]
    return parts


def load(paths, variants):
    rows = []
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            if r.get("mode", "cot") != "cot" or (r.get("system_name") or "none") != "none":
                continue
            v = "open" if r["kind"] == "stage1_open" else r.get("variant")
            if variants and v not in variants:
                continue
            if not (r.get("reasoning") or "").strip():
                continue
            r["_v"] = v
            r["_outcome"] = ("truncated" if r.get("truncated")
                             else "correct" if r.get("correct") else "wrong")
            rows.append(r)
    return rows


def embed(rows, model_name, max_tokens, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True).to(device).eval()
    feats, owner, stepidx = [], [], []
    for k, r in enumerate(rows):
        steps = segment(r["reasoning"])
        q_ids = tok(r["question"] + "\n\n", add_special_tokens=False).input_ids
        ids, spans = list(q_ids), []
        for s in steps:
            t = tok(s, add_special_tokens=False).input_ids
            if len(ids) + len(t) > max_tokens:
                break
            spans.append((len(ids), len(ids) + len(t)))
            ids += t
        if not spans:
            continue
        with torch.inference_mode():
            out = m.model(torch.tensor([ids], device=device))   # last hidden state
            h = out.last_hidden_state[0].float()
        for i, (a, b) in enumerate(spans):
            if b > a:
                feats.append(h[a:b].mean(0).cpu().numpy())
                owner.append(k)
                stepidx.append(i)
        del out, h
        torch.cuda.empty_cache()
        if (k + 1) % 20 == 0:
            print(f"  embedded {k + 1}/{len(rows)} traces, {len(feats)} steps", flush=True)
    return np.stack(feats), np.array(owner), np.array(stepidx)


def analyse(rows, X, owner, stepidx, K=5, seed=42):
    from sklearn.cluster import KMeans
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    lab = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(Z)
    mean_pos = {c: stepidx[lab == c].mean() for c in range(K)}
    order = sorted(range(K), key=lambda c: mean_pos[c])
    phase = {c: order.index(c) for c in range(K)}            # 0..K-1 = P1..PK
    seqs = collections.defaultdict(list)
    for o, s, c in sorted(zip(owner, stepidx, lab)):
        seqs[o].append(phase[c])

    T = np.zeros((K, K))
    for s in seqs.values():
        for a, b in zip(s, s[1:]):
            T[a, b] += 1
    n = T.sum()
    res = dict(
        n_traces=len(seqs), n_steps=int(len(lab)),
        phases=[dict(phase=f"P{i + 1}", mean_step=round(float(mean_pos[order[i]]), 1),
                     n=int((lab == order[i]).sum())) for i in range(K)],
        dwell=float(np.trace(T) / n), forward=float(np.triu(T, 1).sum() / n),
        backward=float(np.tril(T, -1).sum() / n),
        P_last_last=float(T[K - 1, K - 1] / max(T[K - 1].sum(), 1)),
        transition_counts=T.astype(int).tolist(),
    )

    def runs(s):
        best = cur = 1
        for a, b in zip(s, s[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        return best

    by = collections.defaultdict(list)
    for o, s in seqs.items():
        r = rows[o]
        by[(r["_v"], r["_outcome"])].append(dict(
            steps=len(s), visited=len(set(s)), longest_run=runs(s),
            selfloop=float(np.mean([a == b for a, b in zip(s, s[1:])])) if len(s) > 1 else 0.0,
            chars=len(r["reasoning"]), reached_last=int(K - 1 in s)))
    res["by_variant_outcome"] = {
        f"{v}/{oc}": dict(n=len(L), **{k: round(float(np.mean([d[k] for d in L])), 3)
                                      for k in L[0]})
        for (v, oc), L in sorted(by.items())}
    return res


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="append", required=True, help="results jsonl (repeatable)")
    ap.add_argument("--variants", default="open,aligned,swap,catch,numpair",
                    help="comma list; 'open' = stage1 open-ended")
    ap.add_argument("--embed-model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load(args.raw, set(args.variants.split(",")))
    print(f"{len(rows)} traces: "
          f"{dict(collections.Counter((r['_v'], r['_outcome']) for r in rows))}")
    cache = os.path.splitext(args.out)[0] + ".npz"
    if os.path.exists(cache):
        d = np.load(cache)
        X, owner, stepidx = d["X"], d["owner"], d["stepidx"]
        print(f"  loaded cached embeddings {cache}: {X.shape}")
    else:
        X, owner, stepidx = embed(rows, args.embed_model, args.max_tokens, args.device)
        np.savez(cache, X=X, owner=owner, stepidx=stepidx)
    res = analyse(rows, X, owner, stepidx)
    res.update(tag=args.tag, raw=args.raw, embed_model=args.embed_model,
               markers=MARKERS, max_tokens=args.max_tokens)
    json.dump(res, open(args.out, "w"), indent=1)

    print(f"\nphases: " + "  ".join(f"{p['phase']}@{p['mean_step']}(n={p['n']})" for p in res["phases"]))
    print(f"dwell {res['dwell']:.3f}  forward {res['forward']:.3f}  backward {res['backward']:.3f}"
          f"  P(P5|P5) {res['P_last_last']:.3f}   (paper: dwell 0.94-0.96, backward 0, P55 1.0)")
    print(f"\n{'variant/outcome':24s} {'n':>4s} {'steps':>6s} {'chars':>7s} {'visited':>7s}"
          f" {'selfloop':>8s} {'longRun':>7s} {'reachP5':>7s}")
    for k, d in res["by_variant_outcome"].items():
        print(f"{k:24s} {d['n']:4d} {d['steps']:6.1f} {d['chars']:7.0f} {d['visited']:7.2f}"
              f" {d['selfloop']:8.3f} {d['longest_run']:7.1f} {d['reached_last']:7.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
