"""
"Ask the model itself": where in a looping run did the model come closest to
stopping?  No text heuristic, no second (finished) run of the same question.

  --stage scan   (TransformerLens) one chunked forward pass over every looping run;
                 at every line break (a token containing "\\n": where </think> can
                 naturally follow) record log P(</think>). The model's stopping point
                 = the line break with the highest log P(</think>). The text-detected
                 commit point (build_pairs.commit_point) is recorded too, for
                 comparison.
  --stage stop   (vLLM) at each point (model-chosen; text commit if any), force
                 </think> and let the model write its answer (T=0.6, --seeds
                 seeds, up to --post-cap tokens); score it.
  --stage summary

Looping runs: every sampled natural loop (AQuA, LogiQA, MATH-500; 4 seeds) and the
loops in the controlled valid-answer conditions (open / aligned / swap, boxed prompt).

    python looping_mechanism/model_stop_points.py --stage scan
    python looping_mechanism/model_stop_points.py --stage stop      # vLLM serving
    python looping_mechanism/model_stop_points.py --stage summary
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "r1-distill-qwen-7b", "model_stop_points")
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
sys.path.insert(0, HERE)
from generate import DATA, extract, is_correct  # noqa: E402

NAT = os.path.join(ROOT, "natural_looping", "results", "r1-distill-qwen-7b", "sampled_t0.6_max12k")
CTRL = os.path.join(ROOT, "controlled_looping", "results", "r1-distill-qwen-7b",
                    "sampled_t0.6_max12k", "prompt_boxed")


def load_loops():
    """every looping run, with the data row (item) needed to score an answer"""
    items = {}
    for f in DATA.values():
        for r in map(json.loads, open(os.path.join(ROOT, "natural_looping", "data", f))):
            items[("nat", r["id"])] = r
    for c in ("open", "aligned", "swap"):
        for r in map(json.loads, open(os.path.join(ROOT, "controlled_looping", "data", "boxed",
                                                   f"{c}.jsonl"))):
            items[("ctrl", r["id"])] = r
    loops = []
    for f in sorted(glob.glob(os.path.join(NAT, "seed*", "*.jsonl"))):
        for r in map(json.loads, open(f)):
            if r["truncated"]:
                loops.append(dict(uid=f"{r['id']}|s{r['seed']}", group=r["dataset"],
                                  item=items[("nat", r["id"])], prompt=r["prompt"],
                                  reasoning=r["reasoning"]))
    for c in ("open", "aligned", "swap"):
        for f in sorted(glob.glob(os.path.join(CTRL, "seed*", f"{c}.jsonl"))):
            for r in map(json.loads, open(f)):
                if r["truncated"]:
                    loops.append(dict(uid=f"{r['id']}|s{r['seed']}", group=f"own150_{c}",
                                      item=items[("ctrl", r["id"])], prompt=r["prompt"],
                                      reasoning=r["reasoning"]))
    return loops


def chat_prefix(tok, prompt):
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                add_generation_prompt=True)
    return (s if s.endswith("\n") else s + "\n") if s.rstrip().endswith("<think>") else s + "<think>\n"


# --------------------------------------------------------------------- scan
def stage_scan(args):
    import torch
    from loop_probe import get_model, think_dir
    from transformer_lens import TransformerLensKeyValueCache
    from build_pairs import commit_point
    loops = load_loops()
    model = get_model(MODEL, "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tok = model.tokenizer
    tid, u = think_dir(model)
    u = u.float()
    comp = [int(x) for x in args.components.split(",")]
    buf = {}

    def grab(act, hook):
        buf[hook.name] = act[0].float()
        return act
    hooks = [(f"blocks.{L}.hook_mlp_out", grab) for L in comp] + [("ln_final.hook_scale", grab)]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "scan.jsonl")
    done = {json.loads(l)["uid"] for l in open(path)} if os.path.exists(path) else set()
    print(f"scan: {len(loops)} looping runs, {len(done)} done")
    with open(path, "a") as f:
        for i, lp in enumerate(loops, 1):
            if lp["uid"] in done:
                continue
            pre = chat_prefix(tok, lp["prompt"])
            enc = tok(pre + lp["reasoning"], add_special_tokens=False, return_offsets_mapping=True)
            ids = torch.tensor(enc["input_ids"], device="cuda")
            start = next(j for j, (a, b) in enumerate(enc["offset_mapping"]) if a >= len(pre))
            lp_think = torch.full((len(ids),), float("nan"))
            mlp_dla = torch.full((len(ids),), float("nan"))     # stop MLPs' direct push on </think>
            kv = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, 1)
            with torch.inference_mode(), model.hooks(fwd_hooks=hooks):
                for s in range(0, len(ids), args.chunk):
                    lg = model(ids[s:s + args.chunk].unsqueeze(0), past_kv_cache=kv)[0].float()
                    lp_think[s:s + lg.shape[0]] = (lg[:, tid] - torch.logsumexp(lg, -1)).cpu()
                    tot = sum(buf[f"blocks.{L}.hook_mlp_out"] for L in comp)
                    mlp_dla[s:s + lg.shape[0]] = ((tot / buf["ln_final.hook_scale"]) @ u).cpu()
                    del lg, tot
            del kv
            torch.cuda.empty_cache()
            toks = [tok.decode([t]) for t in enc["input_ids"]]
            cand = [j for j in range(start, len(ids) - 1) if "\n" in toks[j]]
            if not cand:
                continue
            best = max(cand, key=lambda j: float(lp_think[j]))
            it = lp["item"]
            gold = it["gold"] if it["format"] == "open" else None
            cp = commit_point(lp["reasoning"], gold, it["format"] == "mcq")
            text_pos = None
            if cp:
                c = len(pre) + cp[0]
                text_pos = next(j for j, (a, b) in enumerate(enc["offset_mapping"]) if a >= c or b > c) - 1
            top = sorted(cand, key=lambda j: -float(lp_think[j]))[:5]
            mlp_best = max(cand, key=lambda j: float(mlp_dla[j]))
            f.write(json.dumps(dict(
                uid=lp["uid"], group=lp["group"], n_tokens=len(ids), think_start=start,
                model_pos=best, model_logp=float(lp_think[best]),
                model_char=enc["offset_mapping"][best][1] - len(pre),
                model_frac=(best - start) / max(len(ids) - start, 1),
                model_mlp=float(mlp_dla[best]),
                mlp_pos=mlp_best, mlp_max=float(mlp_dla[mlp_best]),
                mlp_median_linebreak=float(np.median([float(mlp_dla[j]) for j in cand])),
                text_pos=text_pos, text_logp=None if text_pos is None else float(lp_think[text_pos]),
                text_mlp=None if text_pos is None else float(mlp_dla[text_pos]),
                text_frac=None if text_pos is None else (text_pos - start) / max(len(ids) - start, 1),
                top5=[(j, float(lp_think[j])) for j in top],
                context=(lp["reasoning"][max(0, enc["offset_mapping"][best][1] - len(pre) - 200):
                                         enc["offset_mapping"][best][1] - len(pre)]))) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(loops)}")


# --------------------------------------------------------------------- stop
def stage_stop(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tid = tok.convert_tokens_to_ids("</think>")
    loops = {lp["uid"]: lp for lp in load_loops()}
    scan = [json.loads(l) for l in open(os.path.join(OUT, "scan.jsonl"))]
    path = os.path.join(OUT, "stop.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["uid"], r["point"], r["seed"]) for r in map(json.loads, open(path))}
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    jobs = []
    for s in scan:
        for point in ("model", "text"):
            pos = s["model_pos"] if point == "model" else s["text_pos"]
            if pos is None:
                continue
            for seed in range(args.seeds):
                if (s["uid"], point, seed) not in done:
                    jobs.append((s, point, pos, seed))
    print(f"stop: {len(jobs)} generations")

    def work(job):
        s, point, pos, seed = job
        lp = loops[s["uid"]]
        ids = tok(chat_prefix(tok, lp["prompt"]) + lp["reasoning"],
                  add_special_tokens=False)["input_ids"][:pos + 1] + [tid]
        r = client.completions.create(model=MODEL, prompt=ids, max_tokens=args.post_cap,
                                      temperature=0.6, top_p=0.95, seed=seed)
        return dict(uid=s["uid"], group=s["group"], point=point, seed=seed, pos=pos,
                    frac=s["model_frac"] if point == "model" else s["text_frac"],
                    answer=r.choices[0].text, answer_tokens=r.usage.completion_tokens)

    with open(path, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"  error: {type(e).__name__}: {e}")
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}")


# --------------------------------------------------------------------- mlp
def stage_mlp(args):
    """At the model-chosen point of --n-mlp random loops: force </think>, paste the
    catch-learned stop-MLP output (mean over the controlled catch-vs-aligned finished
    runs, MLPs --components, once at that token), or the same edit on MLP 15; then
    generate the answer (T=0.6) and score it."""
    import random
    import torch
    import loop_probe
    import rq3_resolve
    from loop_probe import get_model, think_dir
    from rq3_resolve import Cond, generate_batch
    from stop_signal import measure, positions
    loop_probe.RAW_THINK = False
    rq3_resolve.prefill.__defaults__ = (256,)
    comp = [int(x) for x in args.components.split(",")]
    model = get_model(MODEL, "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tid, u = think_dir(model)
    uhat = (u / u.norm()).float()
    dpath = os.path.join(OUT, "catch_donor.pt")
    if os.path.exists(dpath):
        donor = torch.load(dpath)
    else:                                           # mean over the G2 finished runs
        vecs = []
        for p in map(json.loads, open(os.path.join(HERE, "data", "pairs.jsonl"))):
            if p["group"] == "ctrl_cross":
                pos = positions(model, p, tid)
                if pos:
                    vecs.append(measure(model, pos[0], pos[1], u.float(), tid)[1])
        donor = {L: torch.stack([v[L] for v in vecs]).mean(0) for L in comp + [15]}
        torch.save(donor, dpath)
    donor = {L: v.to("cuda") for L, v in donor.items()}
    loops = {lp["uid"]: lp for lp in load_loops()}
    scan = [json.loads(l) for l in open(os.path.join(OUT, "scan.jsonl"))]
    random.Random(0).shuffle(scan)
    scan = scan[: args.n_mlp]
    conds = [Cond("x_force", "x", "force"),
             Cond("x_mlp@1", "x", "interp", 1.0, comp, once=True),
             Cond("x_ctrl@1", "x", "interp", 1.0, [15], once=True)]
    path = os.path.join(OUT, "mlp.jsonl")
    done = {json.loads(l)["uid"] for l in open(path)} if os.path.exists(path) else set()
    rng = torch.Generator(device="cuda").manual_seed(0)
    with open(path, "a") as f:
        for i, s in enumerate(scan, 1):
            if s["uid"] in done:
                continue
            lp = loops[s["uid"]]
            ids = model.tokenizer(chat_prefix(model.tokenizer, lp["prompt"]) + lp["reasoning"],
                                  add_special_tokens=False)["input_ids"][:s["model_pos"] + 1]
            prefix = torch.tensor(ids, device="cuda")
            try:
                outs = generate_batch(model, prefix, conds, tid, 256, args.post_cap, donor, uhat,
                                      {}, temperature=0.6, rng=rng)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {s['uid']}: out of GPU memory, skipped")
                continue
            for c, o in zip(conds, outs):
                post = model.tokenizer.decode(o["post_ids"]) if o["stop"] is not None else None
                f.write(json.dumps(dict(uid=s["uid"], group=s["group"], condition=c.name[2:],
                                        stop=o["stop"], post=post,
                                        post_tokens=len(o["post_ids"])), ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(scan)}] {s['uid']}  stop@: "
                  + " ".join(f"{c.name[2:]}={o['stop']}" for c, o in zip(conds, outs)))


# --------------------------------------------------------------------- summary
def stage_summary(args):
    loops = {lp["uid"]: lp for lp in load_loops()}
    scan = {json.loads(l)["uid"]: json.loads(l) for l in open(os.path.join(OUT, "scan.jsonl"))}
    S = [json.loads(l) for l in open(os.path.join(OUT, "stop.jsonl"))] if os.path.exists(
        os.path.join(OUT, "stop.jsonl")) else []
    for r in S:                               # score in the main thread (math-verify)
        it = loops[r["uid"]]["item"]
        pred = extract(r["answer"], it)
        r["correct"] = is_correct(pred, it)
        r["capped"] = r["answer_tokens"] >= args.post_cap
    groups = sorted({s["group"] for s in scan.values()})
    L = ["# Stopping points chosen by the model itself", "",
         "For every looping run: the line break with the highest P(</think>) (model point) and",
         "the text-detected commit point (text point). At each, </think> is forced and the",
         f"answer generated (T=0.6, {args.seeds} seeds, up to {args.post_cap} tokens).", "",
         "## Where the points are", "",
         "| group | loops | has text point | model point at (median % of loop) | max P(</think>) "
         "(median) | model point within 50 tokens of text point | stop-MLP push: model point / "
         "typical line break / real stop (catch pairs) |", "|---|---|---|---|---|---|---|"]
    for g in groups:
        R = [s for s in scan.values() if s["group"] == g]
        tp = [s for s in R if s["text_pos"] is not None]
        near = [abs(s["model_pos"] - s["text_pos"]) <= 50 for s in tp]
        L.append(f"| {g} | {len(R)} | {100 * len(tp) / len(R):.0f}% | "
                 f"{100 * np.median([s['model_frac'] for s in R]):.0f}% | "
                 f"{np.median([np.exp(s['model_logp']) for s in R]):.3f} | "
                 f"{100 * np.mean(near) if near else float('nan'):.0f}% | "
                 f"{np.median([s['model_mlp'] for s in R]):+.1f} / "
                 f"{np.median([s['mlp_median_linebreak'] for s in R]):+.1f} / ~+24 |")
    if S:
        L += ["", "## Forcing </think> at each point", "",
              "| group | point | runs | answer correct | answer hits the cap (keeps reasoning) |",
              "|---|---|---|---|---|"]
        for g in groups:
            for point in ("model", "text"):
                R = [r for r in S if r["group"] == g and r["point"] == point]
                if not R:
                    continue
                L.append(f"| {g} | {point} | {len(R)} | "
                         f"{100 * np.mean([bool(r['correct']) for r in R]):.0f}% | "
                         f"{100 * np.mean([r['capped'] for r in R]):.0f}% |")
    mpath = os.path.join(OUT, "mlp.jsonl")
    if os.path.exists(mpath):
        M = [json.loads(l) for l in open(mpath)]
        L += ["", "## The catch-learned MLP switch at the model point", "",
              "| condition | runs | stops | answer correct | answer hits the cap |", "|---|---|---|---|---|"]
        for c in ("force", "mlp@1", "ctrl@1"):
            R = [r for r in M if r["condition"] == c]
            if not R:
                continue
            ok = [bool(is_correct(extract(r["post"], loops[r["uid"]]["item"]), loops[r["uid"]]["item"]))
                  if r["post"] is not None else False for r in R]
            L.append(f"| {c} | {len(R)} | {100 * np.mean([r['stop'] is not None for r in R]):.0f}% | "
                     f"{100 * np.mean(ok):.0f}% | "
                     f"{100 * np.mean([r['post_tokens'] >= args.post_cap for r in R if r['post']]) if any(r['post'] for r in R) else 0:.0f}% |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["scan", "stop", "mlp", "summary"])
    ap.add_argument("--components", default="27,26,25,24,22,20")
    ap.add_argument("--n-mlp", type=int, default=60)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--post-cap", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()
    {"scan": stage_scan, "stop": stage_stop, "mlp": stage_mlp,
     "summary": stage_summary}[args.stage](args)


if __name__ == "__main__":
    main()
