"""
Step 1: does restoring the stop signal END a loop, and is the answer then right?

Starting point for every pair: the looping run's COMMIT point (the model has just
stated its answer; its next token was the doubt "Wait").  Two stages:

  --stage continue   (vLLM, fast) no intervention: resample the looping run from
                     its commit point, --seeds times, up to --max-new tokens. Does
                     the model stop by itself, and is it right when it does?
  --stage intervene  (TransformerLens) from the same point, sampled (T=0.6):
        base         no edit (thinking capped at --think-cap; only for reference)
        force        </think> appended immediately (budget forcing)
        bias         + b on logit(</think>) while thinking; b = the start-token logit
                     increase under mlp@1 (same push, on the logit only)
        mlp@1        restore MLPs --components at the commit token: replace their
                     output with the controlled donor (mean mlp_out right before
                     </think> in the sampled catch-vs-aligned finished runs)
        proj@1       same, but only the </think>-unembedding direction of the edit
        random@1     same size as the mlp@1 edit, random fixed direction
        ctrl@1       mlp@1 on MLP --control-layers instead
     All edits apply once, at the commit token; after </think> the answer is
     generated unedited up to --post-cap tokens and scored.
  --stage summary    tables -> summary.md

Groups: the natural pairs (AQuA, LogiQA, MATH-500) and --n-ctrl controlled catch
pairs (no correct option: we record which option it picks).

    python looping_mechanism/stop_intervention.py --stage continue     # vLLM running
    python looping_mechanism/stop_intervention.py --stage intervene    # vLLM stopped
    python looping_mechanism/stop_intervention.py --stage summary
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "r1-distill-qwen-7b", "stop_intervention")
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
from generate import DATA, extract, is_correct  # noqa: E402

NAT = ("nat_aqua", "nat_logiqa", "nat_math500")


# --------------------------------------------------------------------- pairs & items
def load_pairs(n_ctrl, seed=0):
    P = [json.loads(l) for l in open(os.path.join(HERE, "data", "pairs.jsonl"))]
    nat = [p for p in P if p["group"] in NAT]
    ctrl = [p for p in P if p["group"] == "ctrl_cross"]
    random.Random(seed).shuffle(ctrl)
    return nat + ctrl[:n_ctrl]


def load_items():
    items = {}
    for d, f in DATA.items():
        for r in map(json.loads, open(os.path.join(ROOT, "natural_looping", "data", f))):
            items[r["id"]] = r
    for r in map(json.loads, open(os.path.join(ROOT, "controlled_looping", "data", "boxed",
                                               "catch.jsonl"))):
        items[f"catch_{r['idx']}"] = r
    return items


def item_of(pair, items):
    if pair["group"] == "ctrl_cross":
        return items[f"catch_{pair['id'].split('_')[1]}"]
    return items[pair["id"]]


def score(post, item):
    pred = extract(post, item)
    return pred, is_correct(pred, item)


def commit_prefix(tok, pair):
    """token ids of the looping run up to and including the commit token (the
    token right before the doubt phrase) -- same construction as stop_signal.py"""
    s = tok.apply_chat_template([{"role": "user", "content": pair["loop"]["prompt"]}],
                                tokenize=False, add_generation_prompt=True)
    s = (s if s.endswith("\n") else s + "\n") if s.rstrip().endswith("<think>") else s + "<think>\n"
    enc = tok(s + pair["loop"]["reasoning"], add_special_tokens=False, return_offsets_mapping=True)
    c = len(s) + pair["loop"]["commit_off"]
    k = next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a >= c or b > c)
    return enc["input_ids"][:k]


def split_answer(text):
    """(thinking part, answer part or None) of a continuation"""
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        return a, b
    return text, None


# --------------------------------------------------------------------- stage: continue
def stage_continue(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    items = load_items()
    pairs = load_pairs(args.n_ctrl)
    path = os.path.join(OUT, "continue_no_intervention.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["id"], r["seed"]) for r in map(json.loads, open(path))}
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    jobs = [(p, s) for p in pairs for s in range(args.seeds) if (p["id"], s) not in done]
    print(f"continue: {len(pairs)} pairs x {args.seeds} seeds, {len(jobs)} to run")

    def work(job):
        p, s = job
        ids = commit_prefix(tok, p)
        r = client.completions.create(model=MODEL, prompt=ids, max_tokens=args.max_new,
                                      temperature=0.6, top_p=0.95, seed=s)
        text = r.choices[0].text
        think, ans = split_answer(text)
        it = item_of(p, items)
        pred, ok = score(ans, it) if ans is not None else (None, None if it["gold"] is None else False)
        return dict(id=p["id"], group=p["group"], seed=s, prefix_tokens=len(ids),
                    new_tokens=r.usage.completion_tokens, stopped=ans is not None,
                    think_tokens_to_stop=len(tok(think, add_special_tokens=False).input_ids)
                    if ans is not None else None,
                    pred=pred, correct=ok, finish_run_correct=p["finish"].get("correct"),
                    text=text)

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with open(path, "a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:  # noqa: BLE001
                print(f"  error: {type(e).__name__}: {e}")
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  {time.time() - t0:5.0f}s")
    print(f"-> {path}")


# --------------------------------------------------------------------- stage: intervene
def stage_intervene(args):
    import torch
    import loop_probe
    from loop_probe import get_model, think_dir
    from rq3_patch import parse_layers
    import rq3_resolve
    from rq3_resolve import Cond, generate_batch
    rq3_resolve.prefill.__defaults__ = (256,)    # smaller prefill chunks: commit points
                                                 # can sit ~11k tokens into the trace
    sys.path.insert(0, HERE)
    from stop_signal import measure, positions
    loop_probe.RAW_THINK = False
    comp, ctrl = parse_layers(args.components), parse_layers(args.control_layers)
    model = get_model(MODEL, "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tid, u = think_dir(model)
    uhat = (u / u.norm()).float()
    items = load_items()

    # the controlled donor: mean mlp_out right before </think> over the sampled
    # catch-vs-aligned finished runs (as in stop_signal.py)
    allp = [json.loads(l) for l in open(os.path.join(HERE, "data", "pairs.jsonl"))]
    vecs = []
    for p in [q for q in allp if q["group"] == "ctrl_cross"]:
        pos = positions(model, p, tid)
        if pos:
            vecs.append(measure(model, pos[0], pos[1], u.float(), tid)[1])
    L_all = sorted(set(comp) | set(ctrl))
    donor = {L: torch.stack([v[L] for v in vecs]).mean(0) for L in L_all}
    print(f"controlled donor: mean over {len(vecs)} finished runs")
    g = torch.Generator().manual_seed(0)
    rdir = {L: (lambda r: (r / r.norm()).to(uhat.device))(torch.randn(model.cfg.d_model, generator=g))
            for L in L_all}
    rng = torch.Generator(device="cuda").manual_seed(args.seed)

    conds = [Cond("x_base", "x"), Cond("x_force", "x", "force"), Cond("x_bias", "x", "bias"),
             Cond("x_mlp@1", "x", "interp", 1.0, comp, once=True),
             Cond("x_proj@1", "x", "proj", 1.0, comp, once=True),
             Cond("x_random@1", "x", "random", 1.0, comp, once=True),
             Cond("x_ctrl@1", "x", "interp", 1.0, ctrl, once=True)]
    path = os.path.join(OUT, "interventions.jsonl" if args.seed == 0
                        else f"interventions_seed{args.seed}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {r["id"] for r in map(json.loads, open(path))}
    pairs = [p for p in load_pairs(0 if args.natural_only else args.n_ctrl) if p["id"] not in done]
    print(f"intervene: {len(pairs)} pairs to run; conditions: {[c.name[2:] for c in conds]}")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "a") as f:
        for i, p in enumerate(pairs, 1):
            pos = positions(model, p, tid)
            if pos is None:
                continue
            prefix = pos[2][:pos[3] + 1]
            t0 = time.time()
            try:
                outs = generate_batch(model, prefix, conds, tid, args.think_cap, args.post_cap,
                                      donor, uhat, rdir, temperature=0.6, rng=rng)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [{i}/{len(pairs)}] {p['id']}: out of GPU memory "
                      f"(prefix {len(prefix)} tokens), skipped")
                continue
            it = item_of(p, items)
            for c, o in zip(conds, outs):
                post = model.tokenizer.decode(o["post_ids"]) if o["stop"] is not None else None
                pred, ok = score(post, it) if post is not None else \
                    (None, None if it["gold"] is None else False)
                f.write(json.dumps(dict(
                    id=p["id"], group=p["group"], condition=c.name[2:], prefix_tokens=len(prefix),
                    stop=o["stop"], think_new=len(o["think_ids"]), post_tokens=len(o["post_ids"]),
                    first_logit=o["first_logit"], bias=o["bias"], pred=pred, correct=ok,
                    finish_run_correct=p["finish"].get("correct"),
                    think=model.tokenizer.decode(o["think_ids"]), post=post), ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(pairs)}] {p['id']:22s} {time.time() - t0:5.1f}s  stop@: "
                  + " ".join(f"{c.name[2:]}={o['stop']}" for c, o in zip(conds, outs)))


# --------------------------------------------------------------------- stage: summary
def boot_ci(rows, n_boot=2000, seed=0):
    """95% bootstrap CI of the correct-rate, resampling PAIRS (seeds averaged per pair)"""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["id"]].append(bool(r["correct"]))
    v = np.array([np.mean(x) for x in per.values()])
    rng = np.random.default_rng(seed)
    bs = [rng.choice(v, len(v)).mean() for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def stage_summary(args):
    L = ["# Step 1: restoring the stop signal at the looping run's commit point", "",
         "DeepSeek-R1-Distill-Qwen-7B, sampled T=0.6. Start = the looping run's commit point",
         "(it has just stated its answer; originally it continued with \"Wait\" and never",
         "stopped: 0% correct). Natural pairs have a correct answer; controlled catch pairs",
         "do not (we report which option is picked).", ""]
    items = load_items()
    cpath = os.path.join(OUT, "continue_no_intervention.jsonl")
    if os.path.exists(cpath):
        C = []
        for r in map(json.loads, open(cpath)):          # re-score from the stored text
            if r["stopped"]:
                it = item_of(dict(group=r["group"], id=r["id"]), items)
                r["pred"], r["correct"] = score(split_answer(r["text"])[1], it)
            C.append(r)
        L += ["## A. No intervention: resampled from the commit point", "",
              f"Up to {max(r['new_tokens'] for r in C)} new tokens, several seeds per pair.", "",
              "| group | runs | stops by itself | median tokens to stop | correct (all runs) | "
              "correct when it stops | finished-run ceiling |", "|---|---|---|---|---|---|---|"]
        for g in NAT + ("ctrl_cross",):
            R = [r for r in C if r["group"] == g]
            if not R:
                continue
            st = [r for r in R if r["stopped"]]
            ceil = np.mean([bool(r["finish_run_correct"]) for r in R]) if g != "ctrl_cross" else None
            L.append(f"| {g} | {len(R)} | {100 * len(st) / len(R):.0f}% | "
                     f"{int(np.median([r['think_tokens_to_stop'] for r in st])) if st else '-'} | "
                     + ("n/a | n/a" if g == "ctrl_cross" else
                        f"{100 * np.mean([bool(r['correct']) for r in R]):.0f}% | "
                        f"{100 * np.mean([bool(r['correct']) for r in st]):.0f}%" if st else "- | -")
                     + f" | {'n/a' if ceil is None else f'{100 * ceil:.0f}%'} |")
        L.append("")
    import glob
    ipaths = sorted(glob.glob(os.path.join(OUT, "interventions*.jsonl")))
    if ipaths:
        I = []
        for ip in ipaths:                                # every seed; re-scored from the text
            sd = 0 if ip.endswith("interventions.jsonl") else int(ip.split("seed")[-1].split(".")[0])
            for r in map(json.loads, open(ip)):
                if r["post"] is not None:
                    it = item_of(dict(group=r["group"], id=r["id"]), items)
                    r["pred"], r["correct"] = score(r["post"], it)
                r["seed"] = sd
                I.append(r)
        L.append(f"Intervention seeds pooled: {sorted({r['seed'] for r in I})}. "
                 "[95% CI] = bootstrap over pairs (each pair's mean over its seeds).")
        L.append("")
        conds = ["base", "force", "bias", "mlp@1", "proj@1", "random@1", "ctrl@1"]
        L += ["## B. Interventions at the commit point", "",
              "stop = emitted </think> within the thinking cap; tokens = new thinking tokens before "
              "</think>; correct = the answer after </think> (natural groups; no stop = wrong); "
              "catch picks = which option a stopped controlled run chose.", ""]
        for g in NAT + ("ctrl_cross",):
            R = [r for r in I if r["group"] == g]
            if not R:
                continue
            L += [f"### {g} ({len({r['id'] for r in R})} pairs)", "",
                  "| condition | stops | median new thinking tokens | answer hits the "
                  f"{args.post_cap}-token cap (keeps reasoning after </think>) | median total new tokens | " +
                  ("picks A / B / neither (of stopped) |" if g == "ctrl_cross" else
                   "correct (all) | correct (of stopped) |"),
                  "|---|---|---|---|---|---|" + ("" if g == "ctrl_cross" else "---|")]
            for c in conds:
                Rc = [r for r in R if r["condition"] == c]
                if not Rc:
                    continue
                st = [r for r in Rc if r["stop"] is not None]
                med = int(np.median([r["stop"] for r in st])) if st else "-"
                if g == "ctrl_cross":
                    k = collections.Counter(r["pred"] or "neither" for r in st)
                    tail = (" / ".join(f"{100 * k[x] / len(st):.0f}%" for x in ("A", "B", "neither"))
                            if st else "-") + " |"
                else:
                    lo, hi = boot_ci(Rc)
                    tail = (f"{100 * np.mean([bool(r['correct']) for r in Rc]):.0f}% "
                            f"[{100 * lo:.0f}-{100 * hi:.0f}] | "
                            + (f"{100 * np.mean([bool(r['correct']) for r in st]):.0f}%" if st else "-")
                            + " |")
                capped = (f"{100 * np.mean([r['post_tokens'] >= args.post_cap for r in st]):.0f}%"
                          if st else "-")
                tot = int(np.median([r["think_new"] + r["post_tokens"] for r in Rc]))
                L.append(f"| {c} | {100 * len(st) / len(Rc):.0f}% | {med} | {capped} | {tot} | {tail}")
            if g != "ctrl_cross":
                L.append(f"\nfinished-run ceiling (same questions): "
                         f"{100 * np.mean([bool(r['finish_run_correct']) for r in R if r['condition'] == 'base']):.0f}% correct")
            L.append("")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["continue", "intervene", "summary"])
    ap.add_argument("--n-ctrl", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--components", default="27,26,25,24,22,20")
    ap.add_argument("--control-layers", default="15")
    ap.add_argument("--think-cap", type=int, default=256)
    ap.add_argument("--post-cap", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0, help="intervene: sampling seed (0 = the main run)")
    ap.add_argument("--natural-only", action="store_true", help="intervene: natural pairs only")
    args = ap.parse_args()
    {"continue": stage_continue, "intervene": stage_intervene, "summary": stage_summary}[args.stage](args)


if __name__ == "__main__":
    main()
