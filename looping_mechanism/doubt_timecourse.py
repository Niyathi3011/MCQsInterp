"""
Steps 2 and 3: is the withheld stop signal specific to LOOPS?

For every matched pair, one forward pass over each run reads the </think> stop
signal at many points:
  loop run     every commit point: the model re-states its committed answer and
               then doubts it ("... = 57, but 57 isn't an option. | Wait, ...").
               Step 3: does the signal ever rise over the repeated commits?
  finish run   every ORDINARY doubt: it states an answer, doubts it, but later
               does stop.  Step 2: are loop doubts lower than these healthy ones?
               ...and the final stop (token right before </think>).

At each point: logit and probability of </think>, whether it is top-1, and the
direct contribution of the stop-signal MLPs (--components) to the </think> logit.

    python looping_mechanism/doubt_timecourse.py     # -> results/r1-distill-qwen-7b/doubt_timecourse/
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
sys.path.insert(0, HERE)
import loop_probe  # noqa: E402
from build_pairs import CUE, STRONG, claims, norm  # noqa: E402
from loop_probe import get_model, think_dir  # noqa: E402
from rq3_patch import parse_layers  # noqa: E402
from stop_signal import GROUPS, LABEL, chat_prefix, encode, token_at  # noqa: E402

OUT = os.path.join(HERE, "results", "r1-distill-qwen-7b", "doubt_timecourse")


def doubt_points(text, commit_value, gold, mcq, max_points):
    """char offsets of the doubt phrases after claims.  With commit_value: only
    re-claims of that value (the loop's repeated commits); else any claim."""
    if gold is not None:
        cl = claims(text, mcq, CUE, value_at_end=True)
    else:
        cl = claims(text, mcq, STRONG)
    if commit_value is not None:
        cl = [(o, v) for o, v in cl if v == norm(str(commit_value))]
    return [o for o, _ in cl][:max_points]


@torch.inference_mode()
def read_points(model, toks, points, u, tid, comp, chunk=512):
    """one chunked pass; returns {pos: dict(logit, prob, top1, comp_dla)}"""
    from transformer_lens import TransformerLensKeyValueCache
    want = {p for p in points if 0 <= p < len(toks)}
    if not want:
        return {}
    end = max(want) + 1
    kv = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, 1)
    store = collections.defaultdict(dict)
    state = {"start": 0}

    def grab(act, hook):
        s = state["start"]
        for p in want:
            if s <= p < s + act.shape[1]:
                store[p][hook.name] = act[0, p - s].float().clone()
        return act

    names = [f"blocks.{L}.hook_mlp_out" for L in comp] + ["ln_final.hook_scale"]
    out = {}
    with model.hooks(fwd_hooks=[(n, grab) for n in names]):
        for s in range(0, end, chunk):
            state["start"] = s
            logits = model(toks[s:min(s + chunk, end)].unsqueeze(0), past_kv_cache=kv)
            for p in want:
                if s <= p < s + logits.shape[1]:
                    row = logits[0, p - s].float()
                    out[p] = dict(logit=float(row[tid]), prob=float(torch.softmax(row, -1)[tid]),
                                  top1=int(row.argmax()) == tid)
            del logits
    for p in out:
        scale = store[p]["ln_final.hook_scale"]
        out[p]["comp_dla"] = float(sum(store[p][f"blocks.{L}.hook_mlp_out"] / scale for L in comp) @ u)
    del kv
    torch.cuda.empty_cache()
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", default="27,26,25,24,22,20")
    ap.add_argument("--max-points", type=int, default=10)
    ap.add_argument("--groups", default="greedy_ref,ctrl_cross,ctrl_same,nat_aqua,nat_logiqa,nat_math500")
    args = ap.parse_args()
    comp = parse_layers(args.components)
    groups = args.groups.split(",")
    os.makedirs(OUT, exist_ok=True)
    loop_probe.RAW_THINK = False
    model = get_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "cuda", "bfloat16")
    model.eval()
    model.requires_grad_(False)
    tid, u = think_dir(model)
    u = u.float()

    pairs = [p for p in map(json.loads, open(os.path.join(HERE, "data", "pairs.jsonl")))
             if p["group"] in groups]
    path = os.path.join(OUT, "points.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if os.path.exists(path) else set()
    with open(path, "a") as f:
        for i, p in enumerate(pairs, 1):
            if p["id"] in done:
                continue
            mcq = "A)" in p["loop"]["prompt"] or "A. " in p["loop"]["prompt"]
            mcq = mcq and p["group"].startswith("nat_")                   # natural MCQ: letters
            gold = p["gold"] if p["group"] in ("ctrl_cross", "ctrl_same", "greedy_ref") else None
            rec = dict(id=p["id"], group=p["group"])
            # loop run: repeated commits of the committed value
            lp = p["loop"]
            pre = chat_prefix(model, lp["prompt"])
            tl, ol = encode(model, pre + lp["reasoning"])
            offs = doubt_points(lp["reasoning"], lp["commit_value"], gold, mcq, args.max_points)
            lpos = [(token_at(ol, len(pre) + o) or 0) - 1 for o in offs]
            rl = read_points(model, tl, lpos, u, tid, comp)
            rec["loop"] = [dict(char=o, **rl[q]) for o, q in zip(offs, lpos) if q in rl]
            # finish run: ordinary doubts (any claim) + the final stop
            fp = p["finish"]
            pre = chat_prefix(model, fp["prompt"])
            tf, of = encode(model, pre + fp["reasoning"] + "</think>")
            offs = doubt_points(fp["reasoning"], None, gold, mcq, args.max_points)
            fpos = [(token_at(of, len(pre) + o) or 0) - 1 for o in offs]
            stop = token_at(of, len(pre) + len(fp["reasoning"])) - 1
            rf = read_points(model, tf, fpos + [stop], u, tid, comp)
            rec["finish_doubts"] = [dict(char=o, **rf[q]) for o, q in zip(offs, fpos) if q in rf]
            rec["finish_stop"] = rf.get(stop)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(pairs)}")
    summarise(path, groups)


def summarise(path, groups):
    R = [json.loads(l) for l in open(path)]
    L = ["# Steps 2-3: is the withheld stop signal specific to loops?", "",
         "logit(</think>) at: the finished run's stop; ORDINARY doubts in the finished run",
         "(states an answer, doubts it, later stops); the looping run's repeated commits",
         "(re-states its answer, doubts it, never stops). comp = direct contribution of the",
         "stop-signal MLPs to the </think> logit.", "",
         "## Step 2: loop doubts vs ordinary doubts", "",
         "| group | pairs | finish stop | ordinary doubt (finish run) | loop commit #1 | "
         "loop - ordinary (paired) | comp: stop / ordinary / loop | </think> top-1 at ordinary doubts |",
         "|---|---|---|---|---|---|---|---|"]
    stats = {}
    for g in groups:
        G = [r for r in R if r["group"] == g and r["loop"] and r["finish_stop"]]
        if not G:
            continue
        stop = [r["finish_stop"]["logit"] for r in G]
        H = [r for r in G if r["finish_doubts"]]
        ordn = [np.mean([d["logit"] for d in r["finish_doubts"]]) for r in H]
        l1 = [r["loop"][0]["logit"] for r in G]
        diff = [r["loop"][0]["logit"] - np.mean([d["logit"] for d in r["finish_doubts"]]) for r in H]
        cs = np.mean([r["finish_stop"]["comp_dla"] for r in G])
        co = np.mean([np.mean([d["comp_dla"] for d in r["finish_doubts"]]) for r in H]) if H else np.nan
        cl = np.mean([r["loop"][0]["comp_dla"] for r in G])
        t1 = np.mean([d["top1"] for r in H for d in r["finish_doubts"]]) if H else np.nan
        from scipy.stats import wilcoxon
        try:
            p = wilcoxon(diff).pvalue if len(diff) > 5 else float("nan")
        except ValueError:
            p = float("nan")
        stats[g] = dict(n=len(G), n_ord=len(H))
        L.append(f"| {LABEL[g]} | {len(G)} ({len(H)} with ordinary doubts) | {np.mean(stop):+.1f} | "
                 f"{np.mean(ordn) if ordn else float('nan'):+.1f} | {np.mean(l1):+.1f} | "
                 f"{np.mean(diff) if diff else float('nan'):+.2f} (p={p:.1e}) | "
                 f"{cs:+.1f} / {co:+.1f} / {cl:+.1f} | {100 * t1:.0f}% |")
    L += ["", "## Step 3: the stop signal over the loop's repeated commits", "",
          "Mean logit(</think>) at the k-th time the looping run re-states its answer and doubts it.",
          "", "| group | " + " | ".join(f"#{k}" for k in range(1, 11)) + " | pairs reaching #5 |",
          "|---|" + "---|" * 11]
    for g in groups:
        G = [r for r in R if r["group"] == g and r["loop"]]
        if not G:
            continue
        cells = []
        for k in range(10):
            v = [r["loop"][k]["logit"] for r in G if len(r["loop"]) > k]
            cells.append(f"{np.mean(v):+.1f} (n={len(v)})" if len(v) >= 5 else "-")
        L.append(f"| {LABEL[g]} | " + " | ".join(cells) + f" | {sum(len(r['loop']) >= 5 for r in G)} |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
