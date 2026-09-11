"""
Print the E2a read-out positions (opt_end, mid, concl, pre_end) for every
problem x variant in a loop_pairs(.jsonl) file, decoded as a short context
window around each token -- so you can see exactly what's being probed,
across the whole study set (not just individual traces pasted by hand).

    python phase2/show_positions.py --pairs phase2/loop_pairs.jsonl \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B [--raw-think] [--n 10]

Needs the GPU (loads the model to tokenize/chat-template exactly as
loop_probe.py does); stop any vLLM server first.
"""
import argparse
import json

import loop_probe as lp


def window(model, toks, pos, before=8, after=4):
    lo, hi = max(0, pos - before), min(len(toks), pos + after + 1)
    pre = model.tokenizer.decode(toks[lo:pos])
    tok = model.tokenizer.decode(toks[pos:pos + 1])
    post = model.tokenizer.decode(toks[pos + 1:hi])
    return f"{pre!r} [{tok!r}] {post!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="phase2/loop_pairs.jsonl")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--raw-think", action="store_true",
                    help="pairs came from a raw <think>-tag run (no --reasoning-parser)")
    ap.add_argument("--n", type=int, default=0, help="limit problems (0 = all)")
    args = ap.parse_args()
    lp.RAW_THINK = args.raw_think

    recs = [json.loads(l) for l in open(args.pairs)]
    if args.n:
        recs = recs[:args.n]
    print(f"{len(recs)} problems; loading {args.model} ...")
    model = lp.get_model(args.model, args.device, args.dtype)
    model.eval()
    model.requires_grad_(False)

    for rec in recs:
        print(f"\n{'=' * 78}\nsource_idx {rec['source_idx']}")
        for var in ("aligned", "swap", "catch"):
            p = rec[var]
            full, ucl, rcs, close = lp.build_full(model, p)
            toks = lp.toks_of(model, full)
            if p["truncated"]:
                cap = lp.catch_cap(model, full, rcs, p.get("loop_onset_off"), rec["catch_window"])
                if cap:
                    toks = toks[:cap]
            pos = lp.positions_e2a(model, full, ucl, rcs, p, close, len(toks))
            print(f"  [{var:7s}] gold={p['gold_value']!r}  truncated={p['truncated']}")
            for name in ("opt_end", "mid", "concl", "pre_end"):
                if name in pos:
                    print(f"    {name:8s} tok#{pos[name]:5d}  {window(model, toks, pos[name])}")
                else:
                    print(f"    {name:8s} -- (not set)")


if __name__ == "__main__":
    main()
