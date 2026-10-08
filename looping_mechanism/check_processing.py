"""
Check that phase2/loop_probe.py's in-place weight processing (RMSNorm folding,
unembedding centering, value-bias folding) preserves a model's computation:
compare the residual stream and the centered logits before vs after processing on a
test prompt. Exit code 1 if an early-layer residual differs by more than --tol
(bf16 rounding grows with depth, so early layers are the clean test).

    python looping_mechanism/check_processing.py --model Qwen/Qwen3-4B-Thinking-2507
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase2"))
from loop_probe import _process_rms_inplace, get_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tol", type=float, default=0.03)
    args = ap.parse_args()
    m = get_model(args.model, "cuda", "bfloat16", process=False)
    nL = m.cfg.n_layers
    early, late = max(1, nL // 4), nL - 2
    names = {f"blocks.{early}.hook_resid_post", f"blocks.{late}.hook_resid_post"}
    t = m.to_tokens("Natalia sold clips to 48 of her friends in April, and then half as many in "
                    "May. How many clips did she sell altogether?", prepend_bos=False)
    rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
    with torch.no_grad():
        l0, c0 = m.run_with_cache(t, names_filter=lambda n: n in names)
        l0 = l0.float() - l0.float().mean(-1, keepdim=True)
        c0 = {k: v.float().clone() for k, v in c0.items()}
        _process_rms_inplace(m)
        l1, c1 = m.run_with_cache(t, names_filter=lambda n: n in names)
        l1 = l1.float()
    re = rel(c1[f"blocks.{early}.hook_resid_post"].float(), c0[f"blocks.{early}.hook_resid_post"])
    rl = rel(c1[f"blocks.{late}.hook_resid_post"].float(), c0[f"blocks.{late}.hook_resid_post"])
    top1 = float((l1.argmax(-1) == l0.argmax(-1)).float().mean())
    print(f"{args.model}: {nL} layers | residual rel. diff layer {early}: {re:.4f}, layer {late}: "
          f"{rl:.4f} | centered logits rel. diff {rel(l1, l0):.4f} | top-1 agreement {top1:.2f}")
    if re > args.tol:
        print(f"FAIL: early-layer difference {re:.4f} > {args.tol}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
