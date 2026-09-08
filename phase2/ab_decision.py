"""
Step 1-2 of the mechanistic analysis: where and by what is the (A) vs (B)
answer-token decision written, on the "unfaithful selection" catch trials?

  logit lens  : logit(A) - logit(B) at the answer position, per layer
  DLA         : per-head and per-MLP contribution to the (A - B) unembed direction

Run on Qwen2.5-7B (fits a 24GB card in bf16) or -3B for fast iteration.

  pip install transformer_lens transformers
  python phase2/ab_decision.py --pairs phase2/pairs.jsonl --model Qwen/Qwen2.5-7B-Instruct \
      --category unfaithful --n 40
"""
import argparse
import json

import torch


def build_ctx(model, prompt_user, cot):
    """chat-templated user turn + assistant CoT prefix + 'Answer: ('."""
    msgs = [{"role": "user", "content": prompt_user}]
    head = model.tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True)
    return head + cot + "Answer: ("


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="phase2/pairs.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--category", default="unfaithful")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="phase2/ab_decision.json")
    args = ap.parse_args()

    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        args.model, dtype="bfloat16", device=args.device)
    model.eval()

    A = model.to_single_token("A")
    B = model.to_single_token("B")
    dir_AB = (model.W_U[:, A] - model.W_U[:, B]).float()  # [d_model]

    rows = [json.loads(l) for l in open(args.pairs)]
    rows = [r for r in rows if r["role"] == "corrupted" and r["category"] == args.category][: args.n]
    print(f"{len(rows)} '{args.category}' catch trials")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    lens = torch.zeros(n_layers + 1)
    head_dla = torch.zeros(n_layers, n_heads)
    mlp_dla = torch.zeros(n_layers)
    correct_A = 0

    for i, r in enumerate(rows):
        ctx = build_ctx(model, r["prompt_user"], r["cot"])
        toks = model.to_tokens(ctx)
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                toks, names_filter=lambda n: (
                    n.endswith("resid_post") or n.endswith("resid_pre")
                    or n.endswith("hook_z") or n.endswith("mlp_out")
                    or n == "ln_final.hook_scale"))
        correct_A += int((logits[0, -1, A] > logits[0, -1, B]).item())

        # logit lens: accumulated resid -> final LN -> (A - B)
        acc = cache.accumulated_resid(layer=-1, incl_mid=False, pos_slice=-1,
                                      apply_ln=True, return_labels=False)  # [layer+1, d_model]
        lens += (acc.float() @ dir_AB).cpu()

        # per-head DLA
        z = cache.stack_head_results(layer=-1, pos_slice=-1, apply_ln=True)  # [L*H, d_model]
        head_dla += (z.float() @ dir_AB).cpu().reshape(n_layers, n_heads)

        # per-MLP DLA
        for L in range(n_layers):
            mo = cache["mlp_out", L][0, -1].float()
            scale = cache["ln_final.hook_scale"][0, -1].float()
            mlp_dla[L] += (mo / scale) @ dir_AB.cpu() if mo.is_cpu else \
                ((mo / scale.to(mo.device)).cpu() @ dir_AB.cpu())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)}")

    lens /= len(rows); head_dla /= len(rows); mlp_dla /= len(rows)
    flat = [(float(head_dla[L, H]), L, H) for L in range(n_layers) for H in range(n_heads)]
    flat.sort(key=lambda t: -abs(t[0]))

    print(f"\nmodel says (A): {correct_A}/{len(rows)}")
    print("\nlogit-lens (A - B) by layer (accumulated, LN'd):")
    for L in range(n_layers + 1):
        bar = "#" * int(max(lens[L], 0) * 4)
        print(f"  L{L:2d} {lens[L]:+.3f} {bar}")
    print("\ntop-15 heads by |DLA| onto (A - B):")
    for v, L, H in flat[:15]:
        print(f"  L{L:2d}H{H:2d}  {v:+.3f}")
    print("\ntop MLPs by |DLA|:")
    for L in sorted(range(n_layers), key=lambda L: -abs(mlp_dla[L]))[:8]:
        print(f"  L{L:2d}  {mlp_dla[L]:+.3f}")

    json.dump({"n": len(rows), "says_A": correct_A,
               "logit_lens": lens.tolist(),
               "head_dla": head_dla.tolist(),
               "mlp_dla": mlp_dla.tolist()},
              open(args.out, "w"), indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
