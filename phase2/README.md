# Phase 2 — mechanistic account of the unfaithful answer selection

## Phenomenon (from Phase 1)

On `catch` trials — (A) a wrong number, (B) an unrelated sentence, no correct
option — Qwen2.5-7B-Instruct:

- computes the correct answer in its CoT, then emits `(A)` — ~65% of trials it
  can actually solve (matched on GSM8K and MATH-500).
- ~28% of pick-A cases: the reasoning itself is bent toward option A's wrong
  number ("anchored").
- transcripts show explicit "the answer is not among the options ... Answer: (A)".

So at the answer position `logit(A) - logit(B) > 0` even though the content of
the reasoning does not support (A).

## Hypothesis: two pathways to the answer token

| pathway | what it computes | expected locus |
| --- | --- | --- |
| **format** | "(A) is digits, (B) is prose -> pick (A)" | attn from answer pos -> option tokens, keyed on token type; mid layers |
| **content** | carry CoT-derived value, compare to options, pick match | attn from answer pos -> CoT answer span; late layers; says "neither"/weak B on catch |

Claim: on catch the **format pathway dominates** the A-B logit and the **content
pathway is causally inert**.

## Steps

1. `prep.py` — reconstruct the exact answer-position context for each catch trial
   (chat template + CoT prefix + `Answer: (`), tagged `unfaithful` / `anchored` /
   `faithful_B`, plus a matched **clean** trial per problem (aligned/swap where
   the model picked the numeric option correctly) for patching.
2. `ab_decision.py` — **logit lens** on `A - B` at the answer position by layer,
   and **direct logit attribution** (per head, per MLP) onto the `A - B` unembed
   direction. Output: the layer where (A) overtakes (B); the components that write it.
3. **Activation patching** (`patch.py`, next): clean -> corrupted, `resid_pre`
   per (layer, pos), then per-head `z`, then option-token positions only.
   Map components whose patch flips the decision.
4. **Attention knockout** at the answer position: zero attention to (a) the
   (A)-number token, (b) the (B)-sentence span, (c) the CoT span where gold was
   derived. Predicted: (a) collapses the shortcut, (c) ~no effect.
5. **"Prefer-numeric-option" direction**: diff-of-means over resid, numeric
   option present vs not (control for slot). Ablate -> catch pick-A drops; add to
   `aligned` -> overrides a correct answer.
6. **Commitment point**: project resid onto the A-direction at every CoT token.
   Is (A) already dominant before the arithmetic completes? (post-hoc test)

## Setup on the pod

```bash
source /workspace/venv/bin/activate
pip install --no-cache-dir transformer_lens transformers
export HF_HOME=/workspace/hf_cache

python phase2/prep.py --raw results/raw.jsonl          --out phase2/pairs_gsm8k.jsonl
python phase2/prep.py --raw results/raw_math500.jsonl  --out phase2/pairs_math500.jsonl

python phase2/ab_decision.py --pairs phase2/pairs_gsm8k.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --category unfaithful --n 40
```

Qwen2.5-7B loads in bf16 on a 24GB card; `ab_decision.py` caches only the hooks
it needs. For fast iteration use `--model Qwen/Qwen2.5-3B-Instruct` (was in the
cache), confirm findings on 7B.

vLLM does not need to be running for Phase 2 — TransformerLens loads the weights
itself from `$HF_HOME`. Stop the `serve` tmux to free VRAM first.
