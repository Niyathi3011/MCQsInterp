# Which model to run this on

## What the experiment needs from a model

1. **Visible chain-of-thought** — you analyse the CoT tokens, so the reasoning
   trace must be in the output, not hidden or summarised.
2. **Decent GSM8K accuracy** — the headline conditions on "solved open-ended", so
   a weak model just shrinks your sample.
3. **Not frontier-strong** — a top model solves everything and the shortcut
   effect can vanish. The interesting regime is "capable but takes the bait".
4. **Open weights** — Phase 2 (probing / activation patching for *where* the
   answer is decided) needs the weights and activations. API-only = dead end.
5. **Self-hostable at temp 0** — thousands of deterministic calls, cheap.

## Recommendation

| role | model | why |
| --- | --- | --- |
| **primary** | `Qwen/Qwen2.5-7B-Instruct` | strongest small open math model (~85% GSM8K), clean visible CoT, 1 GPU |
| alt primary | `meta-llama/Llama-3.1-8B-Instruct` | best interp-tooling ecosystem (SAEs, TransformerLens, nnsight) |
| alt primary | `google/gemma-2-9b-it` | pick this if Phase 2 is serious — Gemma Scope SAEs are the most mature |
| **contrast** | `Qwen/Qwen2.5-72B-Instruct` or `Llama-3.1-70B-Instruct` | does the shortcut scale away or persist? |
| optional | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | long-CoT distilled — does explicit reasoning training reduce the shortcut? `<think>` trace is visible |

Start with **Qwen2.5-7B-Instruct primary + Qwen2.5-72B-Instruct contrast**. The
comparison *is* the finding.

## Avoid for Phase 1

- `o1` / `o3` / `o4-mini` / Gemini "thinking" / Claude extended-thinking — the
  reasoning trace is hidden or summarised; you can't score the CoT.
- `gpt-4o` / Claude Sonnet as the *primary* — closed weights kill Phase 2, and
  they may be strong enough to mask the effect. Fine as an extra behavioural
  data point.

## No GPU? API-only fallback

`gpt-4o-mini` primary, `gpt-4o` contrast. Purely behavioural + CoT-text study;
Phase 2 mech-interp is off the table.

## Serving + running

vLLM (one server per model):

```bash
# terminal 1 - primary
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# terminal 2 - contrast (needs multi-GPU)
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8001 --tensor-parallel-size 4
```

Run each against its endpoint, appending to the same `results/raw.jsonl`
(the resume key is `(model, id, mode)`, so runs don't collide):

```bash
python run_model.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
python run_model.py --base-url http://localhost:8001/v1 --model Qwen/Qwen2.5-72B-Instruct
python analyze.py            # per-model blocks + a cross-model summary table
```

If one endpoint serves multiple model ids (e.g. an internal gateway), pass them
comma-separated in one call:

```bash
python run_model.py --base-url $GW --model "model-a,model-b,model-c"
```
