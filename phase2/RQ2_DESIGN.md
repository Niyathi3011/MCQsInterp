# RQ2 — Where / what is the "no correct option" representation?

Mechanistic study of R1-Distill's **non-termination** on multiple-choice math
questions.

## The phenomenon (from the behavioural run)

DeepSeek-R1-Distill-Qwen-7B, MATH-500 level ≥3, greedy, 12k-token cap.
Fraction of trials that hit the cap **without producing an answer**:

| question type | non-termination |
| --- | --- |
| open-ended | 8% |
| MCQ, a correct option exists (`aligned`/`swap`) | 10% |
| MCQ, **no** correct option (`catch`) | **65%** |

On `catch` the model computes the true answer (94% of traces contain it), then
loops — one sentence repeated 50–440× ("…the answer is 2… but 2 isn't an
option… maybe the problem meant…") — and never emits `</think>`.

**RQ2:** is "my computed answer is not among the options" encoded as a linear
direction? In which layer does it form? Which components use it to suppress the
end-of-think token?

Three experiments: **E2a** linear probe (find the direction + layer), **E2b**
logit lens on `</think>` (where the stop-decision breaks), **E2c** direct logit
attribution (which components suppress it).

---

## Study set

From `raw_r1.jsonl`, `mode == "cot"`, `system_name == "none"`. Per `source_idx`:

| variant | keep if | role |
| --- | --- | --- |
| `aligned` | not truncated **and** correct | probe **label 0**; the *terminated* trace for E2b/E2c |
| `swap` | not truncated **and** correct | probe **label 0** (correct answer in slot B → label 0 is position-balanced) |
| `catch` | truncated | probe **label 1**; the *looping* trace for E2b/E2c |

A problem enters only if all three survive → ~86 problems ("catch loops on a
problem it demonstrably solves").

`broken` and `decoy` are **excluded**: `broken` confounds "no valid option" with
"unsolvable / truncated question"; `decoy` isn't needed once the split is simply
correct-option-present vs absent.

---

## Stage A — prep (`prep_loop.py`, CPU, no model)

Selects the triples and writes `phase2/loop_pairs.jsonl`, one line per problem,
carrying the raw pieces (`question`, `option_A/B`, `reasoning`, `completion`,
`gold_value`, `truncated`) plus **character offsets into the `reasoning` string**:

- `ans_off` — first occurrence of `str(gold_value)` (the model states its answer)
- `loop_onset_off` — first backtrack phrase (`catch` only), regex over
  `wait / hold on / not an option / none of the / maybe the problem / …`

No tokenizer here — `loop_probe.py` does chat-templating and tokenisation so the
token indexing matches the model that will actually run.

### Reconstructing the exact sequence (done in `loop_probe.py`)

The behavioural run used vLLM `--reasoning-parser deepseek_r1`, which **stripped**
the `<think>…</think>` tags. Rebuild:

```
user  = apply_chat_template(question + "\n(A) …\n(B) …" + suffix, add_generation_prompt=True)
        (+ "<think>\n"  iff the template didn't already add it)
FULL  = user + reasoning                                   # catch: no closing tag
      | user + reasoning + "\n</think>\n\n" + completion    # aligned/swap: terminated
```

Assert `</think>` is a single token. Tokenise with `prepend_bos=False` (the
chat template already carries `<｜begin▁of▁sentence｜>`).

---

## Stage B — forward passes (`loop_probe.py`, GPU)

- Model: `DeepSeek-R1-Distill-Qwen-7B` via **TransformerLens** (append the name to
  `OFFICIAL_MODEL_NAMES`). 1.5B for fast iteration.
- **No regeneration.** Feed `FULL` truncated:
  - `aligned`/`swap`: the whole thing (~1–4k tokens).
  - `catch`: up to `loop_onset + catch_window` (~1.5–2.5k tokens). You need the
    onset and a little past it, not all 12k.
- One `run_with_cache` per (problem × variant), `names_filter` to just the hooks
  each experiment needs (`resid_post`, `ln_final.hook_scale`; `hook_z`,
  `hook_mlp_out` for E2c).
- ~2.5k tokens × 29 layers × 3584 × 2 B ≈ 0.5 GB/tensor → fine on a 24 GB card
  next to the 15 GB of weights.

---

## Stage C — E2a: linear probe

### Read-out positions (all *before* `loop_onset` in catch)

| name | token index | expectation |
| --- | --- | --- |
| `opt_end` | end of the user prompt | ~chance (problem not solved yet) — **confound check** |
| `r0+100` | 100 tokens into the reasoning | rising |
| `r0+300` | 300 tokens in | rising |
| `ans` | where `gold_value` first appears | high — "…and it's not option A" forms here |
| `ans+8` | 8 tokens after | high |

### Per (layer L ∈ 0..28, position P)

`X` = `resid_post[L]` at `P`: ~86 `catch` (y=1), ~86 `aligned` + ~86 `swap` (y=0),
shape ≈ (258, 3584). `groups` = `source_idx`.

```
GroupKFold(5) on source_idx
LogisticRegression(class_weight="balanced", C ∈ {0.003, 0.01, 0.03, 0.1}, max_iter=2000)
```

3584 dims / ~200 train rows → regularise hard; if still overfitting, PCA→50
(fit on train fold only) then logistic. Report **mean held-out balanced accuracy
± std**.

### Outputs

- Heat-map `bal-acc` over (layer × position). First cell above **~0.85** = where
  and when "no correct option" becomes linearly represented.
- `d_noopt[L]` = probe weight vector at the best `(L, P)`, refit on all data,
  L2-normalised → `phase2/d_noopt.npy` (feeds RQ3 patching, RQ4 steering).

### Confound / sanity checks

- `opt_end` probe should be ≈ chance. If it's already high, you're reading a
  surface option-text difference, not the computed state.
- Label shuffle within group → collapses to ~0.5.
- `aligned` vs `swap` (both label 0) → ~chance (probe isn't just reading slot).

### Null result

No `(L, P)` above ~0.65 → not a single linear direction here; move read-out
later, try a non-linear probe, or go straight to whole-activation patching (RQ3).

---

## Stage D — E2b: logit lens on `</think>`

`u = W_U[:, think_id]` (unembedding column for `</think>`).

### Matched positions

- `aligned`: `k_a` = token just before `</think>`.
- `catch`: the **same absolute index** `k_a` (model has "thought for the same
  number of tokens"), **and** `k_a + 400` (well inside the loop).

### Per layer

`acc = accumulated_resid(layer=L, pos_slice=k, apply_ln=True) · u` → curve
"how much does the model want to emit `</think>` as of layer L", per trace,
averaged over the 86 problems (per-problem CIs).

### Read it

- `aligned@k_a` should climb through layers ~18–28. `catch@k_a` and
  `catch@k_a+400` stay flat/negative. Divergence layer = where "should I stop"
  is computed and, in `catch`, blocked.
- Stat: Wilcoxon signed-rank on `(aligned − catch)` final-layer projection over
  the 86 pairs; **positive result = p < 0.01** with a clear divergence layer.

---

## Stage E — E2c: direct logit attribution onto `</think>`

`logit(</think>) = Σ_c [ (component_c output, LN-scaled) · u ]` — exact.

- Per head: `stack_head_results(pos_slice=k, apply_ln=True) · u` → (L·H,).
- Per MLP: `mlp_out[L][k] / ln_final_scale · u`.
- For each component `c`: `Δ_c = mean_pairs( contrib_c(aligned) − contrib_c(catch) )`.
  Sort descending. Large `Δ_c` = supports stopping in `aligned` but not `catch`
  (or actively suppresses in `catch`).

### Outputs

- Top ~15 heads + ~8 MLPs by `Δ_c`, with mean contribution per condition.
- Check `Σ_c Δ_c ≈ logit(</think>)_aligned − logit(</think>)_catch`.
- These are the **ablation targets for RQ3 (E3c)**.
- (Optional) attention pattern of the top 2–3 heads at `k` in `catch` — do they
  attend to the option tokens / `<think>` / `\n\n`? If so, suppression is
  *conditioned on the options*.

---

## Statistical standards

| exp | positive result |
| --- | --- |
| E2a | held-out bal-acc ≥ ~0.85 at some `(L, P)`; label-shuffle at chance; `aligned`-vs-`swap` at chance |
| E2b | `aligned` `</think>` projection > `catch` at the final layer, Wilcoxon p < 0.01 over 86 pairs, clear divergence layer |
| E2c | < 10 components account for most of the `</think>` logit gap |

---

## Iteration & compute

1. **Smoke:** 7B, `--n 12`, catch window 1200, positions `{opt_end, ans, ans+8}`,
   all layers. Pipeline check + first divergence-layer read (~20 min on a 4090).
2. **Full:** all 86 pairs, full position sweep.
3. Drop to `-1.5B` only on OOM — it needs a small 1.5B behavioural run first
   (regenerate ~30 problems × {aligned, swap, catch}), so prefer subsetting 7B.

Requires the GPU free — stop the R1 vLLM server first.

---

## What RQ2 can / cannot conclude

- **Can:** "no correct option" is (or isn't) linearly encoded, at layer X,
  forming after the model states its answer; the `</think>` logit is actively
  suppressed from layer Y; components {…} do the suppressing.
- **Cannot:** that `d_noopt` *causes* the loop. Probing = information present;
  DLA = a correlate. Causation is RQ3 — patch `d_noopt` / the components and see
  if the loop starts or stops.

---

## Files

| file | stage | GPU |
| --- | --- | --- |
| `prep_loop.py` | A — build `loop_pairs.jsonl` | no |
| `loop_probe.py` | B–E — `--experiment {e2a,e2b,e2c,all}` | yes |
| `loop_pairs.jsonl` | study set | — |
| `d_noopt.npy` | E2a output → RQ3/RQ4 | — |
