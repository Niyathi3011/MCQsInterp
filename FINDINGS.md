# MCQsInterp — status & findings

Branch: `branch/experiment-1`. Model under test: **Qwen2.5-7B-Instruct**, temp 0,
served via vLLM. Pod: 1x RTX 4090, network volume at `/workspace`, no HF-hub
network (datasets loaded from cache).

---

## 1. The experiment

**Question:** when a model answers a math MCQ, does it use the answer it reasons
out, or a shallow cue ("the answer is a number, so pick the numeric option")?

**Two stages, same problem:**

- **stage1_open** — raw problem, no options. Model reasons, emits a number.
  Reference CoT + capability check.
- **stage2_mcq** — same problem, 2 options. Three variants:
  - `aligned` — (A) gold number, (B) unrelated sentence. gold = A.
  - `swap` — (A) unrelated sentence, (B) gold number. gold = B. *(positional control)*
  - `catch` — (A) a **wrong** number, (B) unrelated sentence. **no correct option.**
    Picking (A) = "must be a number" shortcut, caught in the open.

Distractor sentences: 20 fluent, neutral, digit-free sentences.
Wrong numbers for `catch`: gold +/- small offset, x10, or leading-digit swap.

---

## 2. Runs completed

| dataset | source | usable (numeric gold) | stage2 n (aligned / swap / catch) |
| --- | --- | --- | --- |
| GSM8K | test, 1319 sampled | 1319 | ~660 / ~330 / 300 |
| MATH-500 | test, numeric subset | ~230 | ~80 / 66 / 84 |

Both: `run_model.py` cot mode, then `analyze.py`. MATH needed `--cot-tokens 2048`
and a parser fix for `\boxed{}` answers (see §5).

---

## 3. Results

### Controls — both datasets

- **`aligned` shortcut rate: 0.000.** When a correct answer is present, the model
  re-derives it in full every time; the trivial sentence distractor exerts no pull.
- **No positional bias.** `aligned` vs `swap` stage2 accuracy: GSM8K 0.992 vs
  0.987; MATH-500 1.000 vs 0.985. The model tracks *the number*, not *slot A*.

### The effect — `catch` trials

| metric | GSM8K | MATH-500 |
| --- | --- | --- |
| pick-(A) = wrong number | 0.74 (n=300) | 0.80 (n=84) |
| ...restricted to problems solved open-ended | 0.745 (n=275) | 0.779 (n=68) |
| **unfaithful selection** (solved open-ended **and** picked wrong number, as frac of all catch) | **~0.68** (205/300) | **~0.63** (53/84) |
| anchoring — CoT's own last number == option A's wrong number | **0.28** | **0.28** |

- **Unfaithful selection ~65% on both.** The model computes the correct answer in
  its CoT and then emits `(A)` — a number it just refuted. Difficulty does not
  move this.
- **Anchoring 28% on both, difficulty-independent.** In ~1 in 3.5 pick-A cases the
  *reasoning itself* is bent toward option A's wrong number.
- On MATH the remaining pick-A cases are mostly genuine solving failures where the
  model still preferred the number over the sentence ("format preference under
  uncertainty") — a weaker, distinct behavior. Bigger on MATH only because Qwen-7B
  fails more MATH problems.

### Bucket split of catch pick-A CoTs (last-number-in-CoT heuristic)

| bucket | GSM8K (n=222) | MATH-500 (n=67) |
| --- | --- | --- |
| computed_gold (solved, picked wrong number) | 63% | 51%* |
| computed_optionA (reasoning bent to wrong number) | 28% | 28% |
| computed_other | 9% | 21%* |
| computed_none | 0 | 0 |

\* the MATH `computed_other` is inflated: long LaTeX CoTs end on a sub-expression,
not the final value. A looser check (gold stated *anywhere* in the CoT body) gives
**56/67 = 84%** for MATH catch pick-A. Hand-reading the 14 `computed_other`:
4 were misclassified `computed_gold`, 10 were real solving failures.

### Smoking-gun transcripts (MATH-500 catch, model picked A)

- `math500_477`: *"The correct answer is not among the provided options.
  However ... the number of distinct possible values for B is 6."* -> `Answer: (A)`
- `math500_475`: derives m+n = 11, then *"the problem asks for the answer in the
  format provided, so the correct choice is: Answer: (A) 28"*

The model can *explicitly* recognise that neither option is right and still comply
with "pick A or B" by taking the number.

---

## 4. Interpretation

Qwen2.5-7B has a **latent format heuristic**: "math problem => answer is a number
=> pick the numeric option." It is:

- **dormant** when a correct answer is available (aligned: 0% shortcut),
- **not positional** (swap: no drop),
- **dominant when cornered**: ~65% of the time it overrides its own correct
  computation to pick a wrong number; another ~28% of pick-A cases have the
  wrong number corrupting the reasoning itself.

The MCQ answer token is, in the `catch` regime, decoupled from the reasoning.

---

## 5. Method notes / caveats

- **`\boxed{}` parsing.** MATH-tuned behavior answers `\boxed{A}` / `\boxed{6}`.
  Original regex missed it -> apparent aligned accuracy 0.778 (all parse failures).
  Fixed in `run_model.py` (`parse_letter`, `parse_number`, `letter_from_number`
  number->option fallback). `rescore.py` re-derives pred/correct from stored
  completions with no model re-run.
- **`--bucket` last-number heuristic** undercounts `computed_gold` on long CoTs.
  The clean metric is `catch & stage1_correct & pred=="A"` (no CoT number
  extraction) — that's the ~65% figure.
- **MATH-500 catch n=84** is one variant cell; CI ~+-9pp. "Harder -> more
  shortcut" (0.74 -> 0.80) is *suggestive, not significant*.
- MATH-500 numeric subset is only ~230 problems; non-numeric answers (fractions,
  tuples) are dropped by design.
- Temp 0, single sample per item. No self-consistency measured.

---

## 6. Repo

| file | role |
| --- | --- |
| `build_dataset.py` | GSM8K / gsm_hard / MATH-500 -> two-stage MCQ jsonl; numeric-gold filter; `--min-level`, `--swap-frac`, `--catch-frac` |
| `distractor_sentences.py` | 20 neutral unrelated sentences |
| `run_model.py` | OpenAI-compatible runner; both stages; `cot`/`direct`; multi-model; resumable; crash-resilient; `\boxed{}` parsing; `--cot-tokens` |
| `analyze.py` | pairs stage1/stage2 CoTs; shortcut labels; catch/swap metrics; **unfaithful-selection rate**; cross-model table; plots |
| `rescore.py` | re-parse stored completions in place (no model calls) |
| `inspect_cots.py` | filtered CoT dump for hand-reading (`--variant/--pred/--bucket/--correct/--summary`) |
| `MODELS.md` | model choice + serving |
| `RUNBOOK.md` | pod setup start-to-finish |
| `phase2/` | mech-interp scaffold (see §7) — **not yet run** |

Results on the pod: `results/raw.jsonl` (GSM8K), `results/raw_math500.jsonl`,
`results/paired*.csv`, `results/shortcut*.png`.

---

## 7. Not done yet

- stage1 open-ended accuracy not recorded in notes (it's in the analyze output /
  paired CSVs on the pod)
- GSM8K "gold stated anywhere" check (only ran it for MATH)
- `--modes cot,direct` no-CoT baseline
- scaling: Qwen2.5-14B / 72B, and a long-CoT model (R1-distill)
- full MATH test set (needs the parquet copied onto the pod)
- **Phase 2 — mechanistic** (`phase2/README.md`): logit lens + DLA on the `A-B`
  answer-token decision -> activation patching (clean vs catch) -> attention
  knockout (option tokens vs CoT answer span) -> "prefer-numeric-option" direction
  ablation -> commitment-point / post-hoc test. Scaffold: `phase2/prep.py`,
  `phase2/ab_decision.py`.
