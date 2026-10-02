# Methodology: why reasoning models loop, and what stopping them does

This document records everything done so far: the datasets, every step and its
exact settings, which questions and pairs each step used, worked examples, the bugs
found and fixed, and the known weaknesses of the method. Results are summarised in
`looping_mechanism/RESULTS.md`; this file is about **how** they were obtained.

Status date: 2026-10-01. Running at that time: the AQuA-with-the-correct-option-removed
run (fix 1) and two extra intervention seeds (fix 2).

---

## 0. Model, infrastructure and definitions

| item | value |
|---|---|
| model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (28 layers, d_model 3584, GQA 28/4 heads), bf16 |
| generation | vLLM, OpenAI-compatible server, `--reasoning-parser deepseek_r1`, `--max-model-len 16384` |
| mechanism | TransformerLens 3.8 (`HookedTransformer`), RTX 3090 24 GB, 62 GB RAM container |
| main decoding | **sampled**: temperature 0.6, top-p 0.95 (DeepSeek's recommended setting), 4 seeds (0-3) |
| secondary decoding | greedy (temperature 0); used in the earliest runs and as a comparison |
| token budget | 12,288 new tokens |
| **loop** | a run that reaches the budget without emitting `</think>` (no answer) |
| seed | one independent sampled attempt at the same prompt (vLLM per-request seed) |

**Weight processing.** TransformerLens normally folds the RMSNorm weights into the next
layer, centers the unembedding and folds the value biases. Its built-in routine needs
~60 GB of RAM for this model (it copies the whole state dict in fp32 several times), so
the same transformations are applied in place on the GPU, one tensor at a time
(`phase2/loop_probe.py: _process_rms_inplace`). Checked against an unprocessed model
and an fp32 HuggingFace reference: processing adds no error beyond bf16 noise. The MLP
weights are stored in a transposed-contiguous layout so TransformerLens does not copy
them on every forward (`_mlp_weights_linear_layout`); numerics are unchanged.

---

## 1. Datasets

### 1a. Controlled conditions (`controlled_looping/`)
- **Source:** 150 MATH-500 problems (level >= 3, numeric answer), the exact items of the
  original greedy experiment (`data/force_r1.jsonl`).
- **Conditions** (each problem in every condition):

| condition | options | valid answer? |
|---|---|---|
| `open` | none | yes (the model writes it) |
| `aligned` | (A) correct number, (B) unrelated sentence | yes, A |
| `swap` | (A) unrelated sentence, (B) correct number | yes, B |
| `catch` | (A) **wrong** number, (B) unrelated sentence | **no** |

- **Unrelated sentences:** 20 neutral, digit-free sentences (`distractor_sentences.py`),
  e.g. "After the renovation, the theater added a side staircase but kept the original
  brass railings."
- **Wrong numbers:** drawn from gold ±1-3, ±10%, ×10, or a digit swap (`build_dataset.py:
  wrong_number`).
- **Prompts** (two versions, both run):
  - `boxed` (main, identical to the natural runs): question, options as `A) ...`, then
    "Please reason step by step, and put the letter of the correct option (A or B) within
    \boxed{}." (open: "...put your final answer within \boxed{}.")
  - `answerline` (byte-identical to the original greedy run): "End with a line formatted
    exactly as: Answer: (X)  where X is A or B." (open: "Answer: <number>")

### 1b. Natural benchmarks (`natural_looping/`)
The four datasets of the SOPHIA paper (Yu et al. 2026), **each in its native format**:

| dataset | source | split | rows used | format |
|---|---|---|---|---|
| GSM8K | `openai/gsm8k` main | test (1,319) | 500 random (seed 0) | open-ended, numeric |
| AQuA | `deepmind/aqua_rat` raw | test (254) | all 254 | MCQ, 5 options A-E |
| LogiQA | `lucasmccabe/logiqa` (parquet export) | test (651) | 500 random (seed 0) | MCQ, 4 options A-D |
| MATH-500 | `HuggingFaceH4/MATH-500` | test (500) | all 500 | open-ended, LaTeX |

Sizes follow the paper's Table 6; the paper gives no splits, subsets or prompts, so
those are our choices. Prompt: DeepSeek's R1 prompt ("Please reason step by step, and
put your final answer within \boxed{}."; for MCQs, the option letter in the box).

### 1c. AQuA forms: the manipulation on real questions (`controlled_looping/data/aqua_forms/`)
Same 227 AQuA items (27 with a "None of these" option excluded: removing the correct
option would make "None of these" correct):
- `mcq`: the original 5 options (= the natural run above)
- `minus_gold`: the correct option removed, the 4 wrong ones kept and relabelled A-D
- `open`: no options (scored against the correct option's text with math-verify)

### 1d. Earlier phase (context)
Qwen2.5-7B-Instruct on the same controlled conditions: it does not loop, it picks the
wrong number (~65% of solvable `catch` items).

---

## 2. Scoring
- Answer = the last `\boxed{...}` after `</think>` (balanced braces).
- MCQ: the option letter in the box; fallbacks: "the answer is (X)", then a stated number
  mapped to the option with that value.
- GSM8K: numeric match. MATH-500 and open AQuA: `math-verify` equivalence (timeouts off,
  see bug 3).
- `catch` / `minus_gold`: no correct answer exists; we record the letter picked.
- A loop has no answer and counts as incorrect.
- Every summary re-scores from the stored text with one parser (greedy and sampled
  alike), so parser changes never require re-generation.

---

## 3. The steps

Worked examples used throughout:
- **Example 1 (controlled):** Frank's aptitude test, answer 56; `catch` options A) 58,
  B) a sentence. Pair id `ctrl_212_seed0`.
- **Example 2 (natural):** AQuA #106, water lilies; the correct maths gives 48 days, which
  is not among the options (15 / 28 / 30 / 53 / 59; the dataset's key says D = 53, which
  looks wrong). Pair id `aqua_106`.

### Step 1: controlled loop rates
- **Questions:** all 150 problems x 4 conditions x 4 seeds x 2 prompts (+ the original
  greedy run). **No pairs.**
- **Measure:** loop rate per condition (mean and range over seeds); for finished `catch`
  runs, which option is picked.
- **Script:** `controlled_looping/run_r1_sampled.sh`, `controlled_looping/summarize.py`.
- **Example 1:** `catch` loops: "...So, c = 56. Wait, so Frank answered 56 ...
  Alternatively, maybe the correct answer is A)58, but that's incorrect ..." (repeated
  to the budget).

### Step 2: natural loop rates
- **Questions:** all rows of the four benchmarks; greedy (all four) and sampled x 4 seeds
  (AQuA, LogiQA, MATH-500; GSM8K only greedy). **No pairs.**
- **Measure:** loop rate, accuracy, trace length of correct vs incorrect runs (the
  paper's Fig. 2 view), repetitiveness of loops (share of the last 60 sentences that
  repeat an earlier one), share of loops that say "not among the options".
- **Scripts:** `natural_looping/run_r1_{greedy,sampled}.sh`, `summarize.py`,
  `matched_pairs.py`.
- **Example 2:** seed 1 loops ("I think the answer is 59. But I don't know. Wait, ...").

### Step 3: matched pairs
One looping run and one finished run of the same question; one pair per question
(`looping_mechanism/build_pairs.py`).

| group | looping run | finished partner | candidate | after commit-point filter |
|---|---|---|---|---|
| G1 controlled, greedy | `catch` (original greedy run, answerline prompt) | `aligned`, same problem | 94 | **73** |
| G2 controlled | `catch`, sampled | `aligned`, same problem, same seed; `aligned` correct and `open` solved | 150 | **128** |
| G3 controlled | `catch` looping in one seed | `catch` of the same problem that finished in another seed | 86 | **71** |
| G4 natural AQuA | the question in one seed | the same question finished in another seed | 30 | **28** |
| G5 natural LogiQA | " | " | 116 | **95** |
| G6 natural MATH-500 | " | " | 51 | **31** |

Total **426** pairs (272 controlled, 154 natural).

**Why other seeds for the natural partner:** a natural question has no `aligned` twin;
the only way to see the identical prompt both loop and finish is another sampled attempt.
Same prompt, same model: only the trajectory differs.

**Not included:** natural questions that looped in all 4 seeds (AQuA 5, LogiQA 4,
MATH-500 6; no finished partner), loops without a detectable commit point, GSM8K, and
the 4-5% loops in the controlled `open` / `aligned` / `swap` conditions.

### Step 4: positions
- **Finished run:** the token right before `</think>`. The sequence is rebuilt exactly as
  generated: chat template + reasoning + `</think>` + answer (see bug 1).
- **Looping run: the commit point**, the token right before the first sentence-initial
  doubt phrase ("Wait", "But hold on", "Maybe", ...) whose preceding sentence states the
  answer the model commits to:
  - controlled (G1-G3): the correct value, stated at the end of a sentence with a
    conclusion cue ("... so P(5) = 15.");
  - natural (G4-G6): the value the looping run claims most often, stated with a strong
    conclusion cue ("the answer is", `\boxed`, "option X", "... is not an option", "so the
    value/sum/... is"); open MATH tries the gold value first.
- Character offsets are mapped to tokens with the tokenizer's offset mapping.
- **Example 1:** "...So, c = 56. ⟦commit⟧ Wait, so Frank answered ..."
  **Example 2:** "...but 57 isn't an option. ⟦commit⟧ Wait, 57 is close to 59 ..."
- **Hand-check** (`looping_mechanism/data/commit_point_handcheck.md`, script
  `check_commit_points.py`): controlled 11/12; AQuA 4/5; LogiQA 5/5; MATH-500 19/31
  (all 13 gold-commit pairs correct, ~6/18 own-value commits correct). Two rounds of
  detector tightening (sentence-initial doubts only; strong cues for own values; value at
  the sentence end for gold).
- After the fix, `</think>` is the top-1 next token at every finished position (P ~ 1.00)
  and "Wait" (or "Maybe") follows every commit point.

### Step 5: where the stop signal is written
- **Pairs:** G1-G6 (all 426). Script: `looping_mechanism/stop_signal.py`.
- At each of the two positions, one forward pass (chunked KV-cache prefill, then the
  position's token):
  - `</think>` logit, log-probability, top-1;
  - **logit lens:** the residual stream after every layer, final RMSNorm applied, projected
    on the `</think>` unembedding;
  - **direct logit attribution:** each attention layer's and MLP's output at the position,
    divided by the final RMSNorm scale, dotted with the (centered) `</think>` unembedding.
- **Gap** = finished minus looping; per-MLP gap contributions; Spearman correlation of the
  28-layer MLP profile with G1; Wilcoxon signed-rank test per group.
- **Example 1:** `</think>` logit 34.2 (finish) vs 4.8 (loop); MLPs 20-27 carry 25.6 of the
  26.9-logit MLP gap. **Example 2:** 35.2 vs 7.0; MLPs 20-27 carry 24.9 of 26.3.

### Step 6: patching (causal test)
- **Pairs:** G1-G6. At the looping run's commit token only (frozen KV cache), replace the
  output of MLPs 20, 22, 24, 25, 26, 27 with:
  - **own:** the same MLPs' output at the pair's finished position;
  - **controlled donor:** the mean over G2's 128 finished positions (one vector per layer,
    the same for every pair);
  - **control:** the own patch applied to MLP 15 instead.
- **Measure:** share of the finish-loop logit gap recovered.
- **Example 1:** 4.8 -> 27.8 (own), 26.8 (donor), 4.8 (MLP 15).
  **Example 2:** 7.0 -> 28.9, 27.0, 6.9.
- (A "greedy donor" column in the output was computed at the old, buggy position and is
  not used.)

### Step 7: is the gate loop-specific?
- **Pairs:** G1-G6. Script: `looping_mechanism/doubt_timecourse.py`.
- **Healthy doubts:** in the finished run, every sentence-initial doubt that follows a
  claim (the run later stops). **Repeated commits:** in the looping run, every doubt after
  a re-statement of its committed value (up to 10).
- `</think>` logit and the stop-MLPs' attribution at each, read in one chunked pass per run.
- **Example 1:** healthy doubts 5.1, 6.2, 2.7, 13.1; loop commits 4.8, 5.5, 6.1; real stop
  34.2. **Example 2:** healthy 1.8, 5.8, 6.6, 6.5, 1.8; loop 7.0, 6.7, 7.0, 8.2, ...; stop 35.2.

### Step 8: forcing the stop
- **Questions:** the looping runs of G4-G6 (all 154) + 60 random G2 pairs. Script:
  `looping_mechanism/stop_intervention.py --stage intervene`.
- **Start:** the looping run's token sequence up to and including the commit token.
- **Conditions** (one batch per pair, T=0.6, seed 0; seeds 1-2 running):

| condition | what is done at the commit token |
|---|---|
| `base` | nothing (thinking capped at 256 tokens; reference only) |
| `force` | `</think>` appended immediately (budget forcing) |
| `bias` | +b on the `</think>` logit while thinking; b = this pair's first-step logit increase under `mlp@1` |
| `mlp@1` | MLPs 20, 22, 24-27 replaced by the controlled donor (once, at the commit token) |
| `proj@1` | only the `</think>`-unembedding direction of that edit |
| `random@1` | an edit of the same size in a fixed random direction |
| `ctrl@1` | the `mlp@1` edit on MLP 15 |

- After `</think>`, the answer is generated unedited up to 1,200 tokens (covers >99% of
  normal answers) and scored.
- **Measures:** stop rate; answer runs to the cap (keeps reasoning after `</think>`); total
  new tokens; accuracy; for `catch`, the option picked. Split by whether the committed
  answer was correct (computed afterwards).
- The finished partner is used only for the "ceiling" accuracy; the intervention itself
  needs only the looping run.
- **Example 1, `mlp@1`:** stops at once; "Frank answered 56 questions correctly. **Answer:
  A) 58** Wait, hold on. I just got 56, but the first option is 58 ..."
  **Example 2, `mlp@1`:** "The correct answer is 48 days, but ... E) 59 is the closest but
  not correct ..." -> `\boxed{E}` (wrong). `proj@1` does not stop.

### Step 9: further checks
- **9a, left alone** (same 214 looping runs as step 8): the looping run up to its commit
  token is sent to vLLM as token ids and resampled 4 times (seeds 0-3), up to 4,000 new
  tokens. Does it stop by itself, and is it right? (`--stage continue`)
- **9b, repetition check** (G4-G6 looping runs): k = how many times the loop states its
  committed answer and then doubts it; does k predict that the answer is correct?
  (`results/r1-distill-qwen-7b/selective_stopping.md`)

### Fixes in progress
- **Fix 1:** AQuA `minus_gold` and `open`, 4 seeds each (step 1-style loop rates).
- **Fix 2:** step 8 with seeds 1 and 2 on the natural pairs; bootstrap 95% CIs over pairs.

---

## 4. Bugs found and fixed (all reported numbers are after the fixes)
1. **Stop position.** vLLM's reasoning text already ends in a newline; adding "\n" before
   `</think>` merged into tokens like ".\n\n" after which the model expects "Wait"
   (P(`</think>`) ~ 0). This invalidated the original greedy E3a static-patching logs
   (`phase2/loop_pairs.jsonl`); E2b and E3b/E3c used correctly rebuilt raw pairs. Redone in
   step 5-6 (group G1); the six-MLP result holds.
2. **Number-for-letter answers** ("Answer: 5" when option A is 5) were not mapped to the
   option: number -> option fallback added, everything re-scored.
3. **math-verify in worker threads:** its signal-based timeout raised in threads and was
   counted as wrong (MATH continuations 0% -> 74% correct after the fix); timeouts disabled.
4. **Memory:** TransformerLens processing OOM in RAM; attention prefill OOM on the GPU for
   ~11k-token prefixes (chunked prefill, 256-512 tokens, expandable segments, per-pair
   out-of-memory guard).

---

## 5. Known weaknesses of the methodology

### Design
1. **One model.** All mechanism and intervention results come from R1-Distill-Qwen-7B. A
   second model (e.g. Qwen3-4B-Thinking) is blocked on disk space.
2. **The stop-vs-loop comparison is partly built in.** Step 5 compares a position where the
   model *did* emit `</think>` with one where it *did* write "Wait". A large `</think>`
   gap there is close to guaranteed by the selection; what the step really shows is which
   components carry "stop vs continue". Step 7 is what makes this interpretable (healthy
   doubts look like loop doubts), and the claim should be phrased as a stop/continue gate,
   not as loop-specific suppression.
3. **Positions are not matched in context.** The finished run's stop comes at the end of a
   complete solution; the loop's commit point is usually earlier: median 9-10% into the
   12k-token trace for controlled loops (interquartile range 4-24%), 31-35% for natural
   loops (16-77%). Context length and reasoning stage differ between the two positions.
4. **The `catch` condition is artificial** (two options, an unrelated sentence, an
   instruction that forces "A or B"). Fix 1 (AQuA with the correct option removed) is the
   natural replacement; early seeds give 72-76% loops vs 8%.
5. **Prompt sensitivity.** `catch` loops 74.7% with the boxed prompt but 48.8% with the
   answerline prompt: the size of the effect depends on wording.
6. **AQuA has noisy answer keys** (e.g. #106: correct maths 48, key 53). Part of the
   "natural conflict" is caused by dataset errors, not only model errors.
7. **LogiQA is hard for this model** (50% accuracy); its loops mix the conflict with
   ordinary difficulty.

### Selection
8. **Natural pairs need a loop in one seed and a finish in another.** This keeps borderline
   questions and drops the most stuck ones (15 that loop in all 4 seeds).
9. **Commit-point filter.** Loops without a detectable commit are dropped (2 AQuA, 21 LogiQA,
   20 MATH-500, plus 58 controlled); these may be the least structured loops.
10. **The 4-5% loops in the controlled valid-answer conditions** (`open` / `aligned` / `swap`)
    were counted but never analysed in steps 3-9.
11. **GSM8K** was only run greedy.

### Measurement
12. **Commit-point detector is a regex heuristic:** about 90% correct on controlled, AQuA and
    LogiQA samples, 61% on MATH-500 (open-ended conclusions are hard to detect). The hand-checks
    are small (12 + 10 + 31).
13. **Direct logit attribution** measures only the direct path to the unembedding; indirect
    effects (an MLP whose output later layers read) are not captured. Attention heads were
    not analysed in the new pair analysis.
14. **Patching** replaces whole MLP outputs at one token, with donor values from a different
    context; recovering the logit is not the same as recovering behaviour (step 8 tests
    behaviour). Only MLPs 20-27 and one control layer were patched; no search for the
    upstream components that would open the gate.
15. **bf16:** late layers are sensitive to precision (on a test prompt, bf16 logits ~30% off
    an fp32 reference; top-1 agreement 80-87%).
16. **Answer cap (1,200 tokens)** after a forced stop: "keeps reasoning after `</think>`" is
    inferred from hitting the cap (supported by 11-13 "Wait"s on median), but long answers
    also hit it.
17. **`base` in step 8 is capped at 256 thinking tokens** and is not a fair token comparison;
    step 9a (4,000 tokens) is, but it is shorter than the original 12k budget.
18. **Scoring** relies on regex extraction and math-verify; checked against the original
    labels (100% agreement on 534 finished runs), but MCQ answers given only in words could be
    missed.

### Statistics
19. **One intervention sample per pair** so far (seeds 1-2 running); no CIs yet in the
    intervention table.
20. **Small cells:** AQuA has 28 natural pairs; the "committed answer correct" split has n=4
    for AQuA. Many comparisons, no multiple-comparison correction.
21. **The repetition check counts repetitions over the whole loop** (it looks ahead), so it is
    not yet an online stopping rule.

### Interpretation
22. **"The conflict is upstream" is inferred, not measured.** We show the gate is generic and
    that opening it does not resolve the doubt; we do not identify where the conflict is
    represented (an earlier linear probe for "no correct option", phase 2 E2a, was null).

---

## 6. File map

| what | where |
|---|---|
| natural datasets, runs, summaries | `natural_looping/` (`build_datasets.py`, `generate.py`, `summarize.py`, `matched_pairs.py`, `results/`) |
| controlled conditions, AQuA forms | `controlled_looping/` (`build_datasets.py`, `build_aqua_forms.py`, `summarize.py`, `results/`) |
| pairs, commit points, hand-check | `looping_mechanism/build_pairs.py`, `check_commit_points.py`, `data/` |
| steps 5-6 | `looping_mechanism/stop_signal.py` -> `results/r1-distill-qwen-7b/summary.md` |
| step 7 | `looping_mechanism/doubt_timecourse.py` -> `results/r1-distill-qwen-7b/doubt_timecourse/` |
| steps 8-9a | `looping_mechanism/stop_intervention.py` -> `results/r1-distill-qwen-7b/stop_intervention/` |
| step 9b | `results/r1-distill-qwen-7b/selective_stopping.md` |
| run scripts | `looping_mechanism/run_step1_stop_intervention.sh`, `run_step1b_resume.sh`, `run_step23_doubt_timecourse.sh`, `run_fixes.sh` |
| shared mechanism code | `phase2/loop_probe.py` (model loading, processing), `phase2/rq3_resolve.py` (KV-cached batched generation with edits) |
| results, related work, paper | `looping_mechanism/RESULTS.md`, `RELATED_WORK.md`, `paper/naacl_short/` |
