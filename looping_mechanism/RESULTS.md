# Results: why reasoning models loop, and why forcing a stop does not fix it

Model: DeepSeek-R1-Distill-Qwen-7B (bf16), vLLM for generation, TransformerLens
for the mechanism. Unless marked "greedy", everything is sampled with DeepSeek's
recommended settings (T=0.6, top-p 0.95) and the boxed prompt. A **loop** = the
model hits the 12k-token cap without finishing its reasoning (no answer).

All planned runs are complete except a second model (blocked on disk space).

## 1. When do models loop? (behaviour)

### 1a. Controlled: remove the valid answer
Same 150 MATH-500 problems (level >= 3, numeric answers), 4 seeds
(`controlled_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/summary.md`).

| condition | options | loop rate, sampled [range] | greedy |
|---|---|---|---|
| open-ended | none | 4.2% [3.3-4.7] | 9.3% |
| aligned | (A) correct, (B) sentence | 5.3% [4.7-6.0] | 11.3% |
| swap | (A) sentence, (B) correct | 4.8% [3.3-6.0] | 8.0% |
| **catch** | (A) wrong number, (B) sentence | **74.7%** [70.0-78.7] | 75.3% |

- Removing the valid answer raises looping ~15x, and it is not a greedy artefact:
  sampling halves looping only where a valid answer exists.
- With the original "Answer: (X)" prompt, catch loops 48.8% (sampled) vs 2.5-5%
  for the other conditions: the instruction wording changes how often the
  conflict ends in a loop, not whether it does.
- When catch does finish, it picks (A) the wrong number 82% of the time (boxed
  prompt). Qwen2.5-7B-Instruct shows the same pick without looping (~65% of
  solvable items; earlier phase).

### 1b. Natural: the SOPHIA datasets in their native format
`natural_looping/results/r1-distill-qwen-7b/{greedy_max12k,sampled_t0.6_max12k}/summary.md`

| dataset | format | loop, greedy | loop, sampled (4 seeds) | accuracy, sampled |
|---|---|---|---|---|
| GSM8K | open | 4.2% | not run | – |
| MATH-500 | open | 10.0% | 5.2% [4.0-6.0] | 91.4% |
| AQuA | MCQ (5) | 25.6% | 7.9% [7.5-8.7] | 86.2% |
| LogiQA | MCQ (4) | 39.4% | 9.5% [8.2-10.4] | 50.0% |

- Greedy decoding inflates natural looping 2-4x; the sampled rates are the
  model's real behaviour.
- The same conflict appears naturally: 58% of AQuA greedy loops and 23/30 AQuA
  sampled matched-pair loops say the model's answer is not among the options.

### 1c. The manipulation on real questions: AQuA without the correct option
227 AQuA questions (27 with a "None of these" option excluded), sampled, 4 seeds
(`controlled_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/aqua_forms/summary.md`).

| form | valid answer? | loop rate, mean [range] | accuracy |
|---|---|---|---|
| open-ended (no options) | yes | 2.5% [1.3-3.5] | 70.3% (lower bound: scored against option text) |
| original 5 options | yes | 8.5% [8.4-8.8] | 86.2% |
| **correct option removed** (4 real wrong options) | **no** | **74.4%** [72.2-76.2] | n/a |

- The controlled effect reproduces on real questions with real distractors, at the
  same size as the synthetic catch condition (74.7%).
- Runs that finish without a valid option always pick a wrong letter (232/232);
  the model never declines.
- Options themselves add looping (open 2.5% -> MCQ 8.5%), and removing the valid
  one multiplies it ~9x.

## 2. Where is the stop decided? (mechanism)
426 matched loop/finish pairs; positions: the token before </think> in the
finished run vs the token before "Wait" after the looping run commits to its
answer (`looping_mechanism/results/r1-distill-qwen-7b/summary.md`).

| group | pairs | logit(</think>) finish -> loop | MLP share of gap | stop-MLP profile vs greedy (Spearman) | patch recovery (controlled donor) | control MLP15 |
|---|---|---|---|---|---|---|
| controlled, greedy (original) | 73 | +32.1 -> +5.9 | 91% | 1.00 | 85% | 0% |
| controlled: catch vs aligned | 128 | +32.7 -> +5.0 | 91% | 1.00 | 79% | 0% |
| controlled: catch vs catch | 71 | +31.2 -> +4.9 | 92% | 0.99 | 84% | 0% |
| natural AQuA | 28 | +32.5 -> +5.5 | 91% | 0.98 | 82% | -1% |
| natural LogiQA | 95 | +30.0 -> +6.4 | 90% | 0.98 | 92% | 1% |
| natural MATH-500 | 31 | +32.9 -> +5.1 | 90% | 0.98 | 79% | 1% |

- Top MLPs by contribution in every group: 27, 26, 23, 22, 24, 25; the original
  six (27, 26, 25, 24, 22, 20) carry 70-73% of the gap. Gap > 0 in 100% of pairs
  (p <= 4e-6 per group).
- Restoring those MLPs with the pair's own finish values recovers 76-80%; the
  controlled-experiment donor transfers to natural loops (79-92%).

### 2b. The gate is generic, not loop-specific (steps 2-3)
`looping_mechanism/results/r1-distill-qwen-7b/doubt_timecourse/summary.md`

| group | at the stop | healthy doubt (run later stops) | loop doubt |
|---|---|---|---|
| controlled, greedy | +31.9 | +6.4 | +5.9 |
| controlled, catch vs aligned | +32.5 | +5.3 | +4.9 |
| natural AQuA | +32.5 | +5.9 | +5.5 |
| natural LogiQA | +29.9 | +7.4 | +6.3 |
| natural MATH-500 | +33.0 | +6.3 | +5.3 |

- Loop doubts and healthy doubts are indistinguishable (paired differences ~0;
  LogiQA -1.0, p=0.007): the stop gate is closed whenever the model says "Wait".
- Across a loop's repeated commits, the signal stays flat in catch (+5-6 from the
  1st to the 10th) and creeps up in natural loops (LogiQA +6.3 -> ~+10), always far
  below the +30 of a real stop.
- So loops are not a suppressed gate: the model never reaches the state that
  opens it.
- **Position-matched control** (`doubt_timecourse/position_matched.md`): comparing
  loop doubts and healthy doubts at the same distance into the reasoning (5 position
  bins; regression with log-position and group), loop doubts are +0.35 logits higher
  (95% CI +0.08 to +0.62; stop-MLP contribution +0.18), against ~+26 for a real stop.
  The generic-gate conclusion does not come from loops and finished runs doubting at
  different stages.

## 3. Does stopping fix loops? (intervention)
Start = the looping run's commit point. `looping_mechanism/results/r1-distill-qwen-7b/stop_intervention/summary.md`

### 3a. Left alone (resampled from the commit point, 4 seeds, up to 4k tokens)
| group | stops by itself | median tokens to stop | correct when it stops | finished-run ceiling |
|---|---|---|---|---|
| natural AQuA | 20% | ~2,200 | 36% | 50% |
| natural LogiQA | 50% | ~1,700 | 40% | 71% |
| natural MATH-500 | 40% | ~2,200 | 74% | 90% |
| controlled catch | 3% | ~3,100 | n/a | n/a |

### 3b. Interventions at the commit point (3 sampling seeds per pair, 95% bootstrap CIs over pairs)
| | AQuA | LogiQA | MATH-500 |
|---|---|---|---|
| stops immediately (restore MLPs) | 89% | 99% | 89% |
| answer runs to the 1,200-token cap (keeps reasoning after </think>) | 45% | 31% | 63% |
| correct: restore MLPs | 12% [2-24] | 31% [24-39] | 31% [18-44] |
| correct: force </think> | 14% [4-26] | 34% [26-42] | 35% [22-51] |
| correct: logit boost | 24% [10-39] | 36% [28-44] | 39% [24-53] |
| ceiling (finishing runs, same questions) | 50% | 71% | 90% |

- Controls (only the </think> direction, a random vector, MLP 15) almost never stop
  the loop (0-7%); controlled catch loops: all three stops end 100%, 72-73% of answers
  run to the cap, wrong option picked 42-45%.
- The three ways of stopping overlap within their CIs: restoring the MLPs is just another
  way to stop, no better than forcing </think>.

### 3c. Stopping helps only when the committed answer was already right (3 seeds pooled)
| | committed answer | loops | correct after stop (force / bias / restore) | correct if left alone |
|---|---|---|---|---|
| MATH-500 | correct | 16 | 56 / 58 / 48% | 38% |
| | wrong | 15 | 13 / 18 / 13% | 22% |
| LogiQA | correct | 36 | 66 / 66 / 59% | 39% |
| | wrong | 59 | 15 / 18 / 14% | 9% |
| AQuA | correct | 4 | 42 / 58 / 50% | 6% |
| | wrong | 24 | 10 / 18 / 6% | 7% |

### 3d. And it cannot be targeted by repetition
How often the loop re-states its committed answer does not predict that the answer
is right (31-40% at every repetition count;
`looping_mechanism/results/r1-distill-qwen-7b/selective_stopping.md`).

### 3e. Stopping points chosen by the model itself (no text heuristic, no pairs)
All 460 natural looping runs (AQuA 80, LogiQA 190, MATH-500 104, and the 86 loops in
the controlled open / aligned / swap conditions). One forward pass per loop; at every
line break: P(</think>) and the stop MLPs' direct push on </think>
(`results/r1-distill-qwen-7b/model_stop_points/summary.md`).

- **Loops never come close to stopping:** the highest P(</think>) anywhere in a loop
  is ~0.000-0.001 (median), against ~1.0 at a real stop.
- **Where they come closest is where the stop MLPs push hardest:** +8.5 to +11.5 at
  the model-chosen point vs ~+5 at a typical line break (a real stop: ~+24). The
  `catch`-identified switch governs natural loops too, shown without any matched pair.
- The model's closest point rarely coincides with the text commit point (0-5% within
  50 tokens); it is usually later in the loop.
- Forcing </think> there gives accuracy similar to the commit point (AQuA 19 vs 15%,
  LogiQA 29 vs 36%, MATH-500 23 vs 28%) but less doubt continuing after </think> (AQuA
  34 vs 64%, LogiQA 24 vs 33%).
- **The catch-learned MLP switch on natural loops (60, at the model point):** stops
  100% (force 100%, MLP-15 control 17%); answers correct 33% (force 30%, control 8%).

### 3f. Can the model's hidden state tell whether stopping will be right?
6,115 "states an answer, then doubts it" points and real stops from 1,594 sampled runs
(all 460 loops + 1,134 finished), labelled by whether the stated answer is correct;
logistic regression on the residual stream, 5-fold split by question
(`results/r1-distill-qwen-7b/answer_probe/summary.md`).

| | AUC, all points | AUC, loop points |
|---|---|---|
| baseline: position in trace / claim index | 0.70 / 0.59 | 0.51 / 0.51 |
| probe, layer 24 (best) | **0.88** | **0.77** |

- Within each dataset (so not just "which dataset is this"): loop-point AUC MATH-500
  0.87, AQuA 0.80, controlled open/aligned/swap 0.80/0.79/0.72, **LogiQA 0.53 (chance)**.
  The model encodes whether its committed answer is right for maths, not for logic
  puzzles.
- **Selective stopping:** stopping at the first commit point gives 21% correct; stopping
  only where the probe predicts correct (P > 0.7) stops 43% of loops at 46% correct
  (P > 0.9: 24% of loops, 53% correct). Precision doubles, but total correct over all
  loops stays ~20%, since unstopped loops still give no answer. (Correct = the stated
  answer at the stopping point; the forced-stop answer tracks it, section 3c.)

### 3g. Live controller: probe as sensor, stop-MLPs as switch (negative)
All 460 natural loops replayed; at every "states an answer, then doubts it" point (in order,
online) the held-out probe gives P(correct); the controller stops at the first point with
P > 0.7, then the answer is generated (T=0.6): with the catch-learned MLP switch (1 sample)
or by forcing </think> (2 seeds) (`results/r1-distill-qwen-7b/live_controller/summary.md`).

| group | loops | controller stops | correct of stopped: switch / force | correct over all loops: controller / first commit / model point* / never |
|---|---|---|---|---|
| AQuA | 80 | 40% | 19 / 28% | 8 / 14 / 19 / 0% |
| LogiQA | 190 | 49% | 36 / 35% | 18 / 28 / 29 / 0% |
| MATH-500 | 104 | 40% | 24 / 27% | 10 / 14 / 23 / 0% |
| own150 open | 25 | 56% | 21 / 21% | 12 / 24 / 18 / 0% |
| own150 aligned | 32 | 53% | 41 / 56% | 22 / 28 / 64 / 0% |
| own150 swap | 29 | 45% | 46 / 50% | 21 / 38 / 48 / 0% |

\* the model point maximises P(</think>) over the whole loop (it looks ahead), so it is a
reference, not an online rule.

- The probe does pick better points by the STATED answer (40% correct at P > 0.7, vs 21% at
  the first commit point), but the answer generated after the forced stop is no more often
  correct than after stopping blindly: the doubt survives the stop even when the model
  "knows" its answer is right.
- Because it stops only ~half the loops, the controller's total accuracy is lower than
  stopping everywhere.
- The MLP switch is again no better than forcing </think> (sometimes worse).
- Conclusion: hidden states encode answer correctness, but a forced stop does not let the
  model act on it. This sharpens "stopping is not resolving" rather than providing a fix.

## 4. Bugs found and fixed (all reported numbers are after the fixes)
1. **Stop-position reconstruction.** Joining reasoning + "\n</think>" added a second
   newline (vLLM's reasoning text already ends with one), so the "stop" position was
   one where P(</think>) ~ 0. It affected the original greedy E3a static-patching logs
   (`phase2/rq3_e3a_*.log`, built from `phase2/loop_pairs.jsonl`); the E2b logit lens
   and E3b/E3c generation used the correctly rebuilt raw pairs. Redone correctly
   (section 2, "controlled, greedy"): the six-MLP result holds.
2. **Number-for-letter answers.** "Answer: 5" (when option A is 5) was not mapped to
   its option in the new scorer; added the number -> option fallback and re-scored
   every run (greedy and sampled) with one parser.
3. **math-verify in worker threads.** Its signal-based timeout raises in threads and
   was counted as "wrong" (MATH continuations showed 0% instead of 74%); timeouts off,
   all summaries re-score from the stored text.
4. **TransformerLens memory.** Weight processing peaked at ~60 GB RAM (container cap
   62 GB) and per-step MLP weight copies took 85% of decode time; replaced by in-place
   GPU processing (verified against an fp32 reference) and a copy-free weight layout.

## 5. Limitations
- **One model** (a second model is blocked on disk space).
- Commit points come from a heuristic detector: hand-checked 11/12 controlled,
  9/10 AQuA+LogiQA, 19/31 MATH-500 (all 13 gold-commit MATH pairs correct)
  (`looping_mechanism/data/commit_point_handcheck.md`).
- Interventions use three samples per pair; AQuA has 28 pairs.
- bf16: late layers are sensitive to precision (logits ~30% off an fp32 reference on a
  test prompt; top-1 agreement 80-87%).
- What opens the gate upstream is not identified.

## Where everything is
| what | where |
|---|---|
| natural datasets, runs, summaries | `natural_looping/` |
| controlled conditions (sampled), AQuA forms | `controlled_looping/` |
| matched pairs, commit points, hand-checks | `looping_mechanism/data/` |
| mechanism, doubts, interventions | `looping_mechanism/results/r1-distill-qwen-7b/` |
| related work | `looping_mechanism/RELATED_WORK.md` |
