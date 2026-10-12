# Per-benchmark stop switch (R1-Distill-Qwen-7B)

Each benchmark gets its own stop switch, learned from its own catch-vs-aligned pairs,
and is tested on its own natural loops, which are held out. Code:
`looping_mechanism/bench_switch.py`. Run scripts: `controlled_looping/run_r1_catch_sets.sh`
(generation), `looping_mechanism/run_bench_live_gen.sh` (fresh live runs),
`looping_mechanism/run_bench_pipeline.sh` (all GPU stages for one benchmark).

## Design

- **Held out (test):** questions that loop naturally at this temperature, in any seed.
- **Learning:** catch looped vs aligned finished correctly, same question and seed, on the
  other questions only.
  - GSM8K / MATH: two-option MCQ, (A) correct number vs (A) wrong number, (B) a sentence
  - AQuA / LogiQA: original options vs correct option removed
- **Switch:** the 6 MLPs with the largest contribution to the stop-signal gap on the learning
  pairs, set once to the donor (their mean output right before `</think>` in the finished
  aligned runs). Also tested: the MATH six (27, 26, 25, 24, 22, 20) with the MATH donor.
- **Option A, replay:** each held-out natural loop is fed up to its commit point; the
  intervention is applied there and the answer generated fresh.
- **Option B, live:** the held-out questions are generated from scratch with new seeds
  (vLLM). A rule that only reads the text written so far fires on any run with no
  `</think>` after 6,000 thinking tokens, at the next "states an answer, then doubts it"
  point (no answer key used). Every run it fires on, loop or not, is continued from there
  under each intervention (TransformerLens). Runs it doesn't fire on keep their outcome.
- The trigger threshold was fixed in advance, not tuned.

## Files in `<bench>_<t0|t0.6>/`

| file | stage | content |
|---|---|---|
| config.json | all | model, temperature, trigger, caps, MLP sets, live seeds, when each stage ran |
| heldout.json | pairs | held-out question ids |
| pairs_stats.json | pairs | counts (held out, learning pairs, commit points found) |
| learn_pairs.jsonl | pairs | learning pairs: full loop and finished texts, commit point |
| test_loops.jsonl | pairs | one natural loop per held-out question, with commit point |
| live_items.jsonl | pairs | the held-out questions (input rows for the live runs) |
| signal_pairs.jsonl | signal | per learning pair: logit(`</think>`) at stop / commit, per-MLP gap, % recovered by patching |
| components.json | signal | own stop MLPs, MATH six, control MLP |
| donor.pt | signal | donor: mean MLP output (all layers) before `</think>` in finished aligned runs |
| replay.jsonl | replay | per held-out loop and condition: stop, new thinking, answer text, prediction, correct |
| live_runs/seed*.jsonl | live-gen | fresh runs, full text, outcome |
| live_triggers.jsonl | trigger | per fresh run: did the rule fire, where, and why (commit / paragraph) |
| live_branches.jsonl | live | per triggered run and condition: answer text, prediction, correct |
| summary.md | summary | all tables |
| logs/, live_runs/logs/ | | console logs of every stage |

`math_ref_donor.pt` (this folder): the MATH donor used in the earlier experiments (128
sampled catch-vs-aligned finished runs), shared by all benchmarks.
