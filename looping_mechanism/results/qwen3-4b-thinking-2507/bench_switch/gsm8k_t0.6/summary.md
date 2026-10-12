# Per-benchmark stop switch: gsm8k, T=0.6

Learned from this benchmark's catch-vs-aligned pairs; tested on its natural loops (held out). Settings: `config.json`.

## Data

- learning: no commit point: 2
- questions: 500
- held-out questions (loop naturally): 1
- held-out questions tested: 1
- learning pairs: 0
- replay test loops: 1
- replay test loops with a commit point: 1

## Option A: replay on the held-out natural loops

One loop per held-out question, fed up to its commit point; answer generated fresh (T=0.6, answer cap 1200, thinking cap 256).

| condition | loops | stops | correct [95% CI] | correct (of stopped) | answer hits the cap |
|---|---|---|---|---|---|
| force | 1 | 100% | 0% [0-0] | 0% | 100% |
| switch | 1 | 0% | 0% [0-0] | - | - |
| switch_math | 1 | 0% | 0% [0-0] | - | - |

## Option B: live test on the held-out questions (fresh runs)

- fresh runs: 2 (1 questions x seeds [10, 11]); loops 0; correct with no intervention 50%
- rule fires on 1 runs: 0 loops, 1 that would have finished (0 of them correctly)

Accuracy over ALL fresh runs. Runs the rule did not fire on keep their own outcome; so do triggered runs whose branch did not stop within 256 tokens (the intervention did not take effect: a loop stays a loop, a run that would finish keeps its answer). Only runs whose branches are done are counted.

| condition | runs | correct [95% CI] | loops left | of triggered: loops rescued | would-finish runs hurt (right -> wrong) |
|---|---|---|---|---|---|
| none | 2 | 50% [0-100] | 0 | - | - |
| force | 2 | 50% [0-100] | 0 | 0 | 0 |
| switch | 2 | 50% [0-100] | 0 | 0 | 0 |
| switch_math | 2 | 50% [0-100] | 0 | 0 | 0 |
