# Per-benchmark stop switch: math, T=0.6

Learned from this benchmark's catch-vs-aligned pairs; tested on its natural loops (held out). Settings: `config.json`.

## Data

- catch questions skipped (held out): 22
- learning: no commit point: 26
- test loops: no commit point: 23
- questions: 500
- held-out questions (loop naturally): 57
- held-out questions tested: 57
- learning pairs: 116
- replay test loops: 57
- replay test loops with a commit point: 34

## Learning: the stop signal on this benchmark's catch-vs-aligned pairs

- pairs: 116
- logit(</think>): finished stop +32.9, loop commit +5.0, gap +27.9 (gap > 0 in 100%)
- MLP share of the gap: 91%
- own stop MLPs (top-6): [27, 26, 23, 22, 24, 25]; MATH six: [27, 26, 25, 24, 22, 20]; share of gap in own six: 77%, in MATH six: 72%
- gap recovered by patching at the commit point: own 81%, math 79%, control 0%

## Option A: replay on the held-out natural loops

One loop per held-out question, fed up to its commit point; answer generated fresh (T=0.6, answer cap 1200, thinking cap 256).

| condition | loops | stops | correct [95% CI] | correct (of stopped) | answer hits the cap |
|---|---|---|---|---|---|
| base | 34 | 3% | 0% [0-0] | 0% | 0% |
| force | 34 | 100% | 38% [24-56] | 38% | 50% |
| bias | 34 | 100% | 26% [12-41] | 26% | 71% |
| switch | 34 | 74% | 29% [15-44] | 40% | 48% |
| switch_math | 34 | 94% | 32% [18-47] | 34% | 59% |
| ctrl | 34 | 0% | 0% [0-0] | - | - |

## Option B: live test on the held-out questions (fresh runs)

- fresh runs: 114 (57 questions x seeds [10, 11]); loops 54; correct with no intervention 45%
- rule fires on 70 runs: 53 loops, 17 that would have finished (14 of them correctly)

Accuracy over ALL fresh runs. Runs the rule did not fire on keep their own outcome; so do triggered runs whose branch did not stop within 256 tokens (the intervention did not take effect: a loop stays a loop, a run that would finish keeps its answer). Only runs whose branches are done are counted.

| condition | runs | correct [95% CI] | loops left | of triggered: loops rescued | would-finish runs hurt (right -> wrong) |
|---|---|---|---|---|---|
| none | 114 | 45% [36-54] | 54 | - | - |
| force | 114 | 38% [29-46] | 1 | 1 | 9 |
| switch | 114 | 42% [33-51] | 4 | 2 | 5 |
| switch_math | 114 | 40% [32-49] | 3 | 2 | 7 |
