# Per-benchmark stop switch: aqua, T=0.6

Learned from this benchmark's catch-vs-aligned pairs; tested on its natural loops (held out). Settings: `config.json`.

## Data

- catch questions skipped (held out): 33
- test loops: no commit point: 2
- questions: 254
- held-out questions (loop naturally): 35
- learning pairs: 170
- replay test loops: 35
- replay test loops with a commit point: 33

## Learning: the stop signal on this benchmark's catch-vs-aligned pairs

- pairs: 170
- logit(</think>): finished stop +32.9, loop commit +5.0, gap +27.9 (gap > 0 in 100%)
- MLP share of the gap: 90%
- own stop MLPs (top-6): [27, 26, 23, 24, 22, 25]; MATH six: [27, 26, 25, 24, 22, 20]; share of gap in own six: 76%, in MATH six: 71%
- gap recovered by patching at the commit point: own 82%, math 81%, control 0%

## Option A: replay on the held-out natural loops

One loop per held-out question, fed up to its commit point; answer generated fresh (T=0.6, answer cap 1200, thinking cap 256).

| condition | loops | stops | correct [95% CI] | correct (of stopped) | answer hits the cap |
|---|---|---|---|---|---|
| base | 33 | 0% | 0% [0-0] | - | - |
| force | 33 | 100% | 18% [6-33] | 18% | 48% |
| bias | 33 | 100% | 12% [3-24] | 12% | 48% |
| switch | 33 | 79% | 15% [3-27] | 19% | 50% |
| switch_math | 33 | 85% | 18% [6-33] | 21% | 43% |
| ctrl | 33 | 0% | 0% [0-0] | - | - |

## Option B: live test on the held-out questions (fresh runs)

- fresh runs: 140 (35 questions x seeds [10, 11, 12, 13]); loops 76; correct with no intervention 31%
- rule fires on 102 runs: 76 loops, 26 that would have finished (13 of them correctly)

Accuracy over ALL fresh runs. Runs the rule did not fire on keep their own outcome; so do triggered runs whose branch did not stop within 256 tokens (the intervention did not take effect: a loop stays a loop, a run that would finish keeps its answer). Only runs whose branches are done are counted.

| condition | runs | correct [95% CI] | loops left | of triggered: loops rescued | would-finish runs hurt (right -> wrong) |
|---|---|---|---|---|---|
| none | 140 | 31% [24-39] | 76 | - | - |
| force | 140 | 39% [31-47] | 0 | 13 | 3 |
| bias | 140 | 39% [31-46] | 0 | 14 | 4 |
| switch | 140 | 41% [33-49] | 22 | 14 | 1 |
| switch_math | 140 | 41% [34-50] | 13 | 16 | 1 |
| ctrl | 140 | 31% [24-39] | 76 | 0 | 0 |
