# Per-benchmark stop switch: gsm8k, T=0

Learned from this benchmark's catch-vs-aligned pairs; tested on its natural loops (held out). Settings: `config.json`.

## Data

- learning: no commit point: 36
- catch questions skipped (held out): 6
- test loops: no commit point: 3
- questions: 500
- held-out questions (loop naturally): 21
- held-out questions tested: 21
- learning pairs: 145
- replay test loops: 21
- replay test loops with a commit point: 18

## Learning: the stop signal on this benchmark's catch-vs-aligned pairs

- pairs: 145
- logit(</think>): finished stop +33.6, loop commit +5.0, gap +28.6 (gap > 0 in 100%)
- MLP share of the gap: 91%
- own stop MLPs (top-6): [27, 26, 23, 22, 24, 25]; MATH six: [27, 26, 25, 24, 22, 20]; share of gap in own six: 78%, in MATH six: 72%
- gap recovered by patching at the commit point: own 82%, math 79%, control 0%

## Option A: replay on the held-out natural loops

One loop per held-out question, fed up to its commit point; answer generated fresh (T=0, answer cap 1200, thinking cap 256).

| condition | loops | stops | correct [95% CI] | correct (of stopped) | answer hits the cap |
|---|---|---|---|---|---|
| force | 18 | 100% | 39% [17-61] | 39% | 39% |
| switch | 18 | 100% | 39% [17-61] | 39% | 44% |
| switch_math | 18 | 100% | 33% [11-56] | 33% | 17% |

## Option B: live test on the held-out questions (fresh runs)

- fresh runs: 21 (21 questions x seeds [None]); loops 11; correct with no intervention 38%
- rule fires on 11 runs: 11 loops, 0 that would have finished (0 of them correctly)

Accuracy over ALL fresh runs. Runs the rule did not fire on keep their own outcome; so do triggered runs whose branch did not stop within 256 tokens (the intervention did not take effect: a loop stays a loop, a run that would finish keeps its answer). Only runs whose branches are done are counted.

| condition | runs | correct [95% CI] | loops left | of triggered: loops rescued | would-finish runs hurt (right -> wrong) |
|---|---|---|---|---|---|
| none | 21 | 38% [19-57] | 11 | - | - |
| force | 21 | 57% [38-76] | 0 | 4 | 0 |
| switch | 21 | 57% [38-76] | 1 | 4 | 0 |
| switch_math | 21 | 57% [38-76] | 2 | 4 | 0 |
