# Step 1: restoring the stop signal at the looping run's commit point

DeepSeek-R1-Distill-Qwen-7B, sampled T=0.6. Start = the looping run's commit point
(it has just stated its answer; originally it continued with "Wait" and never
stopped: 0% correct). Natural pairs have a correct answer; controlled catch pairs
do not (we report which option is picked).

## A. No intervention: resampled from the commit point

Up to 4000 new tokens, several seeds per pair.

| group | runs | stops by itself | median tokens to stop | correct (all runs) | correct when it stops | finished-run ceiling |
|---|---|---|---|---|---|---|
| nat_aqua | 112 | 20% | 2197 | 7% | 36% | 50% |
| nat_logiqa | 380 | 50% | 1659 | 20% | 40% | 71% |
| nat_math500 | 124 | 40% | 2193 | 30% | 74% | 90% |
| ctrl_cross | 240 | 3% | 3127 | n/a | n/a | n/a |

Intervention seeds pooled: [0, 1, 2]. [95% CI] = bootstrap over pairs (each pair's mean over its seeds).

## B. Interventions at the commit point

stop = emitted </think> within the thinking cap; tokens = new thinking tokens before </think>; correct = the answer after </think> (natural groups; no stop = wrong); catch picks = which option a stopped controlled run chose.

### nat_aqua (28 pairs)

| condition | stops | median new thinking tokens | answer hits the 1200-token cap (keeps reasoning after </think>) | median total new tokens | correct (all) | correct (of stopped) |
|---|---|---|---|---|---|---|
| base | 4% | 108 | 0% | 256 | 1% [0-4] | 33% |
| force | 100% | 0 | 48% | 743 | 14% [4-26] | 14% |
| bias | 100% | 0 | 44% | 840 | 24% [10-39] | 24% |
| mlp@1 | 89% | 0 | 45% | 591 | 12% [2-24] | 13% |
| proj@1 | 7% | 139 | 0% | 256 | 1% [0-4] | 17% |
| random@1 | 1% | 145 | 0% | 256 | 0% [0-0] | 0% |
| ctrl@1 | 4% | 74 | 0% | 256 | 1% [0-4] | 33% |

finished-run ceiling (same questions): 50% correct

### nat_logiqa (95 pairs)

| condition | stops | median new thinking tokens | answer hits the 1200-token cap (keeps reasoning after </think>) | median total new tokens | correct (all) | correct (of stopped) |
|---|---|---|---|---|---|---|
| base | 3% | 159 | 0% | 256 | 1% [0-2] | 25% |
| force | 100% | 0 | 34% | 208 | 34% [26-42] | 34% |
| bias | 100% | 0 | 31% | 291 | 36% [28-44] | 36% |
| mlp@1 | 99% | 0 | 31% | 236 | 31% [24-39] | 31% |
| proj@1 | 5% | 69 | 0% | 256 | 1% [0-2] | 13% |
| random@1 | 7% | 80 | 0% | 256 | 1% [0-3] | 14% |
| ctrl@1 | 4% | 94 | 0% | 256 | 1% [0-2] | 18% |

finished-run ceiling (same questions): 71% correct

### nat_math500 (31 pairs)

| condition | stops | median new thinking tokens | answer hits the 1200-token cap (keeps reasoning after </think>) | median total new tokens | correct (all) | correct (of stopped) |
|---|---|---|---|---|---|---|
| base | 0% | - | - | 256 | 0% [0-0] | - |
| force | 100% | 0 | 60% | 1201 | 35% [22-51] | 35% |
| bias | 100% | 0 | 60% | 1201 | 39% [24-53] | 39% |
| mlp@1 | 89% | 0 | 63% | 1201 | 31% [18-44] | 35% |
| proj@1 | 0% | - | - | 256 | 0% [0-0] | - |
| random@1 | 1% | 167 | 0% | 256 | 1% [0-3] | 100% |
| ctrl@1 | 1% | 94 | 0% | 256 | 1% [0-3] | 100% |

finished-run ceiling (same questions): 90% correct

### ctrl_cross (60 pairs)

| condition | stops | median new thinking tokens | answer hits the 1200-token cap (keeps reasoning after </think>) | median total new tokens | picks A / B / neither (of stopped) |
|---|---|---|---|---|---|
| base | 0% | - | - | 256 | - |
| force | 100% | 0 | 73% | 1201 | 42% / 3% / 55% |
| bias | 100% | 0 | 73% | 1201 | 42% / 5% / 53% |
| mlp@1 | 100% | 0 | 72% | 1201 | 45% / 3% / 52% |
| proj@1 | 0% | - | - | 256 | - |
| random@1 | 0% | - | - | 256 | - |
| ctrl@1 | 2% | 238 | 0% | 256 | 100% / 0% / 0% |

