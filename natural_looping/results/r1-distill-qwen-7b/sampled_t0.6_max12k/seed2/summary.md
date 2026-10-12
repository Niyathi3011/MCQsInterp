# Results: `natural_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/seed2`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| aqua | MCQ (5 options) | 254 | 84.6% | **7.9%** (±3.3) | 7.5% |
| logiqa | MCQ (4 options) | 500 | 51.6% | **10.4%** (±2.7) | 38.0% |
| math500 | open-ended | 500 | 92.2% | **4.0%** (±1.7) | 3.8% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| aqua | 2590 | 9492 | 6550 | 12288 |
| logiqa | 2788 | 5219 | 3284 | 12288 |
| math500 | 3042 | 9303 | 6161 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| aqua | 0.13 | 0.01 |
| logiqa | 0.88 | 0.00 |
| math500 | 0.00 | 0.00 |
