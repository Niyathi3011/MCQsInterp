# Results: `natural_looping/results/qwen3-4b-thinking-2507/sampled_t0.6_max12k/seed3`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| gsm8k | open-ended | 500 | 96.4% | **0.0%** (±0.0) | 3.6% |
| aqua | MCQ (5 options) | 254 | 88.2% | **2.0%** (±1.7) | 9.8% |
| logiqa | MCQ (4 options) | 500 | 71.6% | **3.6%** (±1.6) | 24.8% |
| math500 | open-ended | 500 | 83.8% | **15.8%** (±3.2) | 0.4% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| gsm8k | 1416 | 3195 | 3195 | – |
| aqua | 2828 | 7178 | 6156 | 12288 |
| logiqa | 4932 | 7463 | 6763 | 12288 |
| math500 | 4356 | 12092 | 4346 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| gsm8k | – | 0.00 |
| aqua | 0.00 | 0.00 |
| logiqa | 0.02 | 0.02 |
| math500 | 0.00 | 0.00 |
