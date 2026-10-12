# Results: `natural_looping/results/qwen3-4b-thinking-2507/sampled_t0.6_max12k/seed1`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| gsm8k | open-ended | 500 | 95.2% | **0.0%** (±0.0) | 4.8% |
| aqua | MCQ (5 options) | 254 | 87.4% | **3.9%** (±2.4) | 8.7% |
| logiqa | MCQ (4 options) | 500 | 73.6% | **3.8%** (±1.7) | 22.6% |
| math500 | open-ended | 500 | 84.6% | **14.4%** (±3.1) | 1.0% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| gsm8k | 1408 | 3290 | 3290 | – |
| aqua | 2759 | 7602 | 5472 | 12288 |
| logiqa | 4996 | 7490 | 6683 | 12288 |
| math500 | 4493 | 11986 | 7641 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| gsm8k | – | 0.00 |
| aqua | 0.00 | 0.00 |
| logiqa | 0.02 | 0.02 |
| math500 | 0.00 | 0.00 |
