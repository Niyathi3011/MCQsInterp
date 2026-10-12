# Results: `natural_looping/results/qwen3-4b-thinking-2507/sampled_t0.6_max12k/seed0`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| gsm8k | open-ended | 500 | 95.4% | **0.0%** (±0.0) | 4.6% |
| aqua | MCQ (5 options) | 254 | 88.6% | **3.1%** (±2.1) | 8.3% |
| logiqa | MCQ (4 options) | 500 | 74.4% | **3.4%** (±1.6) | 22.2% |
| math500 | open-ended | 500 | 84.4% | **15.0%** (±3.1) | 0.6% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| gsm8k | 1450 | 3448 | 3448 | – |
| aqua | 2844 | 7458 | 5617 | 12288 |
| logiqa | 5041 | 7241 | 6468 | 12288 |
| math500 | 4371 | 12044 | 5935 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| gsm8k | – | 0.00 |
| aqua | 0.02 | 0.00 |
| logiqa | 0.02 | 0.02 |
| math500 | 0.00 | 0.00 |
