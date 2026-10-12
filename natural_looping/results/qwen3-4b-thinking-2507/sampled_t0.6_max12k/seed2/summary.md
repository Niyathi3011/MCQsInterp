# Results: `natural_looping/results/qwen3-4b-thinking-2507/sampled_t0.6_max12k/seed2`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| gsm8k | open-ended | 500 | 95.6% | **0.2%** (±0.4) | 4.2% |
| aqua | MCQ (5 options) | 254 | 90.2% | **2.4%** (±1.9) | 7.5% |
| logiqa | MCQ (4 options) | 500 | 73.0% | **2.2%** (±1.3) | 24.8% |
| math500 | open-ended | 500 | 83.6% | **16.0%** (±3.2) | 0.4% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| gsm8k | 1411 | 4327 | 3947 | 12288 |
| aqua | 2987 | 7171 | 5556 | 12288 |
| logiqa | 4936 | 7169 | 6714 | 12288 |
| math500 | 4370 | 12179 | 7828 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| gsm8k | 0.00 | 0.00 |
| aqua | 0.00 | 0.00 |
| logiqa | 0.00 | 0.02 |
| math500 | 0.00 | 0.00 |
