# Results: `natural_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/seed0`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| aqua | MCQ (5 options) | 254 | 86.6% | **7.5%** (±3.2) | 5.9% |
| logiqa | MCQ (4 options) | 500 | 49.2% | **9.4%** (±2.6) | 41.4% |
| math500 | open-ended | 500 | 90.6% | **5.6%** (±2.0) | 3.8% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| aqua | 2483 | 9664 | 6340 | 12288 |
| logiqa | 2797 | 5105 | 3474 | 12288 |
| math500 | 2988 | 9660 | 5786 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| aqua | 0.12 | 0.00 |
| logiqa | 0.43 | 0.00 |
| math500 | 0.01 | 0.00 |
