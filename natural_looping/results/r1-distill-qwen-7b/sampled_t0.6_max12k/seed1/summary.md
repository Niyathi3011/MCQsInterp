# Results: `natural_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/seed1`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| aqua | MCQ (5 options) | 254 | 86.6% | **7.5%** (±3.2) | 5.9% |
| logiqa | MCQ (4 options) | 500 | 50.4% | **10.0%** (±2.6) | 39.6% |
| math500 | open-ended | 500 | 91.4% | **5.2%** (±1.9) | 3.4% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| aqua | 2692 | 9844 | 6749 | 12288 |
| logiqa | 2799 | 5131 | 3324 | 12288 |
| math500 | 3054 | 9829 | 6068 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| aqua | 0.82 | 0.00 |
| logiqa | 0.11 | 0.00 |
| math500 | 0.02 | 0.00 |
