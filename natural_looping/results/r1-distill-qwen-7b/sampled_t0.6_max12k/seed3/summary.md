# Results: `natural_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/seed3`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| aqua | MCQ (5 options) | 254 | 87.0% | **8.7%** (±3.5) | 4.3% |
| logiqa | MCQ (4 options) | 500 | 49.0% | **8.2%** (±2.4) | 42.8% |
| math500 | open-ended | 500 | 91.4% | **6.0%** (±2.1) | 2.6% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| aqua | 2613 | 9862 | 5010 | 12288 |
| logiqa | 2961 | 4839 | 3412 | 12288 |
| math500 | 3017 | 10469 | 6272 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| aqua | 0.08 | 0.00 |
| logiqa | 0.35 | 0.00 |
| math500 | 0.02 | 0.00 |
