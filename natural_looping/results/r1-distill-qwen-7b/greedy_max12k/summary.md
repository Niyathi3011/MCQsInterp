# Results: `natural_looping/results/r1-distill-qwen-7b/greedy_max12k`

Loop = the model hit the token cap without finishing its reasoning (no answer).

| dataset | format | n | accuracy | **loop rate** (±95% CI) | wrong answer |
|---|---|---|---|---|---|
| gsm8k | open-ended | 500 | 91.4% | **4.2%** (±1.8) | 4.4% |
| aqua | MCQ (5 options) | 254 | 71.7% | **25.6%** (±5.4) | 2.8% |
| logiqa | MCQ (4 options) | 500 | 35.6% | **39.4%** (±4.3) | 25.0% |
| math500 | open-ended | 500 | 87.4% | **10.0%** (±2.6) | 2.6% |

## Trace length (tokens) - SOPHIA Fig. 2 view

| dataset | correct | incorrect (all) | of which: wrong answer | of which: loop |
|---|---|---|---|---|
| gsm8k | 1200 | 7025 | 2002 | 12288 |
| aqua | 2102 | 11456 | 3733 | 12288 |
| logiqa | 1892 | 8263 | 1919 | 12288 |
| math500 | 2717 | 10765 | 4909 | 12288 |

## Are loops repetitive?

Median fraction of the last 60 sentences that repeat an earlier one.

| dataset | looping traces | finished traces |
|---|---|---|
| gsm8k | 0.77 | 0.00 |
| aqua | 0.82 | 0.02 |
| logiqa | 0.80 | 0.00 |
| math500 | 0.75 | 0.00 |
