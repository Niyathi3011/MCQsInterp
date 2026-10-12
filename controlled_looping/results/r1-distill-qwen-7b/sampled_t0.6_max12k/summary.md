# Controlled looping, sampled (T=0.6, top-p 0.95, 4 seeds) vs the original greedy run

Same 150 MATH-500 problems and items as the greedy catch experiment. Loop = hit the 12k-token cap without finishing (no answer). Sampled: mean over seeds [min-max].

## Prompt: `boxed` (identical to natural_looping)

| condition | options | loop rate, sampled | loop rate, greedy | accuracy, sampled | accuracy, greedy |
|---|---|---|---|---|---|
| open | open-ended, no options | **4.2%** [3.3-4.7] | 9.3% | 90.3% | 84.0% |
| aligned | (A) correct, (B) sentence | **5.3%** [4.7-6.0] | 11.3% | 94.7% | 85.3% |
| swap | (A) sentence, (B) correct | **4.8%** [3.3-6.0] | 8.0% | 95.0% | 91.3% |
| catch | (A) wrong number, (B) sentence | **74.7%** [70.0-78.7] | 75.3% | n/a | n/a |

`catch` runs that finished (152 sampled, 37 greedy):

| pick | sampled | greedy |
|---|---|---|
| A = the wrong number | 82% | 24% |
| B = the unrelated sentence | 7% | 3% |
| no letter | 11% | 73% |

## Prompt: `answerline` (identical to the greedy run)

| condition | options | loop rate, sampled | loop rate, greedy | accuracy, sampled | accuracy, greedy |
|---|---|---|---|---|---|
| open | open-ended, no options | **5.0%** [4.7-5.3] | 9.3% | 86.2% | 84.0% |
| aligned | (A) correct, (B) sentence | **5.0%** [3.3-8.0] | 11.3% | 93.5% | 85.3% |
| swap | (A) sentence, (B) correct | **2.5%** [2.0-3.3] | 8.0% | 97.0% | 91.3% |
| catch | (A) wrong number, (B) sentence | **48.8%** [48.0-49.3] | 75.3% | n/a | n/a |

`catch` runs that finished (307 sampled, 37 greedy):

| pick | sampled | greedy |
|---|---|---|
| A = the wrong number | 57% | 24% |
| B = the unrelated sentence | 10% | 3% |
| no letter | 33% | 73% |

