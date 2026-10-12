# AQuA in three forms (same 227 questions, sampled T=0.6, 4 seeds)

| form | valid answer? | loop rate, mean [range] | accuracy, mean |
|---|---|---|---|
| open-ended (no options) | yes | **2.5%** [1.3-3.5] | 70.3% |
| original 5 options | yes | **8.5%** [8.4-8.8] | 86.2% |
| correct option removed (4 wrong) | **no** | **74.4%** [72.2-76.2] | n/a |

Correct option removed, runs that finished (232): pick one of the wrong letters 100%, no letter 0%.

Open-ended accuracy is scored with math-verify against the correct option's text, which
is not always a clean expression (units, words), so it is a lower bound.
