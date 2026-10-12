# Steps 2-3: is the withheld stop signal specific to loops?

logit(</think>) at: the finished run's stop; ORDINARY doubts in the finished run
(states an answer, doubts it, later stops); the looping run's repeated commits
(re-states its answer, doubts it, never stops). comp = direct contribution of the
stop-signal MLPs to the </think> logit.

## Step 2: loop doubts vs ordinary doubts

| group | pairs | finish stop | ordinary doubt (finish run) | loop commit #1 | loop - ordinary (paired) | comp: stop / ordinary / loop | </think> top-1 at ordinary doubts |
|---|---|---|---|---|---|---|---|
| controlled: catch loop vs aligned finish | 46 (46 with ordinary doubts) | +26.6 | -1.2 | -1.6 | -0.39 (p=3.5e-02) | +13.3 / +0.3 / +0.0 | 0% |
| controlled: catch loop vs catch finish | 36 (36 with ordinary doubts) | +27.5 | -1.1 | -1.5 | -0.37 (p=8.5e-02) | +13.8 / +0.4 / +0.1 | 0% |
| natural AQuA | 8 (6 with ordinary doubts) | +27.2 | +0.3 | -1.5 | -1.49 (p=3.1e-01) | +13.7 / +1.2 / +0.1 | 0% |
| natural LogiQA | 27 (26 with ordinary doubts) | +26.7 | +0.8 | -0.1 | -0.82 (p=1.8e-02) | +13.4 / +1.2 / +0.6 | 0% |
| natural MATH-500 | 17 (10 with ordinary doubts) | +27.0 | -0.1 | +0.2 | +0.19 (p=1.0e+00) | +13.5 / +0.9 / +1.1 | 0% |

## Step 3: the stop signal over the loop's repeated commits

Mean logit(</think>) at the k-th time the looping run re-states its answer and doubts it.

| group | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | pairs reaching #5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| controlled: catch loop vs aligned finish | -1.6 (n=46) | -1.5 (n=42) | -1.0 (n=36) | -0.9 (n=30) | -0.0 (n=22) | -0.2 (n=19) | -0.4 (n=13) | +0.8 (n=9) | -0.1 (n=6) | +0.4 (n=5) | 22 |
| controlled: catch loop vs catch finish | -1.5 (n=36) | -1.4 (n=35) | -1.0 (n=32) | -0.6 (n=27) | +0.1 (n=23) | -0.4 (n=19) | -0.6 (n=14) | +0.8 (n=9) | +0.3 (n=5) | - | 23 |
| natural AQuA | -1.5 (n=8) | -1.2 (n=7) | -1.2 (n=6) | - | - | - | - | - | - | - | 3 |
| natural LogiQA | -0.1 (n=27) | +0.8 (n=23) | +1.3 (n=21) | +1.2 (n=19) | +1.9 (n=16) | +1.3 (n=10) | +1.4 (n=8) | +2.0 (n=7) | +2.0 (n=6) | +1.4 (n=6) | 16 |
| natural MATH-500 | +0.2 (n=17) | - | - | - | - | - | - | - | - | - | 0 |
