# Steps 2-3: is the withheld stop signal specific to loops?

logit(</think>) at: the finished run's stop; ORDINARY doubts in the finished run
(states an answer, doubts it, later stops); the looping run's repeated commits
(re-states its answer, doubts it, never stops). comp = direct contribution of the
stop-signal MLPs to the </think> logit.

## Step 2: loop doubts vs ordinary doubts

| group | pairs | finish stop | ordinary doubt (finish run) | loop commit #1 | loop - ordinary (paired) | comp: stop / ordinary / loop | </think> top-1 at ordinary doubts |
|---|---|---|---|---|---|---|---|
| controlled, greedy (original pairs) | 73 (53 with ordinary doubts) | +31.9 | +6.4 | +5.9 | -0.46 (p=9.1e-01) | +23.8 / +5.7 / +5.3 | 0% |
| controlled: catch loop vs aligned finish | 128 (98 with ordinary doubts) | +32.5 | +5.3 | +4.9 | -0.34 (p=5.1e-01) | +24.5 / +5.1 / +4.7 | 0% |
| controlled: catch loop vs catch finish | 71 (70 with ordinary doubts) | +31.3 | +4.4 | +4.9 | +0.59 (p=4.7e-02) | +23.9 / +4.5 / +4.7 | 0% |
| natural AQuA | 28 (27 with ordinary doubts) | +32.5 | +5.9 | +5.5 | -0.49 (p=9.7e-01) | +24.3 / +5.5 / +5.2 | 0% |
| natural LogiQA | 95 (69 with ordinary doubts) | +29.9 | +7.4 | +6.3 | -1.03 (p=7.0e-03) | +22.1 / +6.5 / +5.7 | 0% |
| natural MATH-500 | 24 (10 with ordinary doubts) | +33.0 | +6.3 | +5.3 | -0.90 (p=4.3e-01) | +25.0 / +6.0 / +5.1 | 0% |

## Step 3: the stop signal over the loop's repeated commits

Mean logit(</think>) at the k-th time the looping run re-states its answer and doubts it.

| group | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | pairs reaching #5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| controlled, greedy (original pairs) | +5.9 (n=73) | +5.5 (n=43) | +5.3 (n=31) | +5.5 (n=23) | +6.0 (n=15) | +5.8 (n=13) | +5.3 (n=12) | +5.2 (n=10) | +6.2 (n=9) | +5.8 (n=9) | 15 |
| controlled: catch loop vs aligned finish | +4.9 (n=128) | +4.9 (n=107) | +4.9 (n=90) | +5.8 (n=69) | +5.5 (n=57) | +5.8 (n=39) | +5.7 (n=32) | +5.2 (n=29) | +5.9 (n=25) | +5.8 (n=21) | 57 |
| controlled: catch loop vs catch finish | +4.9 (n=71) | +5.1 (n=61) | +5.0 (n=54) | +5.8 (n=42) | +5.6 (n=36) | +5.7 (n=25) | +5.4 (n=22) | +5.2 (n=20) | +5.6 (n=17) | +6.2 (n=14) | 36 |
| natural AQuA | +5.5 (n=28) | +6.2 (n=24) | +6.0 (n=21) | +6.6 (n=18) | +7.1 (n=17) | +7.1 (n=16) | +6.2 (n=14) | +7.5 (n=11) | +7.4 (n=9) | +6.5 (n=9) | 17 |
| natural LogiQA | +6.3 (n=95) | +7.0 (n=75) | +7.3 (n=65) | +8.1 (n=61) | +8.4 (n=53) | +8.3 (n=48) | +9.1 (n=42) | +10.1 (n=39) | +9.4 (n=34) | +9.7 (n=32) | 53 |
| natural MATH-500 | +5.3 (n=24) | +5.3 (n=7) | - | - | - | - | - | - | - | - | 2 |
