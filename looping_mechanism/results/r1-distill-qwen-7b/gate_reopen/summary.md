# Does the stop gate reopen once the conflict can be resolved? (r1-distill-qwen-7b)

Read at the end of each sentence where the model says its answer is not among the options (first 5 per run). stop-MLP push = direct contribution of the model's stop MLPs [27, 26, 25, 24, 22, 20] to logit(</think>).

| group | runs | logit(</think>) at conflict statements: first / max over the run | </think> top-1 at any conflict statement | stop-MLP push (first) | at the real stop |
|---|---|---|---|---|---|
| catch loops | 60 | +2.3 / +5.9 | 0% | +2.5 | - |
| allow-None, answers None | 60 | +2.5 / +6.3 | 0% | +2.5 | +30.2 |
| allow-None, still loops | 60 | +2.2 / +6.1 | 0% | +2.3 | - |

By set (max logit at conflict statements):

| set | catch loops | allow-None, answers None | allow-None, still loops |
|---|---|---|---|
| math_catch | +6.4 (n=18) | +7.0 (n=14) | +6.6 (n=17) |
| aqua_minus_gold | +5.3 (n=19) | +5.2 (n=12) | +5.6 (n=26) |
| gsm8k_catch | +5.4 (n=17) | +6.3 (n=30) | +6.3 (n=15) |
| logiqa_minus_gold | +7.7 (n=6) | +6.9 (n=4) | +7.7 (n=2) |
