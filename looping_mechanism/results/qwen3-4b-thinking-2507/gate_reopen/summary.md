# Does the stop gate reopen once the conflict can be resolved? (qwen3-4b-thinking-2507)

Read at the end of each sentence where the model says its answer is not among the options (first 5 per run). stop-MLP push = direct contribution of the model's stop MLPs [28, 31, 32, 33, 34, 35] to logit(</think>).

| group | runs | logit(</think>) at conflict statements: first / max over the run | </think> top-1 at any conflict statement | stop-MLP push (first) | at the real stop |
|---|---|---|---|---|---|
| catch loops | 60 | -1.1 / +0.4 | 0% | +0.5 | - |
| allow-None, answers None | 60 | -0.9 / +1.0 | 0% | +0.6 | +26.1 |
| allow-None, still loops | 60 | -1.0 / +0.4 | 0% | +0.7 | - |

By set (max logit at conflict statements):

| set | catch loops | allow-None, answers None | allow-None, still loops |
|---|---|---|---|
| math_catch | +0.0 (n=29) | +1.1 (n=11) | +0.0 (n=29) |
| aqua_minus_gold | -0.0 (n=17) | +0.8 (n=22) | +0.2 (n=13) |
| gsm8k_catch | +1.6 (n=1) | +1.0 (n=24) | +0.8 (n=1) |
| logiqa_minus_gold | +1.6 (n=13) | +1.6 (n=3) | +1.0 (n=17) |
