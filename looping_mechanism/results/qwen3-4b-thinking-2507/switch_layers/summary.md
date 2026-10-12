# Where is the stop signal written, and what makes qwen3-4b-thinking-2507 stop?

46 MATH catch-vs-aligned pairs. logit(</think>): finished stop +26.6, loop commit -1.6, gap +28.2.

## 1. Direct contribution to the gap (share of the gap)

- all MLPs 59%, all attention 41% (the rest: embeddings / indirect)
- top MLPs: 35 (22%), 34 (11%), 33 (5%), 32 (3%), 28 (3%), 31 (2%), 27 (2%), 30 (2%), 29 (2%), 25 (2%), 26 (1%), 24 (1%)
- top attention: 34 (12%), 32 (8%), 35 (8%), 33 (5%), 29 (3%), 30 (2%), 22 (1%), 26 (1%)
- the six stop MLPs [28, 31, 32, 33, 34, 35]: 47%

## 2. Patching at the loop's commit token

own = this pair's finished-run values (upper bound); mean = the average over pairs (what a switch can use). top-1 = </think> becomes the most likely token (needed to stop).

| patched | gap recovered: own / mean | </think> top-1: own / mean | P(</think>): own / mean |
|---|---|---|---|
| 6 stop MLPs | 49% / 50% | 0% / 0% | 0.00 / 0.00 |
| 6 stop MLPs, alpha 2 | 55% / 57% | 67% / 65% | 0.12 / 0.14 |
| 6 stop MLPs, alpha 3 | 48% / 49% | 50% / 65% | 0.05 / 0.07 |
| top-12 MLPs | 69% / 71% | 13% / 7% | 0.11 / 0.06 |
| all late MLPs (18-35) | 80% / 83% | 54% / 93% | 0.55 / 0.69 |
| top-6 attention | 56% / 56% | 0% / 0% | 0.00 / 0.00 |
| 6 MLPs + top-6 attention | 82% / 84% | 17% / 13% | 0.15 / 0.14 |
| all late MLPs + attention | 99% / 103% | 100% / 100% | 1.00 / 1.00 |
| residual stream at layer 18 | 46% / 45% | 0% / 2% | 0.01 / 0.02 |
| residual stream at layer 21 | 100% / 102% | 96% / 100% | 0.94 / 0.98 |
| residual stream at layer 24 | 101% / 102% | 100% / 100% | 1.00 / 1.00 |
| residual stream at layer 27 | 101% / 103% | 100% / 100% | 1.00 / 1.00 |
| residual stream at layer 30 | 100% / 102% | 100% / 100% | 1.00 / 1.00 |
| residual stream at layer 33 | 100% / 103% | 100% / 100% | 1.00 / 1.00 |

## 3. Generation on 20 held-out natural loops (AQuA + LogiQA), from the commit point

| intervention | stops (writes </think> within 256 tokens) | correct |
|---|---|---|
| force | 100% | 10% |
| 6 stop MLPs | 0% | 0% |
| 6 stop MLPs, alpha 3 | 25% | 5% |
| top-12 MLPs | 45% | 5% |
| all late MLPs | 100% | 10% |
| all late MLPs, alpha 2 | 80% | 0% |
| 6 stop MLPs, every step | 65% | 15% |
