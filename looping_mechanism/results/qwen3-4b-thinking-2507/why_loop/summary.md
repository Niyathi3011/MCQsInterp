# Why do certain questions loop? (qwen3-4b-thinking-2507, T=0.6)

## A. AQuA: the model's own answer (no options shown) vs looping on the 5-option question

| own open-ended answer (majority over 4 seeds) | questions | 5-option runs that loop | questions that loop in >=1 seed |
|---|---|---|---|
| own answer = correct option | 165 | 0% (2/660) | 1% |
| own answer = a wrong option | 8 | 9% (3/32) | 12% |
| own answer matches NO option | 52 | 10% (20/208) | 13% |
| never answers (open form always loops) | 2 | 12% (1/8) | 50% |

## B. Same question, looped in one seed and finished in another: does the first stated answer already fit?

AQuA: first stated number is one of the options. LogiQA: first stated letter is the key. MATH: first stated result equals the gold answer. (Runs with no detectable claim excluded.)

| dataset | pairs | first answer fits: loop run | first answer fits: finished run | pairs where only the finished run's first answer fits | only the loop run's |
|---|---|---|---|---|---|
| aqua | 6 | 33% | 50% | 1 | 0 |
| logiqa | 24 | 29% | 46% | 5 | 1 |
| math500 | 59 | 2% | 7% | 3 | 0 |
