# Why do certain questions loop? (r1-distill-qwen-7b, T=0.6)

## A. AQuA: the model's own answer (no options shown) vs looping on the 5-option question

| own open-ended answer (majority over 4 seeds) | questions | 5-option runs that loop | questions that loop in >=1 seed |
|---|---|---|---|
| own answer = correct option | 164 | 1% (7/656) | 4% |
| own answer = a wrong option | 10 | 10% (4/40) | 20% |
| own answer matches NO option | 52 | 30% (63/208) | 46% |
| never answers (open form always loops) | 1 | 75% (3/4) | 100% |

## B. Same question, looped in one seed and finished in another: does the first stated answer already fit?

AQuA: first stated number is one of the options. LogiQA: first stated letter is the key. MATH: first stated result equals the gold answer. (Runs with no detectable claim excluded.)

| dataset | pairs | first answer fits: loop run | first answer fits: finished run | pairs where only the finished run's first answer fits | only the loop run's |
|---|---|---|---|---|---|
| aqua | 27 | 22% | 37% | 6 | 2 |
| logiqa | 65 | 38% | 45% | 16 | 12 |
| math500 | 51 | 6% | 14% | 6 | 2 |
