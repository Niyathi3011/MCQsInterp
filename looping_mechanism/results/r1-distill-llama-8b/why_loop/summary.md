# Why do certain questions loop? (r1-distill-llama-8b, T=0.6)

## A. AQuA: the model's own answer (no options shown) vs looping on the 5-option question

| own open-ended answer (majority over 4 seeds) | questions | 5-option runs that loop | questions that loop in >=1 seed |
|---|---|---|---|
| own answer = correct option | 160 | 2% (8/320) | 4% |
| own answer = a wrong option | 8 | 25% (4/16) | 38% |
| own answer matches NO option | 51 | 20% (20/102) | 24% |
| never answers (open form always loops) | 8 | 19% (3/16) | 38% |
