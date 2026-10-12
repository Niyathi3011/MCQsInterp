# Allow-None (r1-distill-qwen-7b, T=0.6)

Same items, plus: "If none of the options is correct, put None within \boxed{}." Baseline = the same seeds without that sentence.

| set | valid option? | seeds | runs | loops: before -> allow-None | answers None | picks a letter | correct: before -> allow-None |
|---|---|---|---|---|---|---|---|
| math_catch | **no** | [0, 1] | 300 | 76% -> **35%** | 44% | 17% | n/a |
| gsm8k_catch | **no** | [0, 1] | 400 | 72% -> **19%** | 66% | 14% | n/a |
| aqua_minus_gold | **no** | [0, 1] | 454 | 74% -> **35%** | 30% | 27% | n/a |
| logiqa_minus_gold | **no** | [0, 1] | 400 | 16% -> **8%** | 20% | 70% | n/a |
| math_aligned | yes | [0, 1] | 300 | 6% -> **5%** | 1% | 93% | 94% -> 93% |
| gsm8k_aligned | yes | [0, 1] | 400 | 2% -> **2%** | 1% | 97% | 98% -> 96% |
| aqua_original | yes | [0, 1] | 454 | 8% -> **6%** | 1% | 93% | 87% -> 85% |
| logiqa_original | yes | [0, 1] | 400 | 8% -> **7%** | 4% | 88% | 50% -> 49% |
