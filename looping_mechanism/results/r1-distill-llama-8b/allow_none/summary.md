# Allow-None (r1-distill-llama-8b, T=0.6)

Same items, plus: "If none of the options is correct, put None within \boxed{}." Baseline = the same seeds without that sentence.

| set | valid option? | seeds | runs | loops: before -> allow-None | answers None | picks a letter | correct: before -> allow-None |
|---|---|---|---|---|---|---|---|
| math_catch | **no** | [0, 1] | 300 | 71% -> **67%** | 8% | 20% | n/a |
| gsm8k_catch | **no** | [0, 1] | 400 | 62% -> **60%** | 17% | 22% | n/a |
| aqua_minus_gold | **no** | [0, 1] | 454 | 60% -> **51%** | 14% | 33% | n/a |
| logiqa_minus_gold | **no** | [0, 1] | 400 | 15% -> **10%** | 11% | 78% | n/a |
| math_aligned | yes | [0, 1] | 300 | 8% -> **4%** | 0% | 96% | 92% -> 95% |
| gsm8k_aligned | yes | [0, 1] | 400 | 2% -> **2%** | 1% | 98% | 98% -> 98% |
| aqua_original | yes | [0, 1] | 454 | 8% -> **7%** | 1% | 92% | 84% -> 83% |
| logiqa_original | yes | [0, 1] | 400 | 9% -> **10%** | 4% | 85% | 53% -> 48% |
