# Allow-None (qwen3-4b-thinking-2507, T=0.6)

Same items, plus: "If none of the options is correct, put None within \boxed{}." Baseline = the same seeds without that sentence.

| set | valid option? | seeds | runs | loops: before -> allow-None | answers None | picks a letter | correct: before -> allow-None |
|---|---|---|---|---|---|---|---|
| math_catch | **no** | [0, 1] | 300 | 32% -> **18%** | 82% | 0% | n/a |
| gsm8k_catch | **no** | [0, 1] | 400 | 0% -> **0%** | 96% | 4% | n/a |
| aqua_minus_gold | **no** | [0, 1] | 454 | 9% -> **4%** | 75% | 21% | n/a |
| logiqa_minus_gold | **no** | [0, 1] | 400 | 10% -> **8%** | 18% | 74% | n/a |
| math_aligned | yes | [0, 1] | 300 | 10% -> **10%** | 2% | 88% | 90% -> 88% |
| gsm8k_aligned | yes | [0, 1] | 400 | 0% -> **0%** | 2% | 98% | 100% -> 98% |
| aqua_original | yes | [0, 1] | 454 | 3% -> **3%** | 6% | 91% | 88% -> 86% |
| logiqa_original | yes | [0, 1] | 400 | 3% -> **3%** | 2% | 95% | 75% -> 74% |
