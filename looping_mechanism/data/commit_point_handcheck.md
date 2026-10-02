# Hand-check of detected commit points (`build_pairs.py`)

A commit point is correct if the looping run has just stated its answer (the
correct value, or its own final value) and the next sentence starts doubting it.
It is wrong if the "Wait" interrupts an intermediate step.

## Controlled pairs (catch loops, commit to the correct value)

- Random sample of 12 (seed 11, after the final detector version): 11 correct,
  e.g. "...So, c = 56. | Wait, so Frank answered 56 questions correctly..."
- The error (greedy_396) matched the stray "/ 2" in a shoelace formula to gold = 2.

## Natural AQuA / LogiQA (commit to the model's own answer)

- Random sample (seed 7): AQuA 4/5 correct, LogiQA 5/5 correct,
  e.g. "...exactly $9.00. ... since $9 is not an option | Wait, another approach..."

## Natural MATH-500: all 31 pairs checked

| commit via | pairs | correct commit points |
|---|---|---|
| gold (the model derives the correct answer, then doubts) | 13 | 13 |
| own value (most-claimed value) | 18 | ~6 |
| all | 31 | ~19 (61%) |

Correct own-value examples: #2 (math500_126, "the answer is z_2 = ..."),
#11 (math500_224, "... = 9/256"), #15 (math500_323, "... is 1/3"),
#18 (math500_400, "perhaps the answer is 14(5√2 - 7)"), #30 (math500_497, "= 7/2 = 3.5").

Wrong own-value points (an intermediate step is interrupted):
math500_101, math500_166, math500_168, math500_18 (borderline), math500_217,
math500_381, math500_422, math500_425, math500_458, math500_460, math500_481,
math500_486.

Consequence: MATH-500 results are reported both for all pairs and for the 13
gold-commit pairs (the reliable subset). Open-ended conclusions are harder to
detect than MCQ ones ("X is not an option" / "the answer is B").
