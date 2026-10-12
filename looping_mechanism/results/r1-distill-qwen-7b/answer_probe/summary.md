# Answer-correctness probe: can the hidden state tell whether stopping here is right?

6115 points (2532 correct) from 1594 runs; 2496 points in looping runs. Split: 5 folds by question.

| features | AUC (all points) | AUC (loop points only) |
|---|---|---|
| baseline: position in trace | 0.701 | 0.510 |
| baseline: index of the claim in the run | 0.588 | 0.509 |
| probe on layer 14 | 0.869 | 0.733 |
| probe on layer 20 | 0.874 | 0.758 |
| probe on layer 24 | 0.881 | 0.772 |
| probe on layer 27 | 0.869 | 0.750 |

## Stopping rule on looping runs (probe on layer 24, out-of-fold)

Stop at the first commit point where P(correct) > t; if none, let it loop (no answer).
Accuracy = the stated answer at the stopping point is correct (the forced-stop answer
tracks it: ~55-70% correct when it is right, ~4-21% when wrong).

| rule | loops stopped | correct, of stopped | correct, of all loops |
|---|---|---|---|
| stop at the first commit point | 100% | 21% | 21% |
| probe, t = 0.3 | 63% | 38% | 24% |
| probe, t = 0.5 | 54% | 40% | 22% |
| probe, t = 0.7 | 43% | 46% | 20% |
| probe, t = 0.8 | 35% | 47% | 17% |
| probe, t = 0.9 | 24% | 53% | 12% |

## Within-dataset check (layer 24, out-of-fold): is it more than 'which dataset is this'?

| group | points (all) | correct rate | AUC all points | loop points | AUC loop points |
|---|---|---|---|---|---|
| aqua | 1121 | 37% | 0.919 | 532 | 0.802 |
| logiqa | 1676 | 40% | 0.565 | 784 | 0.532 |
| math500 | 2179 | 36% | 0.953 | 778 | 0.870 |
| own150_aligned | 326 | 72% | 0.942 | 124 | 0.790 |
| own150_open | 562 | 39% | 0.944 | 194 | 0.803 |
| own150_swap | 251 | 80% | 0.940 | 84 | 0.717 |
