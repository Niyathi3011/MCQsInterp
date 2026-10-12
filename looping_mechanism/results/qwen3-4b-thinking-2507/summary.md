# The </think> stop signal: controlled vs natural loops

Qwen3-4B-Thinking-2507. Each pair: one run that FINISHED and one that LOOPED.
finish@stop = token before </think> in the finished run; loop@commit = token
before the doubt phrase after the looping run commits to its answer. Controlled
and natural pairs are sampled (T=0.6, boxed prompt) unless marked greedy.
`strict` keeps pairs whose finished run had </think> as its top-1 token.

## 1. Is </think> suppressed at the looping run's commit point?

| group | pairs | logit(</think>) finish | loop | **gap** | pairs gap>0 | </think> top-1: finish / loop | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| controlled: catch loop vs aligned finish | 46 | +26.64 | -1.59 | **+28.22** | 100% | 100% / 0% | 3.5e-09 |
| controlled: catch loop vs aligned finish (strict) | 46 | +26.64 | -1.59 | **+28.22** | 100% | 100% / 0% | 3.5e-09 |
| controlled: catch loop vs catch finish | 36 | +27.47 | -1.48 | **+28.94** | 100% | 100% / 0% | 1.7e-07 |
| controlled: catch loop vs catch finish (strict) | 36 | +27.47 | -1.48 | **+28.94** | 100% | 100% / 0% | 1.7e-07 |
| natural AQuA | 8 | +27.14 | -1.52 | **+28.66** | 100% | 100% / 0% | 7.8e-03 |
| natural AQuA (strict) | 8 | +27.14 | -1.52 | **+28.66** | 100% | 100% / 0% | 7.8e-03 |
| natural LogiQA | 27 | +26.72 | -0.07 | **+26.79** | 100% | 100% / 0% | 1.5e-08 |
| natural LogiQA (strict) | 27 | +26.72 | -0.07 | **+26.79** | 100% | 100% / 0% | 1.5e-08 |
| natural MATH-500 | 46 | +26.95 | -0.85 | **+27.80** | 100% | 100% / 0% | 2.8e-14 |
| natural MATH-500 (strict) | 46 | +26.95 | -0.85 | **+27.80** | 100% | 100% / 0% | 2.8e-14 |

## 2. Which components carry the gap?

MLP share = MLP part of the gap (vs attention). Stop-signal MLPs: [28, 31, 32, 33, 34, 35]. rho = Spearman correlation of the 36 per-MLP gap contributions with the controlled catch-vs-aligned pairs.

| group | MLP share | top-6 MLPs by gap | share of gap in MLPs [28, 31, 32, 33, 34, 35] | rho vs reference |
|---|---|---|---|---|
| controlled: catch loop vs aligned finish | 59% | [35, 34, 33, 32, 28, 31] | 47% | +1.00 |
| controlled: catch loop vs aligned finish (strict) | 59% | [35, 34, 33, 32, 28, 31] | 47% | +1.00 |
| controlled: catch loop vs catch finish | 59% | [35, 34, 33, 32, 28, 27] | 47% | +0.94 |
| controlled: catch loop vs catch finish (strict) | 59% | [35, 34, 33, 32, 28, 27] | 47% | +0.94 |
| natural AQuA | 60% | [35, 34, 33, 32, 28, 31] | 47% | +0.92 |
| natural AQuA (strict) | 60% | [35, 34, 33, 32, 28, 31] | 47% | +0.92 |
| natural LogiQA | 60% | [35, 34, 33, 28, 32, 27] | 48% | +0.88 |
| natural LogiQA (strict) | 60% | [35, 34, 33, 28, 32, 27] | 48% | +0.88 |
| natural MATH-500 | 59% | [35, 34, 33, 32, 28, 31] | 46% | +0.94 |
| natural MATH-500 (strict) | 59% | [35, 34, 33, 32, 28, 31] | 46% | +0.94 |

## 3. Does restoring the late MLPs restore the stop logit?

% of the finish-loop gap recovered by replacing MLPs [28, 31, 32, 33, 34, 35] at loop@commit.

| group | own finish values | sampled-controlled donor | control: MLP [18] |
|---|---|---|---|
| controlled: catch loop vs aligned finish | 49% | 50% | -0% |
| controlled: catch loop vs aligned finish (strict) | 49% | 50% | -0% |
| controlled: catch loop vs catch finish | 50% | 49% | 0% |
| controlled: catch loop vs catch finish (strict) | 50% | 49% | 0% |
| natural AQuA | 52% | 52% | 0% |
| natural AQuA (strict) | 52% | 52% | 0% |
| natural LogiQA | 54% | 56% | 0% |
| natural LogiQA (strict) | 54% | 56% | 0% |
| natural MATH-500 | 51% | 53% | 0% |
| natural MATH-500 (strict) | 51% | 53% | 0% |
