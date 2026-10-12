# Sampled results: `natural_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k`

4 seeds, T=0.6, top-p 0.95. Loop = hit the token cap without finishing (no answer). Mean over seeds, [min-max].

| dataset | format | n | loop rate | greedy loop | accuracy | greedy acc | loops in 0 seeds | in all seeds | **matched pairs** |
|---|---|---|---|---|---|---|---|---|---|
| aqua | MCQ (5 options) | 254 | 7.9% [7.5-8.7] | 25.6% | 86.2% [84.6-87.0] | 71.7% | 219 | 5 | **30** |
| logiqa | MCQ (4 options) | 500 | 9.5% [8.2-10.4] | 39.4% | 50.0% [49.0-51.6] | 35.6% | 380 | 4 | **116** |
| math500 | open-ended | 500 | 5.2% [4.0-6.0] | 10.0% | 91.4% [90.6-92.2] | 87.4% | 443 | 6 | **51** |

Matched pairs = questions that looped in at least one seed and finished in at least one other (identical prompt). Written to `matched_pairs.jsonl`.
