# Stopping points chosen by the model itself

For every looping run: the line break with the highest P(</think>) (model point) and
the text-detected commit point (text point). At each, </think> is forced and the
answer generated (T=0.6, 2 seeds, up to 1200 tokens).

## Where the points are

| group | loops | has text point | model point at (median % of loop) | max P(</think>) (median) | model point within 50 tokens of text point | stop-MLP push: model point / typical line break / real stop (catch pairs) |
|---|---|---|---|---|---|---|
| aqua | 80 | 95% | 80% | 0.000 | 1% | +9.6 / +5.5 / ~+24 |
| logiqa | 190 | 77% | 60% | 0.001 | 2% | +11.5 / +5.2 / ~+24 |
| math500 | 104 | 51% | 29% | 0.000 | 0% | +8.8 / +5.0 / ~+24 |
| own150_aligned | 32 | 72% | 73% | 0.000 | 4% | +8.9 / +5.0 / ~+24 |
| own150_open | 25 | 80% | 33% | 0.000 | 0% | +8.5 / +5.0 / ~+24 |
| own150_swap | 29 | 72% | 69% | 0.000 | 5% | +8.6 / +4.9 / ~+24 |

## Forcing </think> at each point

| group | point | runs | answer correct | answer hits the cap (keeps reasoning) |
|---|---|---|---|---|
| aqua | model | 160 | 19% | 34% |
| aqua | text | 152 | 15% | 64% |
| logiqa | model | 380 | 29% | 24% |
| logiqa | text | 292 | 36% | 33% |
| math500 | model | 208 | 23% | 64% |
| math500 | text | 106 | 28% | 72% |
| own150_aligned | model | 64 | 64% | 36% |
| own150_aligned | text | 46 | 39% | 67% |
| own150_open | model | 50 | 18% | 56% |
| own150_open | text | 40 | 30% | 62% |
| own150_swap | model | 58 | 48% | 40% |
| own150_swap | text | 42 | 52% | 60% |

## The catch-learned MLP switch at the model point

| condition | runs | stops | answer correct | answer hits the cap |
|---|---|---|---|---|
| force | 60 | 100% | 30% | 42% |
| mlp@1 | 60 | 100% | 33% | 35% |
| ctrl@1 | 60 | 17% | 8% | 0% |
