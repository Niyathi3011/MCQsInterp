# Live stop controller: probe as sensor, stop-MLPs as switch

Each looping run is replayed; the controller stops at the first 'states an answer, then
doubts it' point where the held-out probe gives P(correct) > 0.7. Answers are generated
for real after the stop (T=0.6) and scored. Loops that never trigger keep looping (no
answer). Baselines: stop at the first commit point / at the model's closest-to-stop
point (force </think>, same settings).

| group | loops | controller stops | correct (of stopped): switch / force | correct over ALL loops: controller (switch) / first commit / model point / never stop |
|---|---|---|---|---|
| aqua | 80 | 40% | 19% / 28% | 8% / 14% / 19% / 0% |
| logiqa | 190 | 49% | 36% / 35% | 18% / 28% / 29% / 0% |
| math500 | 104 | 40% | 24% / 27% | 10% / 14% / 23% / 0% |
| own150_open | 25 | 56% | 21% / 21% | 12% / 24% / 18% / 0% |
| own150_aligned | 32 | 53% | 41% / 56% | 22% / 28% / 64% / 0% |
| own150_swap | 29 | 45% | 46% / 50% | 21% / 38% / 48% / 0% |

## Threshold sweep (stated answer at the stopping point; no generation)

| threshold | loops stopped | stated answer correct (of stopped) |
|---|---|---|
| 0.3 | 62% | 33% |
| 0.5 | 55% | 36% |
| 0.7 | 46% | 40% |
| 0.8 | 41% | 43% |
| 0.9 | 29% | 44% |
