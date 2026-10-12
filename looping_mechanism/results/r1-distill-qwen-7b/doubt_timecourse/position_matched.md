# Position-matched control: loop doubts vs healthy doubts at the same place in the trace

Doubt points = 'states an answer, then Wait'. Position = characters into the reasoning.
Cells: mean logit(</think>) (n). Healthy = in runs that later stop; loop = in runs that never stop.

| position in trace (chars) | healthy doubts | loop doubts | loop - healthy |
|---|---|---|---|
| 0-2k | +4.38 (n=141) | +5.22 (n=69) | +0.84 |
| 2k-5k | +4.74 (n=321) | +4.84 (n=230) | +0.10 |
| 5k-10k | +4.81 (n=400) | +4.96 (n=311) | +0.15 |
| 10k-20k | +5.66 (n=458) | +6.37 (n=665) | +0.72 |
| 20k+ | +6.78 (n=167) | +7.11 (n=673) | +0.34 |

Regression over 3435 doubt points (logit ~ loop + log position + group):
- loop vs healthy at the same position: **+0.35 logits** (95% CI +0.08 to +0.62, bootstrap over pairs)
- log(position): +0.61 per unit
- stop-MLP contribution, loop vs healthy at the same position: +0.18

For scale: a real stop is about +26 above a doubt.
