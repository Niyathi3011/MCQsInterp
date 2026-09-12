# RQ3 — is the withheld stop-signal *causal*?

## From RQ2

- **E2b:** at `ka` (token before `</think>` in the terminated trace) the residual's
  projection onto the `</think>` unembedding is far lower for `catch` than for a
  matched `aligned` trace — every one of 94 pairs, Wilcoxon p = 3.8e-17.
- **E2c:** that ~+11.5 logit gap is written almost entirely by late-layer MLPs,
  dominated by **MLP 27** (Δ ≈ +5.6, ~48%), then 26/25/24/22/20. Attention ~15%,
  diffuse. On `catch` these MLPs go quiet (contribution → ~0), they do **not**
  flip sign — the stop-signal is *withheld*, not overridden.
- **E2a:** no clean linear "no correct option" direction at any non-confounded
  read-out (null). RQ3 therefore does **not** use `d_noopt`.

## Question

Does forcing / removing that late-MLP write to `</think>` *change the outcome*,
or is it a spectator that correlates with a decision made elsewhere?

## Interventions (position `ka`, and during generation from `ka`)

| name | trace | edit at `ka` on `blocks.L.hook_mlp_out`, L ∈ components | predict |
| --- | --- | --- | --- |
| RESCUE replace | catch | set to the aligned trace's own `mlp_out[L]@ka` | logit(</think>) → aligned |
| RESCUE add-Δ | catch | add `(aligned − catch)` `mlp_out[L]@ka` | logit(</think>) → aligned |
| RESCUE proj | catch | match aligned **only along** `û = W_U[:,think]/‖·‖`, keep the rest | isolates the `</think>`-specific claim |
| LESION zero | aligned | set to 0 | logit(</think>) → catch |
| LESION proj | aligned | remove the `û` component only | logit(</think>) → catch (surgical) |
| CONTROL | catch | RESCUE replace but on a late MLP **not** in the E2c set | ≈ no change |

`components` default = `27`; also run `27,26,25,24,22,20` (the E2c top set) to see
how much of the gap is causally recoverable and whether MLP 27 alone suffices.

## E3a — static, measure the logit (fast, deterministic)

Per problem: forward `aligned[:ka+1]` and `catch[:ka+1]`, record
`logit(</think>)@ka` for each; cache `mlp_out[L]@ka` for both; run each patched
condition once via `run_with_hooks`. Report, over problems:

- mean `logit(</think>)` aligned / catch / gap
- each RESCUE condition and **% of gap recovered** = `(patched − catch) / gap`
- each LESION condition and **% of gap removed** = `(aligned − patched) / gap`
- CONTROL (expect ~0 %)
- Wilcoxon: RESCUE vs catch, LESION vs aligned (per-problem, n ≈ 94)

**Positive result:** RESCUE replace/add recover ≳ 80 % of the gap; LESION zero
removes ≳ 60 %; CONTROL < ~15 %. RESCUE proj (û-only) still recovers most of it →
the effect really is the `</think>`-direction write, not a side effect.

## E3b — generative, measure behaviour (`--generate`, slow)

Donor = mean over aligned problems of `mlp_out[L]@ka` (`--n`-subset first).
Greedy-decode from the prefix, clamping `mlp_out[L]` at the **last position every
step** to the donor (RESCUE) or 0 (LESION). Up to `--gen-tokens` (start 150–200).

Per condition, over problems: **% that emit `</think>`**, median offset at which
they do, median longest run of identical consecutive sentences (loop fingerprint,
as in `analyze_loop.py`).

| trace | condition | predict |
| --- | --- | --- |
| catch | unpatched | rarely stops, high repeat |
| catch | RESCUE clamp MLP 27 (±26,25,24,22,20) | stops early, low repeat |
| catch | CONTROL clamp MLP 15 | ≈ unpatched |
| aligned | unpatched | stops almost immediately |
| aligned | LESION zero MLP 27 | delayed / no stop, repeat rises |

**Positive result:** RESCUE lifts catch `</think>`-emission from near-0 to a clear
majority while CONTROL does not; LESION drops aligned emission and/or pushes it
into repetition.

## What RQ3 can conclude

- **Can:** the late-MLP `</think>` write is (or isn't) *sufficient* to flip the
  stop-logit on catch and *necessary* for it on aligned; whether that logit flip
  is enough to end the loop under free generation.
- **Cannot:** *why* those MLPs read the options-vs-answer mismatch, or where that
  mismatch is represented (E2a was null — open).

## Files

| file | stage | GPU |
| --- | --- | --- |
| `rq3_patch.py` | E3a static patch (`--experiment e3a`, default) | yes |
| `rq3_patch.py --generate` | E3b clamped generation (`--experiment e3b`) | yes |
