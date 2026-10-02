# Related work and positioning (searched 2026-10-01)

## Closest work

- **Góral, Wiśnios, Sankowski, Budzianowski. "Wait, that's not an option: LLMs
  Robustness with Incorrect Multiple-Choice Options." ACL 2025.** arXiv:2409.00113.
  MCQs with no valid answer (arithmetic, domain knowledge, medical). Aligned models
  default to an invalid option; base models refuse more, improving with scale.
  "Reflective judgment" framing. Not about reasoning models, loops or mechanisms.
  -> Our catch setup and the Qwen2.5-Instruct wrong-pick result are the same
  behaviour. We add: reasoning models LOOP instead, the stop mechanism, and that
  forcing a stop turns the loop back into the wrong pick.

- **Duan et al. "Circular Reasoning: Understanding Self-Reinforcing Loops in Large
  Reasoning Models." arXiv:2601.05693, Jan 2026.** LoopBench (numerical and
  statement loops); reasoning impasses trigger loops, maintained by a "V-shaped"
  attention mechanism; semantic before textual repetition; CUSUM early-warning
  detection. Uses DeepSeek-R1-Distill-Qwen-14B.
  -> Closest mechanistic work. Theirs: attention, loop maintenance, detection. Ours:
  a controlled trigger, the stop decision localised to a generic late-MLP gate shared
  by controlled and natural loops, and causal evidence that opening it does not
  resolve the impasse.

- **Zhang et al. "When Reasoning Goes Astray: Attention Dynamics of Uncontrolled
  Reasoning." arXiv:2609.38817, 30 Sep 2026.** RADAR detects loops from attention;
  "Attention Realignment" reduces looping.
  -> Measures loop reduction, not whether the answer/conflict is resolved.

- **Yu et al. "Can We Break LLMs Out of Self-Loops?" (SOPHIA). arXiv:2607.18100,
  Jul 2026.** Cluster-level self-loops, contrastive steering; hit rate = next step
  leaves the cluster. AQuA cluster "failing to match a multiple-choice option".
  -> We reproduce their datasets; their "self-loop" is phase dwell (94-95% of
  transitions), not non-termination; their metric does not test resolution.

## Behavioural relatives

- **Fan et al. "Missing Premise exacerbates Overthinking." COLM 2025.**
  arXiv:2504.06514. Ill-posed questions: reasoning models produce 2-4x more tokens
  and fail to flag the missing premise; non-reasoning LLMs do better.
  -> Same family of trigger (no valid resolution). We study MCQ format, loops to the
  cap, the mechanism, and stopping.

- **Overthinking / budget forcing.** E.g. "When More Thinking Hurts" (arXiv:2604.10739):
  extra reasoning abandons correct answers; budget forcing gives limited benefit;
  "Early Stopping for LRMs via Confidence Dynamics" (arXiv:2604.04930).
  -> Consistent with our results: stopping saves tokens without hurting accuracy,
  and helps exactly when the committed answer was already right.

- **"Reasoning Models Know When They're Right: Probing Hidden States for
  Self-Verification." arXiv:2504.05419 (2025).** A probe on hidden states predicts
  intermediate-answer correctness; confidence-based early exit cuts 24% of tokens.
  -> Our selective-stopping check finds that surface repetition does NOT predict
  correctness (31-40% at every repetition count); a hidden-state probe might.

## Our claim, positioned

1. A controlled trigger (no valid option) causes non-termination in a reasoning model
   (75% vs ~5%, robust to sampling), and the same conflict underlies most natural MCQ
   loops (AQuA).
2. The stop decision is written by a generic late-MLP gate (same layers, controlled
   and natural; equally closed at healthy doubts): loops never reach the state that
   opens it.
3. Opening the gate (or forcing </think> / boosting its logit) ends the thinking but
   not the doubt: the deliberation continues after </think>, and accuracy is recovered
   only when the committed answer was already correct.
