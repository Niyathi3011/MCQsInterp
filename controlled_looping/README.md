# controlled_looping

The controlled looping experiment, where the valid answer is removed on purpose,
rerun under the same **sampled** decoding as `natural_looping/`, so that
controlled and natural loops can be compared within one decoding regime.

## Conditions (`data/<prompt>/<condition>.jsonl`)

The same 150 MATH-500 problems (level >= 3, numeric answer) and the exact items
of the original greedy run (`data/force_r1.jsonl`):

| condition | options | valid answer? |
|---|---|---|
| `open` | none | yes (model writes it) |
| `aligned` | (A) correct number, (B) unrelated sentence | yes, A |
| `swap` | (A) unrelated sentence, (B) correct number | yes, B (position control) |
| `catch` | (A) wrong number, (B) unrelated sentence | **no** |

Each condition is run under two prompts:

- `boxed`: "Please reason step by step, and put your final answer within
  \boxed{}.", identical to `natural_looping`. This is the main comparison.
- `answerline`: "End with a line formatted exactly as: Answer: (X) where X is A
  or B.", byte-identical to the original greedy run (`results/raw_r1.jsonl`).
  Compared with that run, only the temperature changes.

## Results (`results/r1-distill-qwen-7b/sampled_t0.6_max12k/prompt_<prompt>/seed<s>/`)

T=0.6, top-p 0.95, seeds 0-3, 12k-token cap. One `<condition>.jsonl` per
condition (same row format as `natural_looping`; `catch` rows have
`correct = null` and `pred` = the letter picked), plus `logs/`.

## Reproduce

```bash
python controlled_looping/build_datasets.py
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 \
    --max-model-len 16384 --reasoning-parser deepseek_r1        # separate tmux
bash controlled_looping/run_r1_sampled.sh
```

## Analysis plan

1. Loop rate per condition and prompt (mean and range over seeds) vs the greedy run
2. Behaviour on `catch` when it does finish: picks A (the wrong number), picks B,
   or rejects both
3. Matched pairs, filtered: problem solved open-ended, and the finishing run's
   `</think>` was the model's top-1 choice
   - catch-loop vs aligned-finish (same problem, same seed)
   - catch-loop vs catch-finish (same prompt, different seed)
4. Comparison point in the looping run: where it commits to its answer and starts
   doubting it, not the same token index
5. Stop-signal mechanism (logit lens, attribution, MLP patching) with the same
   code, positions and filters as the natural pairs:
   - same MLPs 20-27 in controlled and natural loops?
   - does the greedy result replicate?
