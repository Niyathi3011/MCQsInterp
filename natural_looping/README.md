# natural_looping

Does a reasoning model loop (never finish its reasoning) on ordinary benchmark
questions, where a correct answer always exists? This folder runs models on the
four datasets used by SOPHIA (Yu et al. 2026, "Can We Break LLMs Out of
Self-Loops?", arXiv 2607.18100), each in its **natural format**: open-ended stays
open-ended, multiple-choice keeps its own options. There are no constructed
options and no unrelated-sentence distractors (contrast with the `catch`
condition in the rest of the repo).

## Datasets (`data/`)

Sizes follow the paper's Table 6. The paper does not name splits, subsets or
prompts, so those are our choices.

| file | source | format | rows |
|---|---|---|---|
| `gsm8k_test_500.jsonl` | `openai/gsm8k` main/test (1,319) | open-ended, numeric | 500 random (seed 0) |
| `aqua_test_254.jsonl` | `deepmind/aqua_rat` raw/test | MCQ, options A-E | all 254 |
| `logiqa_test_500.jsonl` | `lucasmccabe/logiqa` test (651), parquet export | MCQ, options A-D | 500 random (seed 0) |
| `math500_test_500.jsonl` | `HuggingFaceH4/MATH-500` test | open-ended, LaTeX | all 500 |

Prompt: DeepSeek's recommended R1 prompt, "Please reason step by step, and put
your final answer within \boxed{}." For the MCQ datasets it asks for the option
letter in the box.

## Results (`results/<model>/<decoding>/`)

One folder per model and decoding setting, e.g.
`results/r1-distill-qwen-7b/greedy_max12k/`:

- `<dataset>.jsonl`: one row per problem with the full reasoning trace, the
  answer text, `truncated` (hit the token cap = a **loop**), the parsed answer and
  `correct`
- `logs/`: generation logs
- `summary.md` / `summary.json`: accuracy, loop rate with 95% CI, correct vs
  incorrect trace length (the paper's Fig. 2 view), and how repetitive loops are

## Scoring

- GSM8K: numeric match of the last `\boxed{}`
- MATH-500: `math-verify` equivalence of the last `\boxed{}` against the gold LaTeX
- AQuA / LogiQA: the option letter in the last `\boxed{}` (fallback: "the answer is X")
- A trace that hits the token cap has no answer and counts as incorrect

## Reproduce

```bash
python natural_looping/build_datasets.py                      # -> data/
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 \
    --max-model-len 16384 --reasoning-parser deepseek_r1        # separate tmux
bash natural_looping/run_r1_greedy.sh                          # -> results/r1-distill-qwen-7b/greedy_max12k/
python natural_looping/summarize.py natural_looping/results/r1-distill-qwen-7b/greedy_max12k
```

`generate.py` is resumable, and keeps token-capped rows (the truncation is the
result, not a failure to retry).
