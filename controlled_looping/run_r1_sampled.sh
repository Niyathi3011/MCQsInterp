#!/bin/bash
# Controlled looping conditions (open / aligned / swap / catch) on the 150 MATH-500
# problems of the original greedy catch experiment, re-run SAMPLED
# (T=0.6, top-p 0.95, seeds 0-3, 12k-token cap) under BOTH prompts:
#   prompt_boxed       same instruction as natural_looping (main comparison)
#   prompt_answerline  same instruction as the original greedy run (temperature effect only)
# Order: the main comparison (boxed catch + aligned) finishes first.
# Needs: vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 \
#        --max-model-len 16384 --reasoning-parser deepseek_r1
source /workspace/env.sh
cd /workspace/MCQsInterp
BASE=controlled_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k
for style in boxed answerline; do
  for cond in catch aligned open swap; do
    for s in 0 1 2 3; do
      OUT=$BASE/prompt_$style/seed$s
      mkdir -p $OUT/logs
      python -u natural_looping/generate.py --data controlled_looping/data/$style/$cond.jsonl \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max-tokens 12288 --workers 32 \
        --temperature 0.6 --top-p 0.95 --seed $s \
        --out $OUT/$cond.jsonl 2>&1 | grep --line-buffered -v Warning | tee $OUT/logs/$cond.log
    done
  done
done
echo ALL DONE | tee $BASE/ALL_DONE
