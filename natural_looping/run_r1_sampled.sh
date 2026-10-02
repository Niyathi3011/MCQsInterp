#!/bin/bash
# R1-Distill-Qwen-7B on AQuA, MATH-500 and LogiQA, sampled (T=0.6, top-p 0.95,
# DeepSeek's recommended settings), 4 seeds, 12k-token cap.  Gives sampled loop
# rates and matched loop/finish pairs (same prompt, different run).
# Needs: vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 \
#        --max-model-len 16384 --reasoning-parser deepseek_r1
source /workspace/env.sh
cd /workspace/MCQsInterp
BASE=natural_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k
for d in aqua math500 logiqa; do
  for s in 0 1 2 3; do
    OUT=$BASE/seed$s
    mkdir -p $OUT/logs
    python -u natural_looping/generate.py --dataset $d \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max-tokens 12288 --workers 32 \
      --temperature 0.6 --top-p 0.95 --seed $s \
      --out $OUT/$d.jsonl 2>&1 | grep --line-buffered -v Warning | tee $OUT/logs/$d.log
  done
done
echo ALL DONE | tee $BASE/ALL_DONE
