#!/bin/bash
# R1-Distill-Qwen-7B on the four SOPHIA datasets, greedy, 12k-token cap.
# Needs: vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 \
#        --max-model-len 16384 --reasoning-parser deepseek_r1
source /workspace/env.sh
cd /workspace/MCQsInterp
OUT=natural_looping/results/r1-distill-qwen-7b/greedy_max12k
for d in gsm8k aqua math500 logiqa; do
  python -u natural_looping/generate.py --dataset $d \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max-tokens 12288 --workers 32 \
    --out $OUT/$d.jsonl 2>&1 | grep --line-buffered -v Warning | tee $OUT/logs/$d.log
done
echo ALL DONE | tee -a $OUT/logs/logiqa.log
