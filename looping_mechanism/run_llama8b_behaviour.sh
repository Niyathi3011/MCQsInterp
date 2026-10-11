#!/bin/bash
# Third model, behaviour only: DeepSeek-R1-Distill-Llama-8B (Llama-3.1 base, same distillation
# as R1). Same items, prompts and settings as R1/Qwen3 (T=0.6, top-p 0.95, 12k budget, bf16).
# Order = priority (seed 0 first; seed 1 if time allows):
#   1. trigger: MATH-150 aligned/catch, AQuA-227 original/removed
#   2. allow-None on those
#   3. trigger + allow-None: GSM8K-200, LogiQA-200
#   4. AQuA-227 open (why-loop analysis)
#   5. seed 1 of steps 1-3
source /workspace/env.sh
cd /workspace/MCQsInterp
M=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
TAG=r1-distill-llama-8b
C=controlled_looping/results/$TAG/sampled_t0.6_max12k
D=controlled_looping/data
LOG=controlled_looping/results/$TAG/run.log
mkdir -p controlled_looping/results/$TAG
say() { echo "$(date -u '+%d %H:%M') $*" | tee -a $LOG; }
gen() {  # gen <data file> <out subpath> <seed>
  local out=$C/$2; mkdir -p $(dirname $out)/logs
  python -u natural_looping/generate.py --data $1 --model $M --max-tokens 12288 --workers 40 \
    --temperature 0.6 --top-p 0.95 --seed $3 --out $out 2>&1 | grep --line-buffered -v Warning \
    | tee $(dirname $out)/logs/$(basename $out .jsonl).log | grep --line-buffered "^done\|ERROR" | tee -a $LOG
}
if ! curl -s localhost:8000/v1/models | grep -q "$M"; then
  tmux send-keys -t serve C-c
  for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
  tmux send-keys -t serve "vllm serve $M --port 8000 --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.95 --reasoning-parser deepseek_r1" Enter
  until curl -s localhost:8000/v1/models | grep -q "$M"; do sleep 10; done
fi
step() {  # step <seed> <set group>
  local s=$1
  case $2 in
    trig1) gen $D/boxed/aligned.jsonl prompt_boxed/seed$s/aligned.jsonl $s
           gen $D/boxed/catch.jsonl prompt_boxed/seed$s/catch.jsonl $s
           gen $D/subsets/aqua_original_227.jsonl aqua_forms/seed$s/original.jsonl $s
           gen $D/aqua_forms/minus_gold.jsonl aqua_forms/seed$s/minus_gold.jsonl $s ;;
    none1) for n in math_catch aqua_minus_gold math_aligned aqua_original; do
             gen $D/allow_none/$n.jsonl allow_none/seed$s/$n.jsonl $s; done ;;
    trig2) gen $D/gsm8k_ctrl/aligned.jsonl gsm8k_ctrl/seed$s/aligned.jsonl $s
           gen $D/gsm8k_ctrl/catch.jsonl gsm8k_ctrl/seed$s/catch.jsonl $s
           gen $D/subsets/logiqa_original_200.jsonl logiqa_forms/seed$s/original.jsonl $s
           gen $D/logiqa_forms/minus_gold.jsonl logiqa_forms/seed$s/minus_gold.jsonl $s ;;
    none2) for n in gsm8k_catch logiqa_minus_gold gsm8k_aligned logiqa_original; do
             gen $D/allow_none/$n.jsonl allow_none/seed$s/$n.jsonl $s; done ;;
    open)  gen $D/aqua_forms/open.jsonl aqua_forms/seed$s/open.jsonl $s ;;
  esac
}
for g in trig1 none1 trig2 none2 open; do say "seed 0: $g"; step 0 $g; done
touch controlled_looping/results/$TAG/SEED0_DONE
for g in trig1 none1 trig2 none2; do say "seed 1: $g"; step 1 $g; done
say "LLAMA-8B DONE"
