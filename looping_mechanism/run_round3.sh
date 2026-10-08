#!/bin/bash
# Round 3 (after the Qwen3 step-3 resume):
#   1. does the stop gate reopen once the conflict can be resolved? (Qwen3, then R1; TransformerLens)
#   2. finish R1 allow-None: seed 0 answerable sets (the false-None check), then seed 1 (all 8 sets)
source /workspace/env.sh
cd /workspace/MCQsInterp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R1=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
Q3=Qwen/Qwen3-4B-Thinking-2507
LOG=looping_mechanism/results/round3.log
say() { echo "$(date -u '+%d %H:%M') $*" | tee -a $LOG; }
filt() { grep --line-buffered -v -e "Loading weights" -e Warning -e WARNING -e "Timeout is disabled" -e "logic for timeout"; }
gpu_free() {
  tmux send-keys -t serve C-c
  for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
}
serve() {
  tmux send-keys -t serve "vllm serve $1 --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1" Enter
  until curl -s localhost:8000/v1/models | grep -q "$1"; do sleep 10; done
}

say "waiting for the Qwen3 step-3 resume"
while tmux has-session -t q3resume 2>/dev/null; do sleep 30; done

say "1. gate reopen"
gpu_free
for M in $Q3 $R1; do
  T=$(LOOP_MODEL=$M python -c "from looping_mechanism.model_cfg import TAG; print(TAG)" 2>/dev/null || \
      python -c "import sys; sys.path.insert(0,'looping_mechanism'); import os; os.environ['LOOP_MODEL']='$M'; import model_cfg; print(model_cfg.TAG)")
  mkdir -p looping_mechanism/results/$T/gate_reopen
  LOOP_MODEL=$M python -u looping_mechanism/gate_reopen.py 2>&1 | filt | tee looping_mechanism/results/$T/gate_reopen/run.log
done

say "2. R1 allow-None completion"
serve $R1
A=controlled_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/allow_none
gen() {
  mkdir -p $A/seed$2/logs
  python -u natural_looping/generate.py --data controlled_looping/data/allow_none/$1.jsonl --model $R1 \
    --max-tokens 12288 --workers 40 --temperature 0.6 --top-p 0.95 --seed $2 --out $A/seed$2/$1.jsonl 2>&1 \
    | grep --line-buffered -v Warning | tee -a $A/seed$2/logs/$1.log | grep --line-buffered "^done\|ERROR" | tee -a $LOG
}
for n in math_aligned gsm8k_aligned aqua_original logiqa_original; do gen $n 0; done
for n in math_catch aqua_minus_gold gsm8k_catch logiqa_minus_gold math_aligned aqua_original gsm8k_aligned logiqa_original; do gen $n 1; done
for T in r1-distill-qwen-7b qwen3-4b-thinking-2507; do
  python looping_mechanism/allow_none_summary.py --model-tag $T 2>&1 | filt > /dev/null
done
say "ROUND 3 DONE"
