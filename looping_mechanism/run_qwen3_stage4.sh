#!/bin/bash
# Qwen3: the R1 analyses that "ask the model" -- model-chosen stopping points, the
# answer-correctness probe (5 folds split BY QUESTION), and the probe-guided stopping
# rule (each loop scored by a probe trained without its question). Same order as
# run_model_stop_points.sh + run_live_controller.sh for R1.
source /workspace/env.sh
cd /workspace/MCQsInterp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOOP_MODEL=Qwen/Qwen3-4B-Thinking-2507
R=looping_mechanism/results/qwen3-4b-thinking-2507
MSP=$R/model_stop_points; PR=$R/answer_probe; LC=$R/live_controller
mkdir -p $MSP $PR $LC
filt() { grep --line-buffered -v -e "Loading weights" -e Warning -e WARNING -e "Timeout is disabled" -e "logic for timeout"; }
gpu_free() {
  tmux send-keys -t serve C-c
  for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
}
serve() {
  tmux send-keys -t serve "vllm serve $LOOP_MODEL --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1" Enter
  until curl -s localhost:8000/v1/models | grep -q "$LOOP_MODEL"; do sleep 10; done
}
run() { echo "$(date -u +%H:%M) $1 $2"; python -u looping_mechanism/$1.py --stage $2 2>&1 | filt | tee $3/$2.log; }

gpu_free
run model_stop_points scan $MSP
serve;    run model_stop_points stop $MSP
gpu_free; run model_stop_points mlp $MSP
run model_stop_points summary $MSP
run answer_probe extract $PR
run answer_probe train $PR
run live_controller fit $LC
run live_controller scan $LC
serve;    run live_controller force $LC
gpu_free; run live_controller switch $LC
run live_controller summary $LC
echo "$(date -u +%H:%M) QWEN3 STAGE 4 DONE"
