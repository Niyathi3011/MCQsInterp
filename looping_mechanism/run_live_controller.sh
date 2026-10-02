#!/bin/bash
# Live stop controller (probe = sensor, stop-MLPs = switch), then resume the extra
# intervention seeds (fix 2, resumable).
source /workspace/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MCQsInterp
OUT=looping_mechanism/results/r1-distill-qwen-7b/live_controller
S1=looping_mechanism/results/r1-distill-qwen-7b/stop_intervention
mkdir -p $OUT
python -u looping_mechanism/live_controller.py --stage scan 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $OUT/scan.log
tmux send-keys -t serve "source /workspace/env.sh && vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.92" Enter
until curl -s localhost:8000/v1/models | grep -q R1-Distill; do sleep 10; done
python -u looping_mechanism/live_controller.py --stage force 2>&1 | grep --line-buffered -v Warning | tee $OUT/force.log
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
python -u looping_mechanism/live_controller.py --stage switch 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $OUT/switch.log
python looping_mechanism/live_controller.py --stage summary 2>&1 | grep -v "Warning\|Timeout is disabled\|logic for timeout" | tee $OUT/summary.log
echo CONTROLLER_DONE | tee $OUT/CONTROLLER_DONE
for s in 1 2; do
  python -u looping_mechanism/stop_intervention.py --stage intervene --seed $s --natural-only \
    2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee -a $S1/intervene_seed$s.log
done
python looping_mechanism/stop_intervention.py --stage summary 2>&1 | grep -v "Warning\|Timeout is disabled\|logic for timeout" | tee $S1/summary.log
echo FIX2_DONE | tee $S1/FIX2_DONE
