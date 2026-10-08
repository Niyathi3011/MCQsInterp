#!/bin/bash
# Run chosen bench_switch.py stages for several benchmark/temperature folders in ONE GPU
# session: stop vLLM, run the stages, restart vLLM with the same model.
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B bash looping_mechanism/bench_stages.sh \
#       "math:0.6 math:0 gsm8k:0.6 gsm8k:0" "replay trigger live summary" [extra args]
source /workspace/env.sh
cd /workspace/MCQsInterp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
filt() { grep --line-buffered -v -e "Loading weights" -e Warning -e WARNING -e "Timeout is disabled" -e "logic for timeout"; }
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
for bt in $1; do
  B=${bt%%:*}; T=${bt##*:}
  TAG=$(python -c "import sys; sys.path.insert(0,'looping_mechanism'); import bench_switch as b; print(b.tag_of('$M'))")
  D=looping_mechanism/results/$TAG/bench_switch/${B}_t$T
  mkdir -p $D/logs
  for st in $2; do
    X=""; [ "$st" = "replay" ] && X="--lite"
    echo "$(date -u +%H:%M) $B T=$T stage $st"
    python -u looping_mechanism/bench_switch.py --model $M --bench $B --temp $T --stage $st $X $3 2>&1 | filt | tee $D/logs/$st.log
    [ ${PIPESTATUS[0]} -ne 0 ] && { echo "$B T=$T stage $st FAILED" | tee -a $D/FAILED; break; }
  done
done
echo "$(date -u +%H:%M) restarting vLLM ($M)"
tmux send-keys -t serve "vllm serve $M --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1" Enter
until curl -s localhost:8000/v1/models | grep -q "$M"; do sleep 10; done
