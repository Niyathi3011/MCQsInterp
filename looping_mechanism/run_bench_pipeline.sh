#!/bin/bash
# One benchmark end to end (after its live runs are generated): pause the vLLM
# generation queue, run the TransformerLens stages, then restart vLLM and the queue.
#   bash looping_mechanism/run_bench_pipeline.sh aqua 0.6
source /workspace/env.sh
cd /workspace/MCQsInterp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B=$1; T=$2
D=looping_mechanism/results/r1-distill-qwen-7b/bench_switch/${B}_t$T
mkdir -p $D/logs
S="python -u looping_mechanism/bench_switch.py --bench $B --temp $T --stage"
filt() { grep --line-buffered -v -e "Loading weights" -e Warning -e WARNING -e "Timeout is disabled" -e "logic for timeout"; }

until [ -f $D/live_runs/DONE ]; do sleep 30; done
echo "$(date -u +%H:%M) pausing the generation queue and vLLM"
tmux kill-session -t catchsets 2>/dev/null
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done

ok=1
for st in trigger signal replay live summary; do
  echo "$(date -u +%H:%M) stage $st"
  $S $st 2>&1 | filt | tee $D/logs/$st.log
  [ ${PIPESTATUS[0]} -ne 0 ] && { echo "stage $st FAILED" | tee $D/FAILED; ok=0; break; }
done
[ $ok = 1 ] && touch $D/PIPELINE_DONE

echo "$(date -u +%H:%M) restarting vLLM and the generation queue"
tmux send-keys -t serve "vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1" Enter
tmux new -d -s catchsets -c /workspace/MCQsInterp "bash controlled_looping/run_r1_catch_sets.sh 2>&1 | tee -a controlled_looping/results/r1-distill-qwen-7b/catch_sets.log"
