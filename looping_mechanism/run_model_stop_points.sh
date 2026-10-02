#!/bin/bash
# "Ask the model itself" stopping points, slotted between fix 1 and fix 2:
#  1. wait for fix 1 (AQuA forms) to finish, then stop the `fixes` session before fix 2
#     gets going (fix 2 is resumable; at most one pair is redone)
#  2. scan all looping runs for the model's own stopping point (TransformerLens)
#  3. force </think> there and at the text commit point, score the answers (vLLM)
#  3b. at the model point of 60 loops, paste the catch-learned stop-MLP output
#      (vs force and an MLP-15 control), score the answers (TransformerLens)
#  4. answer-correctness detector (extract hidden states, train, stopping rule)
#  5. resume fix 2 (extra intervention seeds) and write its summary
source /workspace/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MCQsInterp
B=controlled_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/aqua_forms
S1=looping_mechanism/results/r1-distill-qwen-7b/stop_intervention
OUT=looping_mechanism/results/r1-distill-qwen-7b/model_stop_points
until [ -f $B/FIX1_DONE ]; do sleep 60; done
sleep 20; tmux kill-session -t fixes 2>/dev/null
pkill -f "stop_intervention.py --stage intervene" 2>/dev/null
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done

python -u looping_mechanism/model_stop_points.py --stage scan 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $OUT/scan.log

tmux send-keys -t serve "source /workspace/env.sh && vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.92" Enter
until curl -s localhost:8000/v1/models | grep -q R1-Distill; do sleep 10; done
python -u looping_mechanism/model_stop_points.py --stage stop 2>&1 | grep --line-buffered -v Warning | tee $OUT/stop.log
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
python -u looping_mechanism/model_stop_points.py --stage mlp 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $OUT/mlp.log
python looping_mechanism/model_stop_points.py --stage summary 2>&1 | grep -v "Warning\|Timeout is disabled\|logic for timeout" | tee $OUT/summary.log
echo MODEL_STOP_DONE | tee $OUT/MODEL_STOP_DONE

# answer-correctness detector: hidden states at every "states an answer, then doubts" point
PR=looping_mechanism/results/r1-distill-qwen-7b/answer_probe
mkdir -p $PR
python -u looping_mechanism/answer_probe.py --stage extract 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $PR/extract.log
python -u looping_mechanism/answer_probe.py --stage train 2>&1 | grep -v "Warning\|Timeout is disabled\|logic for timeout" | tee $PR/train.log
echo PROBE_DONE | tee $PR/PROBE_DONE

# resume fix 2
for s in 1 2; do
  python -u looping_mechanism/stop_intervention.py --stage intervene --seed $s --natural-only \
    2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee -a $S1/intervene_seed$s.log
done
python looping_mechanism/stop_intervention.py --stage summary 2>&1 | grep -v "Warning\|Timeout is disabled\|logic for timeout" | tee $S1/summary.log
echo FIX2_DONE | tee $S1/FIX2_DONE
