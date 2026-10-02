#!/bin/bash
# Paper fixes, queued after step 1b:
#  fix 1  AQuA in natural form without the correct option (+ open-ended control),
#         sampled T=0.6, 4 seeds (vLLM)
#  fix 2  two more sampling seeds for the step-1 interventions on the natural
#         pairs, for confidence intervals (TransformerLens)
source /workspace/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MCQsInterp
S1=looping_mechanism/results/r1-distill-qwen-7b/stop_intervention
until [ -f $S1/STEP1B_DONE ]; do sleep 120; done

# ---- fix 1: AQuA forms (vLLM) ----
tmux send-keys -t serve "source /workspace/env.sh && vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.92" Enter
until curl -s localhost:8000/v1/models | grep -q R1-Distill; do sleep 10; done
B=controlled_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/aqua_forms
for form in minus_gold open; do
  for s in 0 1 2 3; do
    mkdir -p $B/seed$s/logs
    python -u natural_looping/generate.py --data controlled_looping/data/aqua_forms/$form.jsonl \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --max-tokens 12288 --workers 32 \
      --temperature 0.6 --top-p 0.95 --seed $s --out $B/seed$s/$form.jsonl \
      2>&1 | grep --line-buffered -v Warning | tee $B/seed$s/logs/$form.log
  done
done
echo FIX1_DONE | tee $B/FIX1_DONE
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done

# ---- fix 2: extra intervention seeds (TransformerLens) ----
for s in 1 2; do
  python -u looping_mechanism/stop_intervention.py --stage intervene --seed $s --natural-only \
    2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $S1/intervene_seed$s.log
done
python looping_mechanism/stop_intervention.py --stage summary 2>&1 | grep -v "Warning\|Timeout is disabled\|logic for timeout" | tee $S1/summary.log
echo FIX2_DONE | tee $S1/FIX2_DONE
