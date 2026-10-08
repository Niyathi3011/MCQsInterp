#!/bin/bash
# Round 2, unattended, in order:
#   1. R1: MATH + GSM8K fix (key-free commit points / live rule) -> replay + live rerun, both temperatures
#   2. R1: allow-None generation (T=0.6, seeds 0-1)
#   3. Qwen3: missing generation (GSM8K natural, GSM8K catch/aligned, LogiQA minus_gold, allow-None)
#   4. Qwen3: per-benchmark stop switch, T=0.6 (pairs -> live runs -> GPU stages)
#   5. Qwen3: resume the old stop-intervention run (lowest priority)
source /workspace/env.sh
cd /workspace/MCQsInterp
R1=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
Q3=Qwen/Qwen3-4B-Thinking-2507
LOG=looping_mechanism/results/round2.log
say() { echo "$(date -u '+%d %H:%M') $*" | tee -a $LOG; }

serve() {  # switch the vLLM server to model $1
  if curl -s localhost:8000/v1/models | grep -q "$1"; then return; fi
  tmux send-keys -t serve C-c
  for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
  tmux send-keys -t serve "vllm serve $1 --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1" Enter
  until curl -s localhost:8000/v1/models | grep -q "$1"; do sleep 10; done
}
gen() {  # gen <model> <generate.py source flags> <out> <seed>
  mkdir -p $(dirname $3)/logs
  python -u natural_looping/generate.py $2 --model $1 --max-tokens 12288 --workers 40 \
    --temperature 0.6 --top-p 0.95 --seed $4 --out $3 2>&1 | grep --line-buffered -v Warning \
    | tee $(dirname $3)/logs/$(basename $3 .jsonl).log | grep --line-buffered "^done\|ERROR"
}

say "waiting for the R1 T=0 chain"
until grep -q "ALL BENCHMARKS DONE (T=0)" looping_mechanism/results/r1-distill-qwen-7b/bench_switch/run_all_t0.log 2>/dev/null; do sleep 60; done
sleep 30; tmux kill-session -t catchsets 2>/dev/null

say "1. R1 MATH + GSM8K fix"
MODEL=$R1 bash looping_mechanism/bench_stages.sh "math:0.6 math:0 gsm8k:0.6 gsm8k:0" "replay trigger live summary" | tee -a $LOG

say "2. R1 allow-None"
serve $R1
for s in 0 1; do for n in math_catch aqua_minus_gold gsm8k_catch logiqa_minus_gold math_aligned aqua_original gsm8k_aligned logiqa_original; do
  gen $R1 "--data controlled_looping/data/allow_none/$n.jsonl" controlled_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/allow_none/seed$s/$n.jsonl $s
done; done
touch controlled_looping/results/r1-distill-qwen-7b/sampled_t0.6_max12k/allow_none/DONE

say "3. Qwen3 generation"
serve $Q3
QN=natural_looping/results/qwen3-4b-thinking-2507/sampled_t0.6_max12k
QC=controlled_looping/results/qwen3-4b-thinking-2507/sampled_t0.6_max12k
for s in 0 1 2 3; do gen $Q3 "--dataset gsm8k" $QN/seed$s/gsm8k.jsonl $s; done
for s in 0 1 2 3; do
  gen $Q3 "--data controlled_looping/data/gsm8k_ctrl/aligned.jsonl" $QC/gsm8k_ctrl/seed$s/aligned.jsonl $s
  gen $Q3 "--data controlled_looping/data/gsm8k_ctrl/catch.jsonl" $QC/gsm8k_ctrl/seed$s/catch.jsonl $s
  gen $Q3 "--data controlled_looping/data/logiqa_forms/minus_gold.jsonl" $QC/logiqa_forms/seed$s/minus_gold.jsonl $s
done
for s in 0 1; do for n in math_catch aqua_minus_gold gsm8k_catch logiqa_minus_gold math_aligned aqua_original gsm8k_aligned logiqa_original; do
  gen $Q3 "--data controlled_looping/data/allow_none/$n.jsonl" $QC/allow_none/seed$s/$n.jsonl $s
done; done
touch $QC/QWEN3_ROUND2_GEN_DONE

say "4. Qwen3 per-benchmark stop switch (T=0.6)"
QB=looping_mechanism/results/qwen3-4b-thinking-2507/bench_switch
for B in aqua logiqa math gsm8k; do
  MAX=0; [ "$B" = "logiqa" ] && MAX=60
  python looping_mechanism/bench_switch.py --model $Q3 --bench $B --temp 0.6 --stage pairs --max-heldout $MAX 2>&1 \
    | grep -v Warning | tee -a $LOG
  for s in 10 11; do gen $Q3 "--data $QB/${B}_t0.6/live_items.jsonl" $QB/${B}_t0.6/live_runs/seed$s.jsonl $s; done
  python -c "import json; p='$QB/${B}_t0.6/config.json'; c=json.load(open(p)); c['live_seeds']=['10','11']; json.dump(c,open(p,'w'),indent=1)"
done
MODEL=$Q3 bash looping_mechanism/bench_stages.sh "aqua:0.6 logiqa:0.6 math:0.6 gsm8k:0.6" "trigger signal replay live summary" | tee -a $LOG

say "5. Qwen3 old stop-intervention resume"
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
R=looping_mechanism/results/qwen3-4b-thinking-2507
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u looping_mechanism/stop_intervention.py --stage intervene --model $Q3 \
  --pairs looping_mechanism/data/pairs_qwen3-4b-thinking-2507.jsonl --components $R/components.json --out $R/stop_intervention \
  2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee -a $R/stop_intervention/intervene.log
python looping_mechanism/stop_intervention.py --stage summary --model $Q3 --pairs looping_mechanism/data/pairs_qwen3-4b-thinking-2507.jsonl \
  --out $R/stop_intervention 2>&1 | grep -v "Warning\|Timeout is disabled\|logic for timeout" > $R/stop_intervention/summary.log
say "ROUND 2 DONE"
