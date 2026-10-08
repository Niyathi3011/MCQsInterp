#!/bin/bash
# Qwen3-4B-Thinking-2507: every method run on R1, in order (T=0.6; Qwen3's model card
# advises against greedy decoding, so no T=0):
#   1. missing generation: GSM8K natural, GSM8K catch/aligned, LogiQA minus_gold, allow-None
#   2. summaries + matched pairs incl. GSM8K, why-loop analyses
#   3. per-benchmark stop switch: pairs -> live runs -> trigger/signal/replay/live/summary
#   4. model-chosen stopping points, answer probe, probe-guided stopping rule
#   5. resume the old stop-intervention run
source /workspace/env.sh
cd /workspace/MCQsInterp
Q3=Qwen/Qwen3-4B-Thinking-2507
T3=qwen3-4b-thinking-2507
LOG=looping_mechanism/results/$T3/round2.log
mkdir -p looping_mechanism/results/$T3
say() { echo "$(date -u '+%d %H:%M') $*" | tee -a $LOG; }
serve() {
  if curl -s localhost:8000/v1/models | grep -q "$1"; then return; fi
  tmux send-keys -t serve C-c
  for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
  tmux send-keys -t serve "vllm serve $1 --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1" Enter
  until curl -s localhost:8000/v1/models | grep -q "$1"; do sleep 10; done
}
gpu_free() {
  tmux send-keys -t serve C-c
  for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
}
gen() {  # gen <generate.py source flags> <out> <seed>
  mkdir -p $(dirname $2)/logs
  python -u natural_looping/generate.py $1 --model $Q3 --max-tokens 12288 --workers 40 \
    --temperature 0.6 --top-p 0.95 --seed $3 --out $2 2>&1 | grep --line-buffered -v Warning \
    | tee $(dirname $2)/logs/$(basename $2 .jsonl).log | grep --line-buffered "^done\|ERROR" | tee -a $LOG
}
filt() { grep --line-buffered -v -e "Loading weights" -e Warning -e WARNING -e "Timeout is disabled" -e "logic for timeout"; }

say "1. Qwen3 generation"
serve $Q3
QN=natural_looping/results/$T3/sampled_t0.6_max12k
QC=controlled_looping/results/$T3/sampled_t0.6_max12k
for s in 0 1 2 3; do gen "--dataset gsm8k" $QN/seed$s/gsm8k.jsonl $s; done
for s in 0 1 2 3; do
  gen "--data controlled_looping/data/gsm8k_ctrl/aligned.jsonl" $QC/gsm8k_ctrl/seed$s/aligned.jsonl $s
  gen "--data controlled_looping/data/gsm8k_ctrl/catch.jsonl" $QC/gsm8k_ctrl/seed$s/catch.jsonl $s
  gen "--data controlled_looping/data/logiqa_forms/minus_gold.jsonl" $QC/logiqa_forms/seed$s/minus_gold.jsonl $s
done
for s in 0 1; do for n in math_catch aqua_minus_gold gsm8k_catch logiqa_minus_gold math_aligned aqua_original gsm8k_aligned logiqa_original; do
  gen "--data controlled_looping/data/allow_none/$n.jsonl" $QC/allow_none/seed$s/$n.jsonl $s
done; done

say "2. summaries, matched pairs, why-loop"
for s in 0 1 2 3; do python natural_looping/summarize.py $QN/seed$s > /dev/null 2>&1; done
python natural_looping/matched_pairs.py $QN 2>&1 | filt | tee -a $LOG
python looping_mechanism/why_loop.py --model-tag $T3 2>&1 | filt > /dev/null

say "3. per-benchmark stop switch"
QB=looping_mechanism/results/$T3/bench_switch
for B in aqua logiqa math gsm8k; do
  MAX=0; [ "$B" = "logiqa" ] && MAX=60
  python looping_mechanism/bench_switch.py --model $Q3 --bench $B --temp 0.6 --stage pairs --max-heldout $MAX 2>&1 | filt | tee -a $LOG
  for s in 10 11; do gen "--data $QB/${B}_t0.6/live_items.jsonl" $QB/${B}_t0.6/live_runs/seed$s.jsonl $s; done
  python -c "import json; p='$QB/${B}_t0.6/config.json'; c=json.load(open(p)); c['live_seeds']=['10','11']; json.dump(c,open(p,'w'),indent=1)"
done
MODEL=$Q3 bash looping_mechanism/bench_stages.sh "aqua:0.6 logiqa:0.6 math:0.6 gsm8k:0.6" "trigger signal replay live summary" | tee -a $LOG

say "4. model-chosen stopping points, answer probe, probe-guided rule"
[ -f looping_mechanism/run_qwen3_stage4.sh ] && bash looping_mechanism/run_qwen3_stage4.sh 2>&1 | tee -a $LOG

say "5. old stop-intervention resume"
gpu_free
R=looping_mechanism/results/$T3
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u looping_mechanism/stop_intervention.py --stage intervene --model $Q3 \
  --pairs looping_mechanism/data/pairs_$T3.jsonl --components $R/components.json --out $R/stop_intervention \
  2>&1 | filt | tee -a $R/stop_intervention/intervene.log
python looping_mechanism/stop_intervention.py --stage summary --model $Q3 --pairs looping_mechanism/data/pairs_$T3.jsonl \
  --out $R/stop_intervention 2>&1 | filt > $R/stop_intervention/summary.log
serve $Q3
say "QWEN3 ROUND 2 DONE"
