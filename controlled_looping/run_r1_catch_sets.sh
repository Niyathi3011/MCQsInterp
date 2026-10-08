#!/bin/bash
# Per-benchmark catch-vs-aligned runs (R1-Distill-Qwen-7B, 12k-token cap), so each
# benchmark's stop-signal values are learned from its own controlled pairs and tested
# on its own natural loops (held out). Both temperatures; T=0.6 (main) first.
#   1. GSM8K natural, T=0.6 seeds 0-3        (defines GSM8K's held-out loops at T=0.6)
#   2. T=0.6 seeds 0-3: GSM8K aligned + catch, LogiQA minus_gold
#   3. T=0 (greedy): GSM8K aligned + catch, LogiQA minus_gold, AQuA minus_gold,
#                    MATH-150 boxed aligned + catch
# Aligned forms already run: AQuA / LogiQA original options (natural_looping), MATH-150
# boxed aligned at T=0.6 (prompt_boxed).
# Needs: vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8000 --dtype bfloat16 \
#        --max-model-len 16384 --reasoning-parser deepseek_r1
source /workspace/env.sh
cd /workspace/MCQsInterp
M=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
NAT=natural_looping/results/r1-distill-qwen-7b
CTL=controlled_looping/results/r1-distill-qwen-7b
D=controlled_looping/data

run() {  # run <data-or-dataset-flag> <out> <temperature> [seed]
  local src=$1 out=$2 t=$3 s=$4
  mkdir -p $(dirname $out)/logs
  local extra="--temperature $t"
  [ "$t" != "0" ] && extra="$extra --top-p 0.95 --seed $s"
  python -u natural_looping/generate.py $src --model $M --max-tokens 12288 --workers 32 $extra \
    --out $out 2>&1 | grep --line-buffered -v Warning | tee $(dirname $out)/logs/$(basename $out .jsonl).log
}

until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done

for s in 0 1 2 3; do
  run "--dataset gsm8k" $NAT/sampled_t0.6_max12k/seed$s/gsm8k.jsonl 0.6 $s
done
for s in 0 1 2 3; do
  run "--data $D/gsm8k_ctrl/aligned.jsonl" $CTL/sampled_t0.6_max12k/gsm8k_ctrl/seed$s/aligned.jsonl 0.6 $s
  run "--data $D/gsm8k_ctrl/catch.jsonl" $CTL/sampled_t0.6_max12k/gsm8k_ctrl/seed$s/catch.jsonl 0.6 $s
  run "--data $D/logiqa_forms/minus_gold.jsonl" $CTL/sampled_t0.6_max12k/logiqa_forms/seed$s/minus_gold.jsonl 0.6 $s
done
touch $CTL/sampled_t0.6_max12k/CATCH_SETS_DONE

G=$CTL/greedy_max12k
run "--data $D/gsm8k_ctrl/aligned.jsonl" $G/gsm8k_ctrl/aligned.jsonl 0
run "--data $D/gsm8k_ctrl/catch.jsonl" $G/gsm8k_ctrl/catch.jsonl 0
run "--data $D/logiqa_forms/minus_gold.jsonl" $G/logiqa_forms/minus_gold.jsonl 0
run "--data $D/aqua_forms/minus_gold.jsonl" $G/aqua_forms/minus_gold.jsonl 0
run "--data $D/boxed/aligned.jsonl" $G/prompt_boxed/aligned.jsonl 0
run "--data $D/boxed/catch.jsonl" $G/prompt_boxed/catch.jsonl 0
echo ALL DONE | tee $G/CATCH_SETS_DONE
