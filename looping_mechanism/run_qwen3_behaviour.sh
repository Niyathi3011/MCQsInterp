#!/bin/bash
# Second model: Qwen3-4B-Thinking-2507, behaviour replication at the same settings as
# R1 (T=0.6, top-p 0.95, 4 seeds, 12k-token budget, boxed prompt). vLLM must be
# serving Qwen/Qwen3-4B-Thinking-2507 on :8000 (--reasoning-parser deepseek_r1).
source /workspace/env.sh
cd /workspace/MCQsInterp
M=Qwen/Qwen3-4B-Thinking-2507
TAG=qwen3-4b-thinking-2507/sampled_t0.6_max12k
run () {   # data-or-dataset  out
  python -u natural_looping/generate.py $1 --model $M --max-tokens 12288 --workers 48 \
    --temperature 0.6 --top-p 0.95 --seed $s --out $2 2>&1 | grep --line-buffered -v Warning
}
for s in 0 1 2 3; do
  C=controlled_looping/results/$TAG/prompt_boxed/seed$s; mkdir -p $C/logs
  for c in catch aligned open swap; do run "--data controlled_looping/data/boxed/$c.jsonl" $C/$c.jsonl | tee $C/logs/$c.log; done
  A=controlled_looping/results/$TAG/aqua_forms/seed$s; mkdir -p $A/logs
  for f in minus_gold open; do run "--data controlled_looping/data/aqua_forms/$f.jsonl" $A/$f.jsonl | tee $A/logs/$f.log; done
  N=natural_looping/results/$TAG/seed$s; mkdir -p $N/logs
  for d in aqua math500 logiqa; do run "--dataset $d" $N/$d.jsonl | tee $N/logs/$d.log; done
done
echo QWEN3_BEHAVIOUR_DONE | tee natural_looping/results/qwen3-4b-thinking-2507/QWEN3_BEHAVIOUR_DONE
