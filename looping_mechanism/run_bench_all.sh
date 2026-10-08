#!/bin/bash
# The per-benchmark stop-switch test for several benchmarks in a row, each end to end:
# pairs -> fresh live runs (vLLM) -> GPU stages (run_bench_pipeline.sh pauses vLLM and
# restarts it afterwards).
#   bash looping_mechanism/run_bench_all.sh 0.6 "gsm8k math logiqa"
# Speed settings: 2 new seeds per question; LogiQA tests 60 random held-out questions.
source /workspace/env.sh
cd /workspace/MCQsInterp
T=$1
for B in $2; do
  MAX=0; [ "$B" = "logiqa" ] && MAX=60
  echo "$(date -u +%H:%M) ===== $B T=$T"
  python looping_mechanism/bench_switch.py --bench $B --temp $T --stage pairs --max-heldout $MAX 2>&1 | grep -v Warning
  bash looping_mechanism/run_bench_live_gen.sh $B $T "10 11"
  bash looping_mechanism/run_bench_pipeline.sh $B $T
done
echo "$(date -u +%H:%M) ALL BENCHMARKS DONE (T=$T)"
