#!/bin/bash
# Step 1: restoring the stop signal at the looping run's commit point.
# (a) no-intervention continuations via vLLM (must be serving in tmux `serve`)
# (b) stop vLLM, then the TransformerLens interventions, then the summary.
source /workspace/env.sh
cd /workspace/MCQsInterp
OUT=looping_mechanism/results/r1-distill-qwen-7b/stop_intervention
python -u looping_mechanism/stop_intervention.py --stage continue 2>&1 | grep --line-buffered -v Warning | tee $OUT/continue.log
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
python -u looping_mechanism/stop_intervention.py --stage intervene 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $OUT/intervene.log
python looping_mechanism/stop_intervention.py --stage summary 2>&1 | grep -v Warning | tee $OUT/summary.log
echo STEP1_DONE | tee $OUT/STEP1_DONE
