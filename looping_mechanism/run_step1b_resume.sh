#!/bin/bash
# Resume step 1b (interventions) after the out-of-memory crash; then the summary.
source /workspace/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MCQsInterp
OUT=looping_mechanism/results/r1-distill-qwen-7b/stop_intervention
python -u looping_mechanism/stop_intervention.py --stage intervene 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee -a $OUT/intervene.log
python looping_mechanism/stop_intervention.py --stage summary 2>&1 | grep -v Warning | tee $OUT/summary.log
echo STEP1B_DONE | tee $OUT/STEP1B_DONE
