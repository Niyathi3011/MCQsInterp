#!/bin/bash
# Steps 2-3 (ordinary doubts vs loop doubts; stop signal over repeated commits).
# Waits for step 1 to finish (it needs the GPU), then runs.
source /workspace/env.sh
cd /workspace/MCQsInterp
S1=looping_mechanism/results/r1-distill-qwen-7b/stop_intervention/STEP1_DONE
until [ -f $S1 ]; do sleep 120; done
OUT=looping_mechanism/results/r1-distill-qwen-7b/doubt_timecourse
python -u looping_mechanism/doubt_timecourse.py 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $OUT/run.log
echo STEP23_DONE | tee $OUT/STEP23_DONE
