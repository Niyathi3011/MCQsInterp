#!/bin/bash
# Second model (Qwen3-4B-Thinking-2507): mechanism and stopping tests, queued after the
# behaviour run. Stop layers are found from Qwen3's own controlled pairs (--components auto).
source /workspace/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MCQsInterp
M=Qwen/Qwen3-4B-Thinking-2507
Q=qwen3-4b-thinking-2507
R=looping_mechanism/results/$Q
mkdir -p $R/doubt_timecourse $R/stop_intervention
until [ -f natural_looping/results/$Q/QWEN3_BEHAVIOUR_DONE ]; do sleep 120; done
tmux send-keys -t serve C-c
for i in $(seq 1 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] && break; sleep 5; done
F="grep --line-buffered -v Loading\ weights\|Warning\|WARNING"

# behaviour summaries + matched pairs
for s in 0 1 2 3; do python natural_looping/summarize.py natural_looping/results/$Q/sampled_t0.6_max12k/seed$s > /dev/null; done
python natural_looping/matched_pairs.py natural_looping/results/$Q/sampled_t0.6_max12k | tee $R/matched_pairs.log
python looping_mechanism/build_pairs.py --model-tag $Q | tee $R/build_pairs.log

# weight processing must preserve Qwen3's computation, else stop here
python looping_mechanism/check_processing.py --model $M 2>&1 | grep -v "Loading weights\|Warning\|WARNING" | tee $R/check_processing.log
grep -q "^OK" $R/check_processing.log || { echo "processing check failed, stopping" | tee $R/STOPPED; exit 1; }

# steps 5-6: which MLPs write the stop signal (found from Qwen3's controlled pairs), patching
python -u looping_mechanism/stop_signal.py --model $M --pairs looping_mechanism/data/pairs_$Q.jsonl \
  --components auto --control-layers auto --out-dir $R 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $R/stop_signal.log
# step 7: healthy vs loop doubts, repeated commits
python -u looping_mechanism/doubt_timecourse.py --model $M --pairs looping_mechanism/data/pairs_$Q.jsonl \
  --components $R/components.json --out $R/doubt_timecourse \
  --groups ctrl_cross,ctrl_same,nat_aqua,nat_logiqa,nat_math500 2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $R/doubt_timecourse/run.log
# step 8: stop interventions at the commit point (natural pairs + 60 controlled)
python -u looping_mechanism/stop_intervention.py --stage intervene --model $M \
  --pairs looping_mechanism/data/pairs_$Q.jsonl --components $R/components.json --out $R/stop_intervention \
  2>&1 | grep --line-buffered -v "Loading weights\|Warning\|WARNING" | tee $R/stop_intervention/intervene.log
python looping_mechanism/stop_intervention.py --stage summary --model $M --pairs looping_mechanism/data/pairs_$Q.jsonl \
  --out $R/stop_intervention 2>&1 | grep -v "Warning\|Timeout is disabled\|logic for timeout" | tee $R/stop_intervention/summary.log
echo QWEN3_MECHANISM_DONE | tee $R/QWEN3_MECHANISM_DONE
