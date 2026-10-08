#!/bin/bash
# Option B step 1: the held-out questions of one benchmark generated FROM SCRATCH with
# new seeds (default 10 11 at T=0.6; one greedy run at T=0). Needs the vLLM server.
#   bash looping_mechanism/run_bench_live_gen.sh aqua 0.6 ["10 11 12 13"]
source /workspace/env.sh
cd /workspace/MCQsInterp
B=$1; T=$2
D=looping_mechanism/results/r1-distill-qwen-7b/bench_switch/${B}_t$T
mkdir -p $D/live_runs/logs
if [ "$T" = "0" ]; then SEEDS="greedy"; else SEEDS=${3:-"10 11"}; fi
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
for s in $SEEDS; do
  if [ "$s" = "greedy" ]; then X="--temperature 0"; else X="--temperature 0.6 --top-p 0.95 --seed $s"; fi
  python -u natural_looping/generate.py --data $D/live_items.jsonl --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --max-tokens 12288 --workers 32 $X --out $D/live_runs/seed$s.jsonl 2>&1 | grep --line-buffered -v Warning \
    | tee $D/live_runs/logs/seed$s.log
done
python -c "import json,sys; p='$D/config.json'; c=json.load(open(p)); c['live_seeds']='$SEEDS'.split(); json.dump(c,open(p,'w'),indent=1)"
touch $D/live_runs/DONE
