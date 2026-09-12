# RUNBOOK — Phase 1 from scratch

Target: RunPod pod, 1x RTX 4090 (24 GB), network volume mounted at `/workspace`.
Repo: `git@github.com:Niyathi3011/MCQsInterp.git`, branch `branch/experiment-1`.

`/workspace` persists across pod restarts; the container disk (`/`, `~`) does not.

---

## 1. Every time you connect to a (new) pod

```bash
# SSH key for git (NFS volume can't hold 0600 perms, so copy to local disk)
mkdir -p ~/.ssh && cp /workspace/.ssh/id_ed25519 ~/.ssh/ && chmod 600 ~/.ssh/id_ed25519
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes"
```

First pod ever (no key on the volume yet):
```bash
mkdir -p /workspace/.ssh
ssh-keygen -t ed25519 -f /workspace/.ssh/id_ed25519 -N "" -C runpod
cat /workspace/.ssh/id_ed25519.pub      # add at github.com -> Settings -> SSH and GPG keys
```

## 2. Repo

```bash
cd /workspace
git clone -b branch/experiment-1 git@github.com:Niyathi3011/MCQsInterp.git \
  || (cd MCQsInterp && git pull)
cd /workspace/MCQsInterp
```

## 3. Python env (once per volume)

```bash
export TMPDIR=/workspace/tmp && mkdir -p "$TMPDIR"
python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir vllm hf_transfer
```

## 4. Per-shell env — put this in `/workspace/env.sh`, source it in every shell

```bash
cat > /workspace/env.sh <<'EOF'
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/workspace/tmp
export GIT_SSH_COMMAND="ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes"
cd /workspace/MCQsInterp
EOF
source /workspace/env.sh
```

## 5. Model server — tmux session `serve`

```bash
tmux new -s serve
source /workspace/env.sh
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype bfloat16 --max-model-len 4096
#   detach: Ctrl-b then d
```

Wait until ready (from any shell):
```bash
curl -s http://localhost:8000/v1/models      # should list Qwen/Qwen2.5-7B-Instruct
```

## 6. Run the experiment — tmux session `run`

```bash
tmux new -s run
source /workspace/env.sh

python build_dataset.py --n 1319                     # data/dataset.jsonl (2638 rows)
python run_model.py --base-url http://localhost:8000/v1 \
                    --model Qwen/Qwen2.5-7B-Instruct --workers 16   # results/raw.jsonl
python analyze.py                                    # results/paired.*.csv + shortcut.*.png
#   detach: Ctrl-b then d
```

`run_model.py` is resumable — if it stops, rerun the exact same line.

## 7. Monitor (a separate shell)

```bash
watch -n 30 'wc -l /workspace/MCQsInterp/results/raw.jsonl'    # climbs to 2638
tmux attach -t run                                             # see the N/2638 counter
```

Done when the `run` shell prints `done -> results/raw.jsonl` and `analyze.py`
has printed the shortcut rate.

## 8. Stop

```bash
tmux attach -t serve      # Ctrl-C to stop vLLM
tmux kill-session -t serve
tmux kill-session -t run
# then stop the pod in the RunPod dashboard to halt GPU billing
```

Outputs on `/workspace/MCQsInterp/` survive: `data/dataset.jsonl`,
`results/raw.jsonl`, `results/paired.Qwen_Qwen2.5-7B-Instruct.csv`,
`results/shortcut.Qwen_Qwen2.5-7B-Instruct.png`.

---

## Troubleshooting

| symptom | fix |
| --- | --- |
| `HfUriError ... Repository id must be 'namespace/name', got 'gsm8k'` | `build_dataset.py` reads the cached file directly now; if it still fails: `find /workspace/hf_cache ~/.cache/huggingface -iname '*gsm8k*' \( -name '*.parquet' -o -name '*.arrow' \)` then `export GSM8K_FILE=/abs/path/to/test-*.parquet` |
| `pip ... Errno 122 Disk quota exceeded` | `pip install --no-cache-dir`, `export TMPDIR=/workspace/tmp`; check `du -sh /workspace/hf_cache/hub/* \| sort -h` and delete unused model dirs |
| `ssh ... bad permissions` on the key | the `cp ~/.ssh/ && chmod 600` in step 1 (NFS won't hold the perms) |
| `tmux: duplicate session: run` | it's still alive — `tmux attach -t run` instead of `tmux new` |
| `curl localhost:8000` fails | `tmux attach -t serve`, read the error; re-run the `vllm serve` line |
| run seems stuck | `wc -l results/raw.jsonl` a minute apart; if not moving, check `tmux attach -t serve` for OOM / errors |
