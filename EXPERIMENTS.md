# EXPERIMENTS — what's left, in run order

Every block assumes you've done, on the pod:

```bash
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf_cache
cd /workspace/MCQsInterp && git pull
```

Results file convention: `results/raw_<dataset>_<modeltag>.jsonl`
(`modeltag` = `qwen7b`, `qwen3b`, `qwen14b`, `r1-7b`).
`analyze.py --raw <file> --out_csv results/paired_<tag>.csv --plot results/shortcut_<tag>.png`

`vllm serve` runs in tmux `serve`; the experiment runs in tmux `run`.
`run_model.py` is resumable and skips finished `(model, id, mode)`.

---

## 1. Screen datasets (cheap — do first)

Serve the base model, then:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000 --dtype bfloat16 --max-model-len 4096   # tmux serve

python bench_accuracy.py --datasets gsm8k,math500 --n 200 \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
python bench_accuracy.py --datasets math500 --n 500 --min-level 3 \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
python bench_accuracy.py --datasets math500 --n 500 --min-level 4 \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct
```

Get the other datasets onto the pod (laptop has HF network):

```bash
# on laptop
python -c "from datasets import load_dataset as L; L('reasoning-machines/gsm_hard',split='train').to_parquet('gsm_hard.parquet')"
python -c "from datasets import load_dataset as L; L('KbsdJames/Omni-MATH',split='test').to_parquet('omnimath.parquet')"
python -c "from datasets import load_dataset as L; L('gneubig/aime-1983-2024',split='train').to_parquet('aime.parquet')"
rsync -avP gsm_hard.parquet omnimath.parquet aime.parquet root@<pod>:/workspace/data/
```

```bash
# on pod
python bench_accuracy.py --n 200 --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --datasets "gsm_hard=/workspace/data/gsm_hard.parquet,omni=/workspace/data/omnimath.parquet,aime=/workspace/data/aime.parquet"
```

**Keep** datasets with stage-1 accuracy **≈ 0.35–0.75** for the two-stage runs.
GSM8K (~0.9) and easy MATH stay only as the "above-band" reference.

---

## 2. Re-baseline on the neutral suffix

The `results/raw.jsonl` / `raw_math500.jsonl` you have used the old
"think step by step" prompt. Redo them so everything is comparable.

```bash
mv results/raw.jsonl          results/raw_gsm8k_qwen7b_stepwise.jsonl
mv results/raw_math500.jsonl  results/raw_math500_qwen7b_stepwise.jsonl

python build_dataset.py --dataset gsm8k  --n 1319 --out data/gsm8k.jsonl
python build_dataset.py --dataset math500 --n 400 --min-level 3 --out data/math500.jsonl

python run_model.py --data data/gsm8k.jsonl   --out results/raw_gsm8k_qwen7b.jsonl \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct --workers 8
python run_model.py --data data/math500.jsonl --out results/raw_math500_qwen7b.jsonl \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct --workers 8 --cot-tokens 2048

python rescore.py --raw results/raw_gsm8k_qwen7b.jsonl
python rescore.py --raw results/raw_math500_qwen7b.jsonl
python analyze.py --raw results/raw_gsm8k_qwen7b.jsonl   --out_csv results/paired_gsm8k_qwen7b.csv   --plot results/shortcut_gsm8k_qwen7b.png
python analyze.py --raw results/raw_math500_qwen7b.jsonl --out_csv results/paired_math500_qwen7b.csv --plot results/shortcut_math500_qwen7b.png
```

Quick GSM8K "gold stated anywhere" check (was only run for MATH):

```bash
python -c "
import json
r=[json.loads(l) for l in open('results/raw_gsm8k_qwen7b.jsonl')]
c=[x for x in r if x.get('variant')=='catch' and x.get('pred_letter')=='A']
h=sum(1 for x in c if str(x['gold_value']) in ((x.get('reasoning') or '')+x['completion']).split('Answer:')[0])
print(f'{h}/{len(c)} = {h/len(c):.0%} state the gold answer')
"
```

---

## 3. Two-stage on the mid-band datasets

For each dataset `D` that passed the screen (example: `gsm_hard`):

```bash
python build_dataset.py --dataset gsm_hard --n 800 --catch-frac 0.4 \
  --out data/gsm_hard.jsonl
# for a path dataset:  DATASET_FILE=/workspace/data/omnimath.parquet python build_dataset.py --dataset math500 --n 800 --catch-frac 0.4 --out data/omni.jsonl

python run_model.py --data data/gsm_hard.jsonl --out results/raw_gsmhard_qwen7b.jsonl \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct --workers 8 --cot-tokens 2048
python rescore.py  --raw results/raw_gsmhard_qwen7b.jsonl
python analyze.py  --raw results/raw_gsmhard_qwen7b.jsonl \
  --out_csv results/paired_gsmhard_qwen7b.csv --plot results/shortcut_gsmhard_qwen7b.png
```

`--catch-frac 0.4` because `aligned`/`swap` are already settled; put the samples
where the signal is.

---

## 4. No-CoT baseline

Does the shortcut get worse without reasoning? One dataset is enough.

```bash
python run_model.py --data data/gsm8k.jsonl --out results/raw_gsm8k_qwen7b.jsonl \
  --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
  --modes cot,direct --workers 8
python analyze.py --raw results/raw_gsm8k_qwen7b.jsonl   # cot rows only; compare pickA in the direct rows by hand:
python -c "
import json,collections
r=[json.loads(l) for l in open('results/raw_gsm8k_qwen7b.jsonl')]
for mode in ('cot','direct'):
    c=[x for x in r if x['kind']=='stage2_mcq' and x['mode']==mode and x.get('variant')=='catch']
    pa=sum(1 for x in c if x.get('pred_letter')=='A')
    print(f'{mode:7s} catch n={len(c)} pickA={pa/max(len(c),1):.3f}')
"
```

---

## 5. Scale down — Qwen2.5-3B (cached, cheap)

```bash
# tmux serve:
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8001 --dtype bfloat16 --max-model-len 4096

python run_model.py --data data/gsm8k.jsonl   --out results/raw_gsm8k_qwen3b.jsonl \
  --base-url http://localhost:8001/v1 --model Qwen/Qwen2.5-3B-Instruct --workers 8
python run_model.py --data data/math500.jsonl --out results/raw_math500_qwen3b.jsonl \
  --base-url http://localhost:8001/v1 --model Qwen/Qwen2.5-3B-Instruct --workers 8 --cot-tokens 2048
python rescore.py --raw results/raw_gsm8k_qwen3b.jsonl
python rescore.py --raw results/raw_math500_qwen3b.jsonl
python analyze.py --raw results/raw_gsm8k_qwen3b.jsonl   --out_csv results/paired_gsm8k_qwen3b.csv   --plot results/shortcut_gsm8k_qwen3b.png
python analyze.py --raw results/raw_math500_qwen3b.jsonl --out_csv results/paired_math500_qwen3b.csv --plot results/shortcut_math500_qwen3b.png
```

## 5b. Scale up — Qwen2.5-14B (re-download ~28 GB) — optional

```bash
hf download Qwen/Qwen2.5-14B-Instruct    # or rsync from laptop
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.95
# ...same run/rescore/analyze with modeltag qwen14b
```

---

## 6. Reasoning model — DeepSeek-R1-Distill-Qwen-7B

```bash
# laptop: huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir r1-7b ; rsync to /workspace/models/r1-7b
vllm serve /workspace/models/r1-7b --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --port 8000 --dtype bfloat16 --max-model-len 16384 --reasoning-parser deepseek_r1

python run_model.py --data data/math500.jsonl --out results/raw_math500_r1-7b.jsonl \
  --base-url http://localhost:8000/v1 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --workers 4 --cot-tokens 8192 \
  --stage1-suffix "End with a line formatted exactly as: Answer: <number>" \
  --stage2-suffix "End with a line formatted exactly as: Answer: (X)  where X is A or B."
python rescore.py --raw results/raw_math500_r1-7b.jsonl
python analyze.py --raw results/raw_math500_r1-7b.jsonl --out_csv results/paired_math500_r1-7b.csv --plot results/shortcut_math500_r1-7b.png
```

Smoke test with `--limit 20` first; check `parse_ok`. Hours for a full run —
start with a subset (`build_dataset.py --n 300`).

---

## 7. Collate

```bash
python -c "
import glob, pandas as pd
rows=[]
for f in glob.glob('results/paired_*.csv'):
    d=pd.read_csv(f); tag=f.split('paired_')[1][:-4]
    c=d[d.variant=='catch']; al=d[(d.variant=='aligned')&d.stage1_correct]; sw=d[(d.variant=='swap')&d.stage1_correct]
    rows.append(dict(tag=tag, n_catch=len(c),
        catch_pickA=(c.pred_letter=='A').mean(),
        unfaithful=((c.pred_letter=='A')&c.gold_in_cot).mean(),
        aligned_shortcut=(al.label=='shortcut').mean() if len(al) else float('nan'),
        swap_acc=sw.stage2_correct.mean() if len(sw) else float('nan')))
print(pd.DataFrame(rows).round(3).to_string(index=False))
"
```

---

## 8. Phase 2 — mechanistic (see `phase2/README.md`)

```bash
tmux kill-session -t serve        # free VRAM; TransformerLens loads weights itself
pip install --no-cache-dir transformer_lens transformers

python phase2/prep.py --raw results/raw_gsm8k_qwen7b.jsonl   --out phase2/pairs_gsm8k.jsonl
python phase2/prep.py --raw results/raw_math500_qwen7b.jsonl --out phase2/pairs_math500.jsonl

python phase2/ab_decision.py --pairs phase2/pairs_gsm8k.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct --category unfaithful --n 40
python phase2/ab_decision.py --pairs phase2/pairs_gsm8k.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct --category anchored --n 40
```

Then build the next scripts: activation patching (`clean` = matched aligned/swap,
`corrupted` = catch), attention knockout (option tokens vs CoT answer span),
"prefer-numeric-option" direction ablation, commitment-point probe.

---

## Priority if compute is limited

1, 2, 3 (one mid-band dataset), 8. That gives: which datasets matter, clean
matched baselines on the neutral prompt, the effect on a genuinely hard dataset,
and the start of the mechanistic account. 4/5/6 are the breadth axes.
