"""
Two-stage dataset builder for the shortcut experiment.

Per source problem we emit:

  1 x stage1_open   - the raw problem, no options. Model must reason and produce
                      a number. This CoT is the reference "did the real work" trace.

  1 x stage2_mcq    - the SAME problem as a 2-option MCQ. Variants:

      aligned   (default)       (A) = gold number       (B) = unrelated sentence   gold = A
      swap      (--swap-frac)    (A) = unrelated sentence  (B) = gold number        gold = B
      catch     (--catch-frac)   (A) = WRONG number      (B) = unrelated sentence   gold = None
                                 (neither option is correct -> picking (A) is a
                                  pure "it's a math problem, answer is a number"
                                  shortcut)

Datasets (--dataset):
  gsm8k    openai/gsm8k main/test           (default; grade-school, near-ceiling for 7B)
  gsm_hard reasoning-machines/gsm_hard      (GSM8K structure, hard numbers; needs download)
  math500  HuggingFaceH4/MATH-500 test      (competition math, levels 1-5; --min-level)

Only problems whose gold answer is numeric are kept (the "option A = the number"
design needs that). Row ids are namespaced by dataset, so several datasets can
share one results file if you also split by --out.
"""
import argparse
import glob
import json
import os
import random
import re
from pathlib import Path

from datasets import load_dataset

from distractor_sentences import SENTENCES

DATASETS = {
    "gsm8k":    dict(hub="openai/gsm8k",   cfg="main", split="test",
                     legacy="gsm8k",       qk="question", ak="answer",
                     glob="*gsm8k*"),
    "gsm_hard": dict(hub="reasoning-machines/gsm_hard", cfg=None, split="train",
                     legacy=None,          qk="input",    ak="target",
                     glob="*gsm[-_]hard*"),
    "math500":  dict(hub="HuggingFaceH4/MATH-500", cfg=None, split="test",
                     legacy=None,          qk="problem",  ak="answer",
                     glob="*MATH-500*"),
}

_ROOTS = [
    os.path.expanduser(os.environ.get("HF_HOME") or "~/.cache/huggingface"),
    "/workspace/hf_cache",
    os.path.expanduser("~/.cache/huggingface"),
]


def _read_table(path):
    import pandas as pd
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    import pyarrow as pa  # .arrow
    with pa.memory_map(path, "r") as src:
        try:
            return pa.ipc.open_stream(src).read_all().to_pandas()
        except pa.lib.ArrowInvalid:
            return pa.ipc.open_file(src).read_all().to_pandas()


def load_rows(name, split):
    """List of raw dataset dicts. Tries the hub, then a cached parquet/arrow file
    (so it works offline and around the legacy-cache-id bug). Force a file with
    DATASET_FILE=/abs/path."""
    spec = DATASETS[name]
    forced = os.environ.get("DATASET_FILE")
    if forced and os.path.exists(forced):
        return _read_table(forced).to_dict("records")

    for repo, cfg in [(spec["hub"], spec["cfg"])] + \
                     ([(spec["legacy"], spec["cfg"])] if spec["legacy"] else []):
        try:
            return list(load_dataset(repo, cfg, split=split))
        except Exception:  # noqa: BLE001
            pass

    pats = []
    for r in _ROOTS:
        pats += glob.glob(f"{r}/**/datasets--*/{spec['glob']}/**/*.parquet", recursive=True)
        pats += glob.glob(f"{r}/**/{spec['glob']}/**/*{split}*.parquet", recursive=True)
        pats += glob.glob(f"{r}/**/{spec['glob']}/**/*.parquet", recursive=True)
        pats += glob.glob(f"{r}/**/{spec['glob']}/**/*{split}*.arrow", recursive=True)
    for p in pats:
        try:
            return _read_table(p).to_dict("records")
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(
        f"Could not load '{name}' ({split}) from the hub or cache. "
        f"Download it elsewhere and set DATASET_FILE=/abs/path/to/file.parquet. "
        f"Searched roots: {_ROOTS}")


def gold_of(name, row):
    """Extract the final answer string; '' if not cleanly numeric."""
    if name == "gsm8k":
        m = re.search(r"####\s*(.+)", row["answer"])
        raw = m.group(1) if m else ""
    elif name == "gsm_hard":
        raw = str(row["target"])
    else:  # math500 - answer already isolated
        raw = str(row["answer"])
    raw = raw.strip().replace(",", "").replace("$", "").replace("\\!", "").strip()
    raw = re.sub(r"^\\text\{(.*)\}$", r"\1", raw)
    try:
        f = float(raw)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return ""


def wrong_number(gold, rng):
    """A plausible wrong number, formatted like `gold`, for catch trials."""
    g = float(gold)
    is_int = g.is_integer()
    g = int(g) if is_int else g
    cands = set()
    if is_int:
        step = max(1, abs(g) // 10)
        for d in (1, 2, 3, step, 2 * step):
            cands.add(g + d)
            cands.add(g - d)
        cands.add(g * 10)
        if abs(g) >= 10:
            s = list(str(abs(g)))
            s[0], s[1] = s[1], s[0]
            cands.add(int("".join(s)) * (-1 if g < 0 else 1))
        cands = {c for c in cands if c > 0 and c != g}
        return str(int(rng.choice(sorted(cands)))) if cands else str(g + 1)
    for d in (0.1, 0.5, 1, 2, 10):
        cands.add(round(g + d, 2))
        cands.add(round(g - d, 2))
    cands = {c for c in cands if c != g}
    c = rng.choice(sorted(cands)) if cands else g + 1
    return f"{c:.2f}".rstrip("0").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="gsm8k", choices=list(DATASETS))
    ap.add_argument("--split", default=None, help="override the dataset's default split")
    ap.add_argument("--n", type=int, default=300, help="number of source problems")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-level", type=int, default=0,
                    help="math500 only: keep problems with level >= this (1-5)")
    ap.add_argument("--swap-frac", type=float, default=0.25)
    ap.add_argument("--catch-frac", type=float, default=0.25)
    ap.add_argument("--out", default="data/dataset.jsonl")
    args = ap.parse_args()

    name = args.dataset
    spec = DATASETS[name]
    split = args.split or spec["split"]
    rng = random.Random(args.seed)

    raw = load_rows(name, split)
    items = []
    for i, row in enumerate(raw):
        if args.min_level and int(row.get("level", 0) or 0) < args.min_level:
            continue
        g = gold_of(name, row)
        if not g:
            continue
        items.append((i, str(row[spec["qk"]]).strip(), g))
    rng.shuffle(items)
    items = items[: args.n]
    print(f"{name}/{split}: {len(raw)} rows -> {len(items)} usable (numeric gold"
          + (f", level>={args.min_level}" if args.min_level else "") + ")")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n1 = n2 = 0
    with out.open("w") as f:
        for src_i, q, gold in items:
            sid = f"{name}_{src_i}"
            sent = rng.choice(SENTENCES)

            f.write(json.dumps({
                "id": f"{sid}__stage1", "kind": "stage1_open", "dataset": name,
                "source_idx": src_i, "question": q, "gold_value": gold,
            }) + "\n")
            n1 += 1

            r = rng.random()
            if r < args.catch_frac:
                variant, opt_a, opt_b, gl = ("catch", wrong_number(gold, rng), sent, None)
            elif r < args.catch_frac + args.swap_frac:
                variant, opt_a, opt_b, gl = ("swap", sent, gold, "B")
            else:
                variant, opt_a, opt_b, gl = ("aligned", gold, sent, "A")

            f.write(json.dumps({
                "id": f"{sid}__stage2", "kind": "stage2_mcq", "dataset": name,
                "source_idx": src_i, "question": q,
                "option_A": str(opt_a), "option_B": str(opt_b),
                "gold_letter": gl, "gold_value": gold, "variant": variant,
            }) + "\n")
            n2 += 1

    print(f"wrote {n1} stage1 + {n2} stage2 rows -> {out}")


if __name__ == "__main__":
    main()
