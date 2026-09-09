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


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_SCALE = ["", " thousand", " million", " billion", " trillion"]


def _under_1000(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    return _ONES[n // 100] + " hundred" + (
        "" if n % 100 == 0 else " " + _under_1000(n % 100))


def num_to_words(s):
    """'333' -> 'three hundred thirty-three', '-2' -> 'negative two',
    '3.5' -> 'three point five'."""
    s = str(s).strip()
    neg = s.startswith("-")
    s = s.lstrip("-")
    if "." in s:
        intp, frac = s.split(".", 1)
        w = num_to_words(("-" if neg else "") + (intp or "0"))
        return w + " point " + " ".join(_ONES[int(d)] for d in frac)
    n = int(s or "0")
    if n == 0:
        return "zero"
    parts, grp = [], 0
    while n > 0:
        n, r = divmod(n, 1000)
        if r:
            parts.append(_under_1000(r) + _SCALE[grp])
        grp += 1
    w = " ".join(reversed(parts))
    return ("negative " + w) if neg else w


def wrong_number(gold, rng, close=False):
    """A plausible wrong number, formatted like `gold`, for catch trials.
    close=True -> only near-misses (gold +/- 1..3), which are hard to reject
    without actually recomputing (pushes anchoring / acceptance up)."""
    g = float(gold)
    is_int = g.is_integer()
    g = int(g) if is_int else g
    cands = set()
    if is_int:
        deltas = (1, 2, 3) if close else \
            (1, 2, 3, max(1, abs(g) // 10), 2 * max(1, abs(g) // 10))
        for d in deltas:
            cands.add(g + d)
            cands.add(g - d)
        if not close:
            cands.add(g * 10)
            if abs(g) >= 10:
                s = list(str(abs(g)))
                s[0], s[1] = s[1], s[0]
                cands.add(int("".join(s)) * (-1 if g < 0 else 1))
        cands = {c for c in cands if c > 0 and c != g}
        return str(int(rng.choice(sorted(cands)))) if cands else str(g + 1)
    for d in ((0.1, 0.5, 1) if close else (0.1, 0.5, 1, 2, 10)):
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
    ap.add_argument("--decoy-frac", type=float, default=0.0,
                    help="fraction of stage2 rows where (A)=wrong number, "
                         "(B)=the CORRECT answer spelled in words (gold=B). "
                         "'pick the digits' is now a detectable error.")
    ap.add_argument("--close-distractor", action="store_true",
                    help="catch/decoy wrong-numbers are near-misses (gold +/- 1..3) only")
    ap.add_argument("--bias-shots", type=int, default=0,
                    help="write <out>.bias.txt: K solved examples where the numeric "
                         "option is always the answer (few-shot format biasing)")
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
    total = len(items)
    held = items[: args.bias_shots]                       # reserved for the bias prefix
    items = items[args.bias_shots: args.bias_shots + args.n]  # disjoint eval set
    print(f"{name}/{split}: {len(raw)} rows -> {total} usable (numeric gold"
          + (f", level>={args.min_level}" if args.min_level else "")
          + f"); eval={len(items)}, bias-held={len(held)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.bias_shots and held:
        lines = ["Here are some solved examples.\n"]
        for j, (_, hq, hg) in enumerate(held, 1):
            hs = rng.choice(SENTENCES)
            num_in_a = rng.random() < 0.5
            a, b = (hg, hs) if num_in_a else (hs, hg)
            letter = "A" if num_in_a else "B"
            lines.append(f"Example {j}:\n{hq}\n(A) {a}\n(B) {b}\n"
                         f"Working through it, the result is {hg}. Answer: ({letter})\n")
        lines.append("Now answer the next question the same way.\n")
        bias_path = out.with_suffix(".bias.txt")
        bias_path.write_text("\n".join(lines))
        print(f"wrote {len(held)}-shot bias prefix -> {bias_path}")
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
            c, s, d = args.catch_frac, args.swap_frac, args.decoy_frac
            if r < c:
                variant, opt_a, opt_b, gl = (
                    "catch", wrong_number(gold, rng, args.close_distractor), sent, None)
            elif r < c + d:
                variant, opt_a, opt_b, gl = (
                    "decoy", wrong_number(gold, rng, args.close_distractor),
                    num_to_words(gold), "B")
            elif r < c + d + s:
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
