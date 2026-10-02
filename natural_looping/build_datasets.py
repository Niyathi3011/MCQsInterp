"""
Build the four datasets used by SOPHIA (Yu et al. 2026, arXiv 2607.18100) at the
sizes the paper reports (Table 6): GSM8K 500, AQuA 254, LogiQA 500, MATH 500.

The paper names the datasets and trace counts but not the splits, subsets or
prompts, so these are our choices (also recorded in each row):
  gsm8k    openai/gsm8k main/test (1,319)         -> random 500, seed 0
  aqua     deepmind/aqua_rat raw/test (254)       -> all 254, 5 options A-E
  logiqa   lucasmccabe/logiqa test (651)          -> random 500, seed 0, 4 options A-D
  math500  HuggingFaceH4/MATH-500 test (500)      -> all 500 (LaTeX answers)

Prompt = DeepSeek's recommended R1 math prompt ("Please reason step by step, and
put your final answer within \\boxed{}."), with the option letter boxed for the
two multiple-choice datasets.

    python natural_looping/build_datasets.py            # writes natural_looping/data/
"""
import argparse
import ast
import json
import os
import random
import re

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OPEN_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


def mcq_instr(letters):
    return ("Please reason step by step, and put the letter of the correct option ("
            + ", ".join(letters[:-1]) + f" or {letters[-1]}) within \\boxed{{}}.")


def gsm8k():
    d = load_dataset("openai/gsm8k", "main", split="test")
    for i, r in enumerate(d):
        gold = re.search(r"####\s*(.+)", r["answer"]).group(1).strip().replace(",", "")
        yield dict(idx=i, format="open", question=r["question"], gold=gold,
                   prompt=f"{r['question']}\n{OPEN_INSTR}")


def math500():
    d = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i, r in enumerate(d):
        yield dict(idx=i, format="open", question=r["problem"], gold=r["answer"],
                   level=r["level"], subject=r["subject"],
                   prompt=f"{r['problem']}\n{OPEN_INSTR}")


def aqua():
    d = load_dataset("deepmind/aqua_rat", "raw", split="test")
    letters = list("ABCDE")
    for i, r in enumerate(d):
        opts = r["options"] if isinstance(r["options"], list) else ast.literal_eval(r["options"])
        # stored as "A)5(√3 + 1)" -> "A) 5(√3 + 1)"
        texts = [re.sub(r"^\s*[A-E]\s*\)\s*", "", o) for o in opts]
        lines = "\n".join(f"{L}) {t}" for L, t in zip(letters, texts))
        yield dict(idx=i, format="mcq", question=r["question"], options=texts,
                   letters=letters[:len(texts)], gold=r["correct"].strip(),
                   prompt=f"{r['question']}\n{lines}\n{mcq_instr(letters[:len(texts)])}")


def logiqa():
    # the repo's loading script is no longer supported; use the hub's parquet export
    d = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet", split="test")
    letters = list("ABCD")
    for i, r in enumerate(d):
        opts = r["options"] if isinstance(r["options"], list) else ast.literal_eval(r["options"])
        lines = "\n".join(f"{L}. {t}" for L, t in zip(letters, opts))
        q = f"{r['context']}\n{r['query']}"
        yield dict(idx=i, format="mcq", question=q, options=list(opts),
                   letters=letters[:len(opts)], gold=letters[int(r["correct_option"])],
                   prompt=f"{q}\n{lines}\n{mcq_instr(letters[:len(opts)])}")


SPECS = {  # name: (loader, subset size or None = all, output file)
    "gsm8k":   (gsm8k,   500,  "gsm8k_test_500.jsonl"),
    "aqua":    (aqua,    None, "aqua_test_254.jsonl"),
    "logiqa":  (logiqa,  500,  "logiqa_test_500.jsonl"),
    "math500": (math500, None, "math500_test_500.jsonl"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="subset sampling seed")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for name, (load, n, fname) in SPECS.items():
        rows = list(load())
        total = len(rows)
        if n and n < total:
            keep = set(random.Random(args.seed).sample(range(total), n))
            rows = [r for r in rows if r["idx"] in keep]
        with open(os.path.join(args.out_dir, fname), "w") as f:
            for r in rows:
                r = dict(id=f"{name}_{r['idx']}", dataset=name, **r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:8s} {total:5d} in split -> {len(rows):4d} rows -> data/{fname}")


if __name__ == "__main__":
    main()
