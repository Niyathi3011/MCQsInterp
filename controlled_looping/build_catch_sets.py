"""
The catch-vs-aligned manipulation on the other benchmarks, so each benchmark gets its
own stop-signal values learned from its own controlled pairs (and tested on its own
natural loops, which are held out at analysis time).

  gsm8k_ctrl/   200 GSM8K questions as two-option MCQs, built like the MATH 150
                (same wrong-number rule and distractor sentences as build_dataset.py):
                  aligned  (A) correct number  (B) unrelated sentence   gold = A
                  catch    (A) WRONG number    (B) unrelated sentence   no correct option
  logiqa_forms/ 200 LogiQA questions with the correct option REMOVED (3 wrong ones
                kept, relabelled A-C).  The aligned form is the original 4-option
                question, already run in natural_looping (same ids, logiqa_<idx>).

AQuA's catch form already exists (aqua_forms/minus_gold.jsonl, 227 items; aligned =
the original 5 options). MATH's is data/boxed/{catch,aligned}.jsonl.

Questions are a random 200 (seed 0) of the natural 500-question files, so the
natural runs of the same questions exist at T=0 and T=0.6. Same boxed prompt as
natural_looping.

    python controlled_looping/build_catch_sets.py
"""
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
from build_dataset import wrong_number  # noqa: E402
from build_datasets import mcq_instr  # noqa: E402
from distractor_sentences import SENTENCES  # noqa: E402

N = 200
NOTA = re.compile(r"none of (these|the above|them)|all of the above", re.I)


def load(name):
    return [json.loads(l) for l in open(os.path.join(ROOT, "natural_looping", "data", name))]


def gsm8k(rng):
    rows = [r for r in load("gsm8k_test_500.jsonl") if re.fullmatch(r"-?\d+(\.\d+)?", r["gold"])]
    rows = sorted(rng.sample(rows, N), key=lambda r: r["idx"])
    out_dir = os.path.join(HERE, "data", "gsm8k_ctrl")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aligned.jsonl"), "w") as fa, \
         open(os.path.join(out_dir, "catch.jsonl"), "w") as fc:
        for r in rows:
            sent = rng.choice(SENTENCES)
            for cond, a, gold, f in (("aligned", r["gold"], "A", fa),
                                     ("catch", wrong_number(r["gold"], rng), None, fc)):
                f.write(json.dumps(dict(
                    id=f"gsm8k_{cond}_{r['idx']}", dataset=f"gsm8k_{cond}", condition=cond,
                    idx=r["idx"], question=r["question"], format="mcq", options=[a, sent],
                    letters=["A", "B"], gold=gold, gold_value=r["gold"],
                    prompt=f"{r['question']}\nA) {a}\nB) {sent}\n{mcq_instr(['A', 'B'])}"),
                    ensure_ascii=False) + "\n")
    print(f"gsm8k : {len(rows)} questions -> {out_dir}/{{aligned,catch}}.jsonl")


def logiqa(rng):
    rows = [r for r in load("logiqa_test_500.jsonl") if not any(NOTA.search(o) for o in r["options"])]
    rows = sorted(rng.sample(rows, N), key=lambda r: r["idx"])
    out_dir = os.path.join(HERE, "data", "logiqa_forms")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "minus_gold.jsonl"), "w") as f:
        for r in rows:
            gi = r["letters"].index(r["gold"])
            wrong = [o for j, o in enumerate(r["options"]) if j != gi]
            letters = list("ABCD")[:len(wrong)]
            lines = "\n".join(f"{L}) {t}" for L, t in zip(letters, wrong))
            f.write(json.dumps(dict(
                id=f"logiqa_minusgold_{r['idx']}", dataset="logiqa_minus_gold", idx=r["idx"],
                format="mcq", question=r["question"], options=wrong, letters=letters,
                gold=None, removed_gold=r["options"][gi], aligned_id=r["id"],
                prompt=f"{r['question']}\n{lines}\n{mcq_instr(letters)}"), ensure_ascii=False) + "\n")
    print(f"logiqa: {len(rows)} questions -> {out_dir}/minus_gold.jsonl")


def main():
    gsm8k(random.Random(0))
    logiqa(random.Random(0))


if __name__ == "__main__":
    main()
