"""
The controlled manipulation on REAL questions: AQuA in three forms, same items.

  mcq         the original 5 options (one correct)  -- already run in natural_looping
  minus_gold  the correct option REMOVED, the 4 wrong ones kept (relabelled A-D):
              a real multiple-choice question with no correct option ("natural catch")
  open        no options at all: the model writes its own answer (format control)

Items whose options include "None of these/the above" are excluded (27 of 254):
removing the correct option would make "None of these" correct.

Same prompt style as natural_looping (DeepSeek's boxed prompt).

    python controlled_looping/build_aqua_forms.py   # -> controlled_looping/data/aqua_forms/
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
from build_datasets import OPEN_INSTR, mcq_instr  # noqa: E402

NOTA = re.compile(r"none of (these|the above|them)", re.I)


def main():
    src = [json.loads(l) for l in open(os.path.join(ROOT, "natural_looping", "data",
                                                     "aqua_test_254.jsonl"))]
    out_dir = os.path.join(HERE, "data", "aqua_forms")
    os.makedirs(out_dir, exist_ok=True)
    keep = [r for r in src if not any(NOTA.search(o) for o in r["options"])]
    with open(os.path.join(out_dir, "minus_gold.jsonl"), "w") as fm, \
         open(os.path.join(out_dir, "open.jsonl"), "w") as fo:
        for r in keep:
            gi = r["letters"].index(r["gold"])
            wrong = [o for j, o in enumerate(r["options"]) if j != gi]
            letters = list("ABCDE")[:len(wrong)]
            lines = "\n".join(f"{L}) {t}" for L, t in zip(letters, wrong))
            fm.write(json.dumps(dict(
                id=f"aqua_minusgold_{r['idx']}", dataset="aqua_minus_gold", idx=r["idx"],
                format="mcq", question=r["question"], options=wrong, letters=letters,
                gold=None, removed_gold=r["options"][gi],
                prompt=f"{r['question']}\n{lines}\n{mcq_instr(letters)}"), ensure_ascii=False) + "\n")
            # open form: the gold option's text as the answer, scored with math-verify
            ans = r["options"][gi].replace("√", "\\sqrt").replace("–", "-")
            fo.write(json.dumps(dict(
                id=f"aqua_open_{r['idx']}", dataset="aqua_open", idx=r["idx"], format="open",
                question=r["question"], gold=ans, gold_letter=r["gold"],
                prompt=f"{r['question']}\n{OPEN_INSTR}"), ensure_ascii=False) + "\n")
    print(f"AQuA: {len(src)} items, {len(src) - len(keep)} with a 'None of these' option excluded "
          f"-> {len(keep)} items in minus_gold.jsonl and open.jsonl ({out_dir})")


if __name__ == "__main__":
    main()
