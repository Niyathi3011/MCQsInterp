"""
The CONTROLLED looping conditions, in the same row format as natural_looping/, so
natural_looping/generate.py can run them under the same sampled settings.

Same 150 MATH-500 problems as the original greedy catch experiment
(data/force_r1.jsonl: level >= 3, numeric answer), four conditions:
  open     no options                                      (can it solve it?)
  aligned  (A) correct number  (B) unrelated sentence       gold = A
  swap     (A) unrelated sentence  (B) correct number       gold = B  (position control)
  catch    (A) WRONG number  (B) unrelated sentence         no correct option

...each under two prompts:
  boxed       "Please reason step by step, and put your final answer within
              \\boxed{}." (letter boxed for the MCQs) -- identical to natural_looping
  answerline  "End with a line formatted exactly as: Answer: (X)  where X is A or B."
              -- identical to the original greedy catch runs (results/raw_r1.jsonl)

Options, wrong numbers and sentences are copied from data/force_r1.jsonl, so every
row is the exact item of the greedy experiment.

    python controlled_looping/build_datasets.py     # -> controlled_looping/data/<prompt>/<condition>.jsonl
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "natural_looping"))
from build_datasets import OPEN_INSTR, mcq_instr  # noqa: E402

CONDITIONS = ["open", "aligned", "swap", "catch"]


def prompt_of(style, q, opts):
    if style == "boxed":
        if opts is None:
            return f"{q}\n{OPEN_INSTR}"
        return f"{q}\nA) {opts[0]}\nB) {opts[1]}\n{mcq_instr(['A', 'B'])}"
    if opts is None:                                       # answerline = original runs
        return f"{q}\n\nEnd with a line formatted exactly as: Answer: <number>"
    return (f"{q}\n(A) {opts[0]}\n(B) {opts[1]}\n\n"
            "End with a line formatted exactly as: Answer: (X)  where X is A or B.")


def main():
    src = [json.loads(l) for l in open(os.path.join(ROOT, "data", "force_r1.jsonl"))]
    for style in ("boxed", "answerline"):
        out_dir = os.path.join(HERE, "data", style)
        os.makedirs(out_dir, exist_ok=True)
        files = {c: open(os.path.join(out_dir, f"{c}.jsonl"), "w") for c in CONDITIONS}
        n = dict.fromkeys(CONDITIONS, 0)
        for r in src:
            cond = "open" if r["kind"] == "stage1_open" else r.get("variant")
            if cond not in CONDITIONS:
                continue
            opts = None if cond == "open" else [r["option_A"], r["option_B"]]
            row = dict(
                id=f"ctrl_{cond}_{r['source_idx']}", dataset=f"ctrl_{cond}", condition=cond,
                prompt_style=style, idx=r["source_idx"], question=r["question"],
                format="open" if opts is None else "mcq",
                options=opts, letters=None if opts is None else ["A", "B"],
                # open: the number; aligned/swap: the letter; catch: no correct answer
                gold=r["gold_value"] if opts is None else r.get("gold_letter"),
                gold_value=r["gold_value"],
                prompt=prompt_of(style, r["question"], opts))
            files[cond].write(json.dumps(row, ensure_ascii=False) + "\n")
            n[cond] += 1
        for f in files.values():
            f.close()
        print(f"{style:10s} " + "  ".join(f"{c}={n[c]}" for c in CONDITIONS)
              + f"  -> controlled_looping/data/{style}/")


if __name__ == "__main__":
    main()
