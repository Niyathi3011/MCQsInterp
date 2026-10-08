"""
"Allow None": the same items as the catch experiment, with ONE sentence added to the
instruction that lets the model decline:
    "If none of the options is correct, put None within \\boxed{}."

If loops come from the answer-option conflict, giving the model a legitimate way out
should remove them (resolving the conflict), where stopping did not.

  math_catch / math_aligned          the 150 MATH two-option items (boxed prompt)
  gsm8k_catch / gsm8k_aligned        the 200 GSM8K two-option items
  aqua_minus_gold / aqua_original    the 227 AQuA items, correct option removed / all 5 options
  logiqa_minus_gold / logiqa_original  the 200 LogiQA items, correct option removed / all 4 options
The "original" / "aligned" forms check the cost: does the model now decline questions
that DO have a correct option?

    python controlled_looping/build_allow_none.py   # -> controlled_looping/data/allow_none/
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(HERE, "data")
EXTRA = " If none of the options is correct, put None within \\boxed{}."
NOTA = re.compile(r"none of (these|the above|them)", re.I)


def load(p):
    return [json.loads(l) for l in open(p)]


def write(name, rows):
    with open(os.path.join(D, "allow_none", f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(rows)} rows")


def with_extra(r, name):
    p = r["prompt"]
    assert p.rstrip().endswith("within \\boxed{}."), p[-120:]
    return dict(r, id=f"an_{name}_{r['idx']}", dataset=f"allow_none_{name}", prompt=p.rstrip() + EXTRA,
                allow_none=True)


def main():
    os.makedirs(os.path.join(D, "allow_none"), exist_ok=True)
    write("math_catch", [with_extra(r, "math_catch") for r in load(os.path.join(D, "boxed", "catch.jsonl"))])
    write("math_aligned", [with_extra(r, "math_aligned") for r in load(os.path.join(D, "boxed", "aligned.jsonl"))])
    write("aqua_minus_gold", [with_extra(r, "aqua_minus_gold")
                              for r in load(os.path.join(D, "aqua_forms", "minus_gold.jsonl"))])
    keep = {r["idx"] for r in load(os.path.join(D, "aqua_forms", "minus_gold.jsonl"))}
    orig = [r for r in load(os.path.join(ROOT, "natural_looping", "data", "aqua_test_254.jsonl")) if r["idx"] in keep]
    write("aqua_original", [with_extra(r, "aqua_original") for r in orig])
    write("gsm8k_catch", [with_extra(r, "gsm8k_catch") for r in load(os.path.join(D, "gsm8k_ctrl", "catch.jsonl"))])
    write("gsm8k_aligned", [with_extra(r, "gsm8k_aligned")
                            for r in load(os.path.join(D, "gsm8k_ctrl", "aligned.jsonl"))])
    lq = load(os.path.join(D, "logiqa_forms", "minus_gold.jsonl"))
    write("logiqa_minus_gold", [with_extra(r, "logiqa_minus_gold") for r in lq])
    keep = {r["idx"] for r in lq}
    orig = [r for r in load(os.path.join(ROOT, "natural_looping", "data", "logiqa_test_500.jsonl")) if r["idx"] in keep]
    write("logiqa_original", [with_extra(r, "logiqa_original") for r in orig])


if __name__ == "__main__":
    main()
