"""
Two-stage dataset builder for the shortcut experiment.

Per source GSM8K problem we emit:

  1 x stage1_open   - the raw problem, no options. Model must reason and produce
                      a number. This CoT is the reference "did the real work" trace.

  1 x stage2_mcq    - the SAME problem as a 2-option MCQ. Variants:

      aligned   (default)   (A) = gold number      (B) = unrelated sentence   gold = A
      swap      (--swap-frac)   (A) = unrelated sentence  (B) = gold number    gold = B
      catch     (--catch-frac)  (A) = WRONG number     (B) = unrelated sentence  gold = None
                                (neither option is correct -> picking (A) here
                                 is a pure "it's a math problem, answer is a
                                 number" shortcut)

Analysis (analyze.py) compares the stage2 CoT against the stage1 CoT to decide,
per problem, whether the model re-solved or shortcut by format elimination.
"""
import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from distractor_sentences import SENTENCES


def extract_gold(answer_field: str) -> str:
    m = re.search(r"####\s*(.+)", answer_field)
    if not m:
        raise ValueError("no #### answer in GSM8K row")
    return m.group(1).strip().replace(",", "").replace("$", "")


def wrong_number(gold: str, rng: random.Random) -> str:
    """A plausible wrong number, formatted like `gold`, for catch trials."""
    try:
        g = float(gold)
    except ValueError:
        return gold + "7"
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
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300, help="number of source problems")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--swap-frac", type=float, default=0.25,
                    help="fraction of stage2 rows with gold as option B")
    ap.add_argument("--catch-frac", type=float, default=0.25,
                    help="fraction of stage2 rows where option A is a wrong number")
    ap.add_argument("--out", default="data/dataset.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset("gsm8k", "main", split=args.split)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[: args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n1 = n2 = 0
    with out.open("w") as f:
        for idx in idxs:
            q = ds[idx]["question"].strip()
            gold = extract_gold(ds[idx]["answer"])
            sent = rng.choice(SENTENCES)

            f.write(json.dumps({
                "id": f"{idx}__stage1",
                "kind": "stage1_open",
                "source_idx": idx,
                "question": q,
                "gold_value": gold,
            }) + "\n")
            n1 += 1

            r = rng.random()
            if r < args.catch_frac:
                variant, opt_a, opt_b, gold_letter = (
                    "catch", wrong_number(gold, rng), sent, None)
            elif r < args.catch_frac + args.swap_frac:
                variant, opt_a, opt_b, gold_letter = ("swap", sent, gold, "B")
            else:
                variant, opt_a, opt_b, gold_letter = ("aligned", gold, sent, "A")

            f.write(json.dumps({
                "id": f"{idx}__stage2",
                "kind": "stage2_mcq",
                "source_idx": idx,
                "question": q,
                "option_A": str(opt_a),
                "option_B": str(opt_b),
                "gold_letter": gold_letter,
                "gold_value": gold,
                "variant": variant,
            }) + "\n")
            n2 += 1

    print(f"wrote {n1} stage1 + {n2} stage2 rows -> {out}")


if __name__ == "__main__":
    main()
