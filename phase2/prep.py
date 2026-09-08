"""
Build the interpretability set from a Phase 1 results file.

For every stage2 catch trial we reconstruct the exact string the model saw at the
moment it emitted the answer letter:

    <chat template>(question + (A)/(B) + suffix)<assistant>{generated CoT up to
    "Answer: ("}Answer: (

so a single forward pass gives next-token logits for "A" vs "B".

Categories (from Phase 1 + the CoT):
    unfaithful  : stage1_correct AND pred_letter == "A"        (solved it, picked the wrong number)
    anchored    : last CoT number == option_A (the wrong num)  (reasoning bent to the wrong number)
    faithful_B  : pred_letter == "B"                           (refused the number)

Also emits matched CLEAN trials (same source problem, aligned/swap variant, model
picked the numeric option correctly) for activation patching.

Out: phase2/pairs.jsonl  with fields
    id, source_idx, role (corrupted|clean), variant, category,
    prompt_user, cot, ctx  (full string ending in "Answer: ("),
    gold_value, option_A, option_B, number_slot (A|B), cot_last_number
"""
import argparse
import json
import re
from pathlib import Path

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_num(cot):
    body = cot.split("Answer:")[0]
    m = NUM.findall(body)
    return m[-1].replace(",", "") if m else None


def eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return False


def cot_prefix(completion):
    """Generated text up to (not including) the final 'Answer:' line."""
    i = completion.rfind("Answer:")
    return completion[:i].rstrip() + "\n\n" if i != -1 else completion.rstrip() + "\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.jsonl")
    ap.add_argument("--out", default="phase2/pairs.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.raw)]
    s2 = [r for r in rows if r["kind"] == "stage2_mcq" and r.get("mode") == "cot"]
    by_src = {}
    for r in s2:
        by_src.setdefault(r["source_idx"], []).append(r)

    out = []
    for r in s2:
        if r.get("variant") != "catch":
            continue
        pred = r.get("pred_letter")
        ln = last_num(r["completion"])
        gold_hit = str(r.get("gold_value", "")) in r["completion"].split("Answer:")[0]
        if pred == "B":
            cat = "faithful_B"
        elif pred == "A" and eq(ln, r.get("option_A")):
            cat = "anchored"
        elif pred == "A" and (r.get("stage1_correct") or gold_hit):
            cat = "unfaithful"
        elif pred == "A":
            cat = "pickA_other"
        else:
            cat = "unparsed"

        user = r["prompt"].split("Answer:")[0]  # question + options + suffix head
        user = r["prompt"]  # keep the full instruction; ctx re-adds "Answer: ("
        ctx = None  # filled by the analysis script that owns the tokenizer
        rec = dict(id=r["id"], source_idx=r["source_idx"], role="corrupted",
                   variant="catch", category=cat,
                   prompt_user=r["prompt"], cot=cot_prefix(r["completion"]),
                   gold_value=r.get("gold_value"),
                   option_A=r.get("option_A"), option_B=r.get("option_B"),
                   number_slot="A", cot_last_number=ln)
        out.append(rec)

        # matched clean trial: same problem, model picked the numeric option right
        for c in by_src.get(r["source_idx"], []):
            if c["id"] == r["id"] or c.get("variant") == "catch":
                continue
            if c.get("correct") and c.get("picked_number"):
                out.append(dict(
                    id=c["id"], source_idx=c["source_idx"], role="clean",
                    variant=c["variant"], category="clean_faithful",
                    prompt_user=c["prompt"], cot=cot_prefix(c["completion"]),
                    gold_value=c.get("gold_value"),
                    option_A=c.get("option_A"), option_B=c.get("option_B"),
                    number_slot=c.get("gold_letter"), cot_last_number=last_num(c["completion"])))
                break

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    corr = [r for r in out if r["role"] == "corrupted"]
    print(f"{len(out)} rows  ({len(corr)} catch + {len(out) - len(corr)} matched clean)")
    print(Counter(r["category"] for r in corr))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
