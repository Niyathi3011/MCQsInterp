"""
Build a run_model.py --data file for the RQ2/RQ3 raw-<think> rebuild.

Takes an existing phase2/loop_pairs.jsonl (the study set) and re-emits, per
problem, the stage1 + aligned/swap/catch stage2 rows using the SAME prompts that
produced the looping, so the regenerated traces line up 1:1 by `id`.

    python phase2/rebuild_data.py --pairs phase2/loop_pairs.jsonl \
        --out data/r1_rebuild.jsonl

Then serve R1 WITHOUT --reasoning-parser and:

    python run_model.py --data data/r1_rebuild.jsonl \
        --out results/raw_r1_rebuild.jsonl \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --modes cot --systems none --cot-tokens 14000 --keep-truncated --workers 4 \
        --stage1-suffix "End with a line formatted exactly as: Answer: <number>" \
        --stage2-suffix "End with a line formatted exactly as: Answer: (X)  where X is A or B."

    python phase2/prep_loop.py --raw results/raw_r1_rebuild.jsonl \
        --out phase2/loop_pairs_raw.jsonl
    python phase2/loop_probe.py --pairs phase2/loop_pairs_raw.jsonl --raw-think \
        --experiment e2b --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   # sanity
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="phase2/loop_pairs.jsonl")
    ap.add_argument("--out", default="data/r1_rebuild.jsonl")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.pairs)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n1 = n2 = 0
    seen = set()
    with open(args.out, "w") as f:
        for r in recs:
            a = r["aligned"]
            sid = a["id"].rsplit("__stage2_", 1)[0]        # "math500_6"
            si = r["source_idx"]
            if sid not in seen:
                f.write(json.dumps({
                    "id": f"{sid}__stage1", "kind": "stage1_open", "dataset": "math500",
                    "source_idx": si, "question": a["question"],
                    "gold_value": a["gold_value"],
                }) + "\n")
                seen.add(sid)
                n1 += 1
            for v in ("aligned", "swap", "catch"):
                p = r[v]
                f.write(json.dumps({
                    "id": f"{sid}__stage2_{v}", "kind": "stage2_mcq", "dataset": "math500",
                    "source_idx": si, "question": p["question"],
                    "option_A": str(p["option_A"]), "option_B": str(p["option_B"]),
                    "gold_letter": p["gold_letter"], "gold_value": p["gold_value"],
                    "variant": v,
                }) + "\n")
                n2 += 1

    print(f"wrote {n1} stage1 + {n2} stage2 rows -> {args.out}")


if __name__ == "__main__":
    main()
