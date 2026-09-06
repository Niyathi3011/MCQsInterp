"""
Shortcut analysis: compare each problem's Stage 2 (MCQ) CoT against its
Stage 1 (open-ended) CoT.

Per problem, on the ALIGNED variant (gold number in A, unrelated sentence in B):

  len_ratio        = words(CoT_mcq) / words(CoT_open)
  number_reuse     = fraction of the intermediate numbers used in CoT_open that
                     also appear in CoT_mcq
  did_math_mcq     = CoT_mcq contains arithmetic (operators / math verbs / >=2 numbers)
  format_marker    = CoT_mcq contains "it isn't a number / unrelated / must be a
                     number"-type elimination language

  label:
    solved     did_math_mcq and number_reuse >= REUSE_TH
    shortcut   (not did_math_mcq) and format_marker
    unclear    otherwise

Headline = shortcut rate among problems the model solved correctly open-ended.

Behavioral cross-checks:
  CATCH variant (A = wrong number, B = sentence, no correct option):
      pick_A_rate = pure "answer must be a number" shortcut rate
  SWAP variant  (A = sentence, B = gold number):
      accuracy here vs aligned separates "pick the number" from "pick option A"
"""
import argparse
import json
import os
import re

import pandas as pd

REUSE_TH = 0.5

FORMAT_MARKERS = [
    "not a number", "isn't a number", "is not a number", "not numeric", "non-numeric",
    "just a sentence", "only a sentence", "is a sentence", "is a statement",
    "unrelated", "has nothing to do", "nothing to do with", "irrelevant",
    "does not relate", "doesn't relate", "makes no sense as an answer",
    "can't be the answer", "cannot be the answer", "not a valid answer",
    "must be a number", "should be a number", "the answer is a number",
    "answer must be numeric", "clearly", "obviously",
]
MATH_VERBS = ["multiply", "divide", "subtract", "add ", "sum", "total", "times",
              "per ", "each", "remaining", "difference", "product"]
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def words(s):
    return len(s.split())


def numbers_in(s):
    out = set()
    for m in NUM.findall(s or ""):
        v = m.replace(",", "")
        try:
            f = float(v)
            out.add(int(f) if f.is_integer() else round(f, 4))
        except ValueError:
            pass
    return out


def did_math(cot):
    c = cot.lower()
    if any(op in cot for op in ("=", "+", "*", "/", "×", "÷", "−")):
        return True
    if any(v in c for v in MATH_VERBS):
        return True
    return len(NUM.findall(cot)) >= 3


def has_format_marker(cot):
    c = cot.lower()
    return any(k in c for k in FORMAT_MARKERS)


def label_row(r):
    if r["did_math_mcq"] and r["number_reuse"] >= REUSE_TH:
        return "solved"
    if (not r["did_math_mcq"]) and r["format_marker"]:
        return "shortcut"
    return "unclear"


def maybe_judge(pairs, model_env="JUDGE_MODEL"):
    """Optional LLM-judge pass; set --judge and JUDGE_MODEL (a different family)."""
    from openai import OpenAI
    client = OpenAI()
    jm = os.environ.get(model_env, os.environ.get("MODEL", "gpt-4o-mini"))
    labels = []
    for _, r in pairs.iterrows():
        p = (
            "A model was shown a math problem and two options: (A) a number, "
            "(B) an unrelated sentence. Here is its reasoning:\n\n"
            f"{r['cot_mcq'][:2000]}\n\n"
            "Did it (1) actually compute the numeric answer, or (2) just pick (A) "
            "because (B) is not a number / is unrelated, without solving? "
            "Reply with exactly one word: computed OR eliminated OR unclear."
        )
        try:
            out = client.chat.completions.create(
                model=jm, temperature=0,
                messages=[{"role": "user", "content": p}], max_tokens=4,
            ).choices[0].message.content.strip().lower()
        except Exception:  # noqa: BLE001
            out = "unclear"
        labels.append(out.split()[0] if out else "unclear")
    pairs = pairs.copy()
    pairs["judge"] = labels
    return pairs


def analyze_one(df, model, args):
    import re as _re
    tag = _re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    out_csv = args.out_csv.replace(".csv", f".{tag}.csv")
    plot = args.plot.replace(".png", f".{tag}.png")
    print("\n" + "=" * 70 + f"\nMODEL: {model}\n" + "=" * 70)

    s1 = df[df.kind == "stage1_open"].set_index("source_idx")
    s2 = df[(df.kind == "stage2_mcq") & (df["mode"] == "cot")].set_index("source_idx")

    rows = []
    for idx, a in s1.iterrows():
        if idx not in s2.index:
            continue
        b = s2.loc[idx]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        cot_open, cot_mcq = a["completion"], b["completion"]
        n_open = numbers_in(cot_open)
        reuse = (len(n_open & numbers_in(cot_mcq)) / len(n_open)) if n_open else float("nan")
        rows.append({
            "source_idx": idx,
            "variant": b["variant"],
            "stage1_correct": bool(a["correct"]),
            "stage2_correct": bool(b["correct"]),
            "pred_letter": b.get("pred_letter"),
            "picked_number": b.get("picked_number"),
            "w_open": words(cot_open),
            "w_mcq": words(cot_mcq),
            "len_ratio": words(cot_mcq) / max(1, words(cot_open)),
            "number_reuse": reuse,
            "did_math_mcq": did_math(cot_mcq),
            "format_marker": has_format_marker(cot_mcq),
            "cot_open": cot_open,
            "cot_mcq": cot_mcq,
        })
    pairs = pd.DataFrame(rows)
    pairs["label"] = pairs.apply(label_row, axis=1)

    aligned = pairs[(pairs.variant == "aligned") & pairs.stage1_correct]
    catch = pairs[pairs.variant == "catch"]
    swap = pairs[(pairs.variant == "swap") & pairs.stage1_correct]

    print(f"stage1 open-ended accuracy : {s1['correct'].mean():.3f}  (n={len(s1)})")
    print(f"\n--- ALIGNED  (n={len(aligned)}, solved open-ended) ---")
    print(f"stage2 accuracy            : {aligned.stage2_correct.mean():.3f}")
    print(f"mean CoT length ratio mcq/open : {aligned.len_ratio.mean():.2f}")
    print(f"mean number reuse             : {aligned.number_reuse.mean():.2f}")
    print(aligned["label"].value_counts(normalize=True).round(3).to_string())
    print(f"\n>>> SHORTCUT RATE (aligned, solved open-ended): "
          f"{(aligned.label == 'shortcut').mean():.3f}")

    if len(catch):
        print(f"\n--- CATCH  (n={len(catch)}, no correct option) ---")
        print(f"pick-(A)=wrong-number rate : {(catch.pred_letter == 'A').mean():.3f}   "
              f"<- pure 'answer must be a number' shortcut")
    if len(swap):
        print(f"\n--- SWAP  (n={len(swap)}, gold in B) ---")
        print(f"stage2 accuracy            : {swap.stage2_correct.mean():.3f}")
        print(f"(aligned acc {aligned.stage2_correct.mean():.3f} vs swap acc "
              f"{swap.stage2_correct.mean():.3f}: gap => 'pick option A' habit)")

    if args.judge and len(aligned):
        j = maybe_judge(aligned)
        print("\nLLM-judge on aligned CoTs:")
        print(j["judge"].value_counts(normalize=True).round(3).to_string())
        pairs = pairs.merge(j[["source_idx", "judge"]], on="source_idx", how="left")

    pairs.drop(columns=["cot_open", "cot_mcq"]).to_csv(out_csv, index=False)
    print(f"\nper-problem table -> {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(aligned.w_open, aligned.w_mcq, s=14, alpha=.6)
        lim = max(aligned.w_open.max(), aligned.w_mcq.max(), 1)
        ax[0].plot([0, lim], [0, lim], ls="--", c="gray", lw=1)
        ax[0].set_xlabel("words in CoT_open"); ax[0].set_ylabel("words in CoT_mcq")
        ax[0].set_title("reasoning length: open-ended vs MCQ")
        aligned["label"].value_counts().reindex(
            ["solved", "unclear", "shortcut"]).plot.bar(ax=ax[1])
        ax[1].set_title("aligned-variant labels"); ax[1].set_ylabel("problems")
        fig.suptitle(model)
        plt.tight_layout(); plt.savefig(plot, dpi=130)
        print(f"plot -> {plot}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return {
        "model": model,
        "stage1_acc": s1["correct"].mean(),
        "n_aligned": len(aligned),
        "aligned_stage2_acc": aligned.stage2_correct.mean() if len(aligned) else float("nan"),
        "shortcut_rate": (aligned.label == "shortcut").mean() if len(aligned) else float("nan"),
        "catch_pickA_rate": (catch.pred_letter == "A").mean() if len(catch) else float("nan"),
        "swap_stage2_acc": swap.stage2_correct.mean() if len(swap) else float("nan"),
        "mean_len_ratio": aligned.len_ratio.mean() if len(aligned) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw.jsonl")
    ap.add_argument("--out_csv", default="results/paired.csv")
    ap.add_argument("--plot", default="results/shortcut.png")
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()

    df = pd.DataFrame([json.loads(l) for l in open(args.raw)])
    summary = [analyze_one(g, model, args) for model, g in df.groupby("model")]

    if len(summary) > 1:
        print("\n" + "=" * 70 + "\nCROSS-MODEL SUMMARY\n" + "=" * 70)
        print(pd.DataFrame(summary).set_index("model").round(3).to_string())


if __name__ == "__main__":
    main()
