"""
Generate figures DIRECTLY from a raw R1 results file (e.g. raw_r1.jsonl) --
unlike make_figures.py (which plots hard-coded numbers from the write-up),
every number here is computed live from the data, so the figures stay correct
if the file changes.

    python phase2/make_figures_from_raw.py --raw ~/Desktop/raw_r1.jsonl \
        --out phase2/figures_raw

Produces:
    figA_nontermination_by_bucket.png   open-ended / MCQ-correct / MCQ-no-correct, by system
    figB_nontermination_by_variant.png  aligned/swap/decoy/catch/broken, by system
    figC_accuracy_by_variant.png        stage1/aligned/swap/decoy accuracy (all vs finished)
    figD_repetition_histogram.png       loop severity -- most-repeated-sentence count

Requires matplotlib only; reuses no pod/model access. Mirrors analyze_loop.py's
aggregation logic (kept a separate, self-contained script since analyze_loop.py
wasn't written to be imported).
"""
import argparse
import collections
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AMBER = "#B5651D"   # "none" system / aligned-family
BLUE = "#2A6FA8"    # "exam" system / catch-family
GREY = "#9A8F7B"
GOOD = "#3E7A4D"
BAD = "#B5451F"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

HAS_CORRECT = {"aligned", "swap", "decoy"}
NO_CORRECT = {"catch", "broken"}


def max_line_repeat(text):
    parts = [re.sub(r"\s+", " ", s).strip().lower()
             for s in re.split(r"(?<=[.\n])", text or "") if len(s.strip()) > 15]
    c = collections.Counter(parts)
    top = c.most_common(1)[0] if c else ("", 0)
    return top[1]


def load(path):
    return [json.loads(l) for l in open(path)]


def figA_nontermination_by_bucket(rows, out):
    s1 = [r for r in rows if r["kind"] == "stage1_open"]
    s2 = [r for r in rows if r["kind"] == "stage2_mcq"]
    systems = sorted(set(r.get("system_name", "none") for r in rows))

    labels = ["Open-ended", "MCQ, correct\noption exists", "MCQ, NO correct\noption"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.35
    x = range(len(labels))
    for i, sy in enumerate(systems):
        oe = [r for r in s1 if r.get("system_name", "none") == sy]
        hc = [r for r in s2 if r.get("variant") in HAS_CORRECT and r.get("system_name", "none") == sy]
        nc = [r for r in s2 if r.get("variant") in NO_CORRECT and r.get("system_name", "none") == sy]
        vals = [100 * sum(bool(r.get("truncated")) for r in g) / len(g) for g in (oe, hc, nc)]
        offset = (i - (len(systems) - 1) / 2) * w
        color = AMBER if sy == "none" else BLUE
        bars = ax.bar([xi + offset for xi in x], vals, width=w, color=color, label=f"system={sy}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of trials that never produce an answer")
    ax.set_title(f"Figure A — Non-termination by question type (n={len(set(r['source_idx'] for r in rows))} problems)",
                 loc="left", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out}/figA_nontermination_by_bucket.png", dpi=200)
    plt.close()


def figB_nontermination_by_variant(rows, out):
    s2 = [r for r in rows if r["kind"] == "stage2_mcq"]
    variants = ["aligned", "swap", "decoy", "catch", "broken"]
    systems = sorted(set(r.get("system_name", "none") for r in rows))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.35
    x = range(len(variants))
    for i, sy in enumerate(systems):
        vals = []
        for v in variants:
            g = [r for r in s2 if r.get("variant") == v and r.get("system_name", "none") == sy]
            vals.append(100 * sum(bool(r.get("truncated")) for r in g) / len(g) if g else 0)
        offset = (i - (len(systems) - 1) / 2) * w
        color = AMBER if sy == "none" else BLUE
        bars = ax.bar([xi + offset for xi in x], vals, width=w, color=color, label=f"system={sy}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants)
    ax.axvspan(2.5, 4.5, color=BAD, alpha=0.06, zorder=0)
    ax.set_ylabel("non-termination rate")
    ax.set_ylim(0, 100)
    ax.set_title("Figure B — Non-termination by variant\n(catch/broken = no correct option, shaded)",
                 loc="left", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out}/figB_nontermination_by_variant.png", dpi=200)
    plt.close()


def figC_accuracy_by_variant(rows, out):
    groups = [("stage1_open", None), ("stage2_mcq", "aligned"), ("stage2_mcq", "swap"), ("stage2_mcq", "decoy")]
    labels = ["open-ended", "aligned", "swap", "decoy"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.35
    x = range(len(labels))
    acc_all, acc_fin = [], []
    for kind, variant in groups:
        g = [r for r in rows if r["kind"] == kind and (variant is None or r.get("variant") == variant)]
        fin = [r for r in g if not r.get("truncated")]
        acc_all.append(100 * sum(bool(r.get("correct")) for r in g) / len(g) if g else 0)
        acc_fin.append(100 * sum(bool(r.get("correct")) for r in fin) / len(fin) if fin else 0)
    b1 = ax.bar([xi - w / 2 for xi in x], acc_all, width=w, color=GREY, label="accuracy, all trials")
    b2 = ax.bar([xi + w / 2 for xi in x], acc_fin, width=w, color=GOOD, label="accuracy, finished trials only")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                     ha="center", fontsize=9.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 112)
    ax.set_title("Figure C — Accuracy where a correct answer exists", loc="left",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out}/figC_accuracy_by_variant.png", dpi=200)
    plt.close()


def figD_repetition_histogram(rows, out):
    tnc = [r for r in rows if r.get("kind") == "stage2_mcq" and r.get("variant") in NO_CORRECT
           and r.get("truncated")]
    counts = [max_line_repeat(r.get("reasoning") or "") for r in tnc]
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = [1, 2, 5, 10, 20, 50, 100, 200, 600]
    ax.hist(counts, bins=bins, color=BLUE, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("most-repeated sentence's count within the trace (log scale)")
    ax.set_ylabel("number of truncated no-correct-option traces")
    loopy = sum(1 for c in counts if c >= 5)
    ax.set_title(f"Figure D — Loop severity (n={len(counts)} truncated traces, "
                 f"{100*loopy/len(counts):.0f}% repeat ≥5x)", loc="left",
                 fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out}/figD_repetition_histogram.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.expanduser("~/Desktop/raw_r1.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures_raw"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load(args.raw)
    print(f"{args.raw}: {len(rows)} rows, {len(set(r['source_idx'] for r in rows))} problems")

    figA_nontermination_by_bucket(rows, args.out)
    figB_nontermination_by_variant(rows, args.out)
    figC_accuracy_by_variant(rows, args.out)
    figD_repetition_histogram(rows, args.out)

    for name in sorted(os.listdir(args.out)):
        if name.endswith(".png"):
            print(f"  wrote {args.out}/{name}")


if __name__ == "__main__":
    main()
