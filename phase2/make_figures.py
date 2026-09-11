"""
Generate all figures for the RQ2/RQ3 write-up (report + docx) from the exact
numbers reported by each experiment -- no external chart tool, no hand-copied
screenshots, so every figure regenerates identically from source.

    python phase2/make_figures.py [--out phase2/figures]

Produces, in --out (default phase2/figures/):
    fig1_nontermination.png   non-termination rate by question type      (background)
    fig2_e2b_logitlens.png    aligned vs catch, drive-to-stop by layer   (E2b, v1 & v2)
    fig3_e2c_attribution.png  contribution to the stop-signal gap        (E2c, v1)
    fig4_e3a_causal.png       % of gap recovered/removed, MLP27 vs top-6 (E3a, v1)
    fig5_e3b_fidelity.png     reconstruction fix, before/after           (E3b fidelity, n=6)
    fig6_e3b_full_flip.png    full behavioural flip                      (E3b, n=30)

Numbers are hard-coded from the experiment logs (see RQ2_RQ3_writeup.md) --
this script has no dependency on the pod, the model, or any data file.
Requires matplotlib only.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AMBER = "#B5651D"   # aligned / terminates
BLUE = "#2A6FA8"    # catch / loops
GREY = "#9A8F7B"    # control / neutral
GOOD = "#3E7A4D"
BAD = "#B5451F"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def fig1_nontermination(out):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = ["Open-ended\n(no options given)", "MCQ, a correct\noption exists",
              "MCQ, NO correct\noption (catch)"]
    vals = [8, 10, 65]
    colors = [BLUE, BLUE, BAD]
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v}%", va="center",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 78)
    ax.set_xlabel("% of trials that never produce an answer")
    ax.set_title("Figure 1 — Non-termination rate by question type", loc="left",
                  fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{out}/fig1_nontermination.png", dpi=200)
    plt.close()


def fig2_e2b_logitlens(out):
    # layer -> [aligned_projection, catch_projection]  (final-layer logit-lens, per layer)
    v1 = {0: [-0.887, -1.090], 1: [-0.359, -1.386], 2: [-0.914, -1.478], 3: [-0.799, -1.402],
          4: [-0.885, -1.175], 5: [0.488, -0.959], 6: [1.410, -0.530], 7: [2.391, -0.094],
          8: [2.568, -0.310], 9: [2.224, -0.332], 10: [2.071, -0.352], 11: [2.827, -0.390],
          12: [3.027, -0.505], 13: [3.213, -0.546], 14: [2.679, -0.638], 15: [2.544, -0.596],
          16: [1.649, -0.774], 17: [2.520, -0.668], 18: [2.659, -0.786], 19: [4.180, -0.898],
          20: [5.140, -0.836], 21: [7.295, -0.751], 22: [6.914, -0.670], 23: [8.571, -0.732],
          24: [7.457, -0.611], 25: [7.881, -0.587], 26: [7.750, -0.399], 27: [7.380, -0.593],
          28: [11.774, 0.293]}
    v2 = {0: [0.341, -0.820], 1: [-0.322, -1.144], 2: [-0.785, -1.230], 3: [-0.427, -1.161],
          4: [0.229, -0.950], 5: [1.279, -0.589], 6: [2.253, -0.167], 7: [4.009, 0.256],
          8: [4.575, 0.078], 9: [3.856, -0.081], 10: [4.068, -0.121], 11: [5.760, -0.126],
          12: [5.264, -0.205], 13: [5.089, -0.320], 14: [4.534, -0.496], 15: [4.987, -0.463],
          16: [4.415, -0.566], 17: [5.342, -0.538], 18: [5.953, -0.602], 19: [7.694, -0.713],
          20: [9.435, -0.569], 21: [11.874, -0.472], 22: [15.491, -0.348], 23: [19.502, -0.397],
          24: [21.455, -0.288], 25: [21.535, -0.217], 26: [21.179, -0.051], 27: [22.517, -0.563],
          28: [31.402, 0.579]}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    titles = ["Dataset v1 · temp 0, n=94\np = 3.8e-17",
              "Dataset v2 · temp 0.6, raw tags, n=81\np = 5.4e-15"]
    for ax, data, title in zip(axes, [v1, v2], titles):
        layers = sorted(data)
        aligned = [data[l][0] for l in layers]
        catch = [data[l][1] for l in layers]
        ax.plot(layers, aligned, color=AMBER, lw=2.2, label="aligned / swap (terminates)")
        ax.plot(layers, catch, color=BLUE, lw=2.2, label="catch (no correct option, loops)")
        ax.axhline(0, color="#999999", lw=0.8, ls=":")
        ax.annotate(f"{aligned[-1]:+.1f}", (28, aligned[-1]), textcoords="offset points",
                    xytext=(-38, 4), color=AMBER, fontweight="bold")
        ax.annotate(f"{catch[-1]:+.2f}", (28, catch[-1]), textcoords="offset points",
                    xytext=(-10, -14), color=BLUE, fontweight="bold")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("layer")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("projection onto the \"</think>\" direction\n(“drive to stop”)")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Figure 2 — Logit-lens: the drive to stop, layer by layer (E2b)",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{out}/fig2_e2b_logitlens.png", dpi=200, bbox_inches="tight")
    plt.close()


def fig3_e2c_attribution(out):
    comps = ["MLP 27", "MLP 26", "MLP 22", "MLP 24", "MLP 20", "MLP 25", "MLP 19", "MLP 23",
             "all 336\nheads"]
    vals = [5.587, 1.060, 0.798, 0.706, 0.556, 0.433, 0.187, 0.143, 1.79]
    colors = [BLUE] * 8 + [GREY]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.barh(comps, vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.08, b.get_y() + b.get_height() / 2, f"+{v:.2f}", va="center", fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("contribution to the aligned−catch “</think>”-logit gap")
    ax.set_title("Figure 3 — Which components write the stop signal (E2c, dataset v1)",
                  loc="left", fontsize=12.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out}/fig3_e2c_attribution.png", dpi=200)
    plt.close()


def fig4_e3a_causal(out):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    groups = ["rescue\n(catch ← aligned)", "lesion\n(aligned → ablated)"]
    mlp27 = [46, 55]
    top6 = [84, 79]
    x = range(len(groups))
    w = 0.32
    b1 = ax.bar([i - w / 2 for i in x], mlp27, width=w, color=BLUE, alpha=0.55, label="MLP 27 alone")
    b2 = ax.bar([i + w / 2 for i in x], top6, width=w, color=BLUE, label="all 6 late MLPs")
    ax.axhline(1, color=GREY, lw=1.6, ls="--")
    ax.text(1.55, 3, "control (unrelated MLP) = 1%", color=GREY, fontsize=9, ha="right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f}%",
                     ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("% of the stop-signal gap moved")
    ax.set_ylim(0, 95)
    ax.set_title("Figure 4 — Forcing the stop signal on/off (E3a, dataset v1, n=94)",
                  loc="left", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out}/fig4_e3a_causal.png", dpi=200)
    plt.close()


def fig5_e3b_fidelity(out):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = ["v1 (reconstructed\nboundary)", "v2 (exact\nboundary)"]
    vals = [17, 100]
    colors = [BAD, GOOD]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("normal trials emitting the stop token\nimmediately from the decision point")
    ax.set_title("Figure 5 — The reconstruction fix, confirmed (E3b fidelity check, n=6)",
                  loc="left", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out}/fig5_e3b_fidelity.png", dpi=200)
    plt.close()


def fig6_e3b_full_flip(out):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    labels = ["catch\nunpatched", "catch\nRESCUE\n(6 MLPs)", "catch\nCONTROL\n(MLP 15)",
              "aligned\nunpatched", "aligned\nLESION\n(6 MLPs)"]
    vals = [0, 100, 0, 100, 0]
    colors = [BLUE, GOOD, GREY, AMBER, BAD]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v}%", ha="center", fontsize=13,
                 fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of 30 trials emitting </think>")
    ax.set_title("Figure 6 — Full behavioural flip: forcing 6 MLPs on/off (n=30, dataset v2)",
                  loc="left", fontsize=12.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{out}/fig6_e3b_full_flip.png", dpi=200)
    plt.close()


FIGURES = [
    fig1_nontermination, fig2_e2b_logitlens, fig3_e2c_attribution,
    fig4_e3a_causal, fig5_e3b_fidelity, fig6_e3b_full_flip,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn in FIGURES:
        fn(args.out)
    for name in sorted(os.listdir(args.out)):
        if name.endswith(".png"):
            print(f"  wrote {args.out}/{name}")
    print(f"\n{len(FIGURES)} figures written -> {args.out}/")


if __name__ == "__main__":
    main()
