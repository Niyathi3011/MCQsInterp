"""
Figures for the NAACL short paper, computed from the stored results.

  figures/stop_gate.pdf  (a) logit(</think>) at the real stop, at healthy doubts
                             (run later stops) and at loop doubts, per pair group
                         (b) per-MLP contribution to the stop-vs-loop logit gap,
                             controlled vs natural pairs

    python paper/naacl_short/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "looping_mechanism", "results", "r1-distill-qwen-7b")

# reference categorical palette, slots 1-3 in fixed order; text in neutral inks
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})

GROUPS = [("greedy_ref", "ctrl\ngreedy"), ("ctrl_cross", "ctrl\nvs aligned"),
          ("ctrl_same", "ctrl\nvs catch"), ("nat_aqua", "nat\nAQuA"),
          ("nat_logiqa", "nat\nLogiQA"), ("nat_math500", "nat\nMATH")]


def panel_a(ax):
    pts = [json.loads(l) for l in open(os.path.join(RES, "doubt_timecourse", "points.jsonl"))]
    vals = {k: [] for k in ("stop", "healthy", "loop")}
    for g, _ in GROUPS:
        G = [r for r in pts if r["group"] == g and r["loop"] and r["finish_stop"]]
        vals["stop"].append(np.mean([r["finish_stop"]["logit"] for r in G]))
        vals["healthy"].append(np.mean([np.mean([d["logit"] for d in r["finish_doubts"]])
                                        for r in G if r["finish_doubts"]]))
        vals["loop"].append(np.mean([r["loop"][0]["logit"] for r in G]))
    x = np.arange(len(GROUPS))
    w = 0.26
    for k, (key, lab, col, hatch) in enumerate([
            ("stop", "real stop (finished run)", BLUE, None),
            ("healthy", "healthy doubt (run later stops)", ORANGE, "////"),
            ("loop", "loop doubt (run never stops)", AQUA, "\\\\\\\\")]):
        ax.bar(x + (k - 1) * w, vals[key], w - 0.03, color=col, label=lab, hatch=hatch,
               edgecolor="white", linewidth=0.0)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in GROUPS], fontsize=6.0)
    ax.set_ylabel(r"logit(</think>)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6.3, loc="upper right", ncol=1, handlelength=1.6)
    ax.set_ylim(0, 50)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_title("(a) The stop gate opens only at the real stop", fontsize=7.5, loc="left", color=INK)
    return vals


def panel_b(ax):
    S = json.load(open(os.path.join(RES, "summary.json")))
    ctrl = np.array(S["ctrl_cross"]["mlp_delta"])
    nat_keys = ["nat_aqua", "nat_logiqa", "nat_math500"]
    n = np.array([S[k]["n"] for k in nat_keys], dtype=float)
    nat = sum(np.array(S[k]["mlp_delta"]) * nk for k, nk in zip(nat_keys, n)) / n.sum()
    L = np.arange(len(ctrl))
    ax.plot(L, ctrl, color=BLUE, linewidth=1.4, marker="o", markersize=2.6, label="controlled (catch vs aligned)")
    ax.plot(L, nat, color=ORANGE, linewidth=1.4, marker="s", markersize=2.6, linestyle="--",
            label="natural (AQuA, LogiQA, MATH)")
    ax.axhline(0, color=INK2, linewidth=0.5)
    ax.set_xlabel("MLP layer")
    ax.set_ylabel("contribution to gap (logits)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    ax.set_title("(b) Same late MLPs carry the gap", fontsize=7.5, loc="left", color=INK)
    rho = S["nat_logiqa"].get("rho_vs_greedy")
    return ctrl, nat, rho


def main():
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 1.95), gridspec_kw=dict(width_ratios=[1.25, 1]))
    vals = panel_a(a)
    ctrl, nat, _ = panel_b(b)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    out = os.path.join(HERE, "figures", "stop_gate.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("panel (a) means:", {k: [round(v, 1) for v in vs] for k, vs in vals.items()})
    print("panel (b) top-6 MLPs controlled:", list(np.argsort(-ctrl)[:6]),
          " natural:", list(np.argsort(-nat)[:6]))
    print("->", out)


if __name__ == "__main__":
    main()
