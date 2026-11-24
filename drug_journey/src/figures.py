"""Step 6: figures for the "new drug's journey" (reads results/journey.json).

Out: results/fig1_gate_auc.png, fig2_hit_rates.png, fig3_drivers.png, fig4_label_check.png,
     fig5_split_check.png, and larger-type slide_fig3/4/5_*.png for the deck
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RESULTS_DIR

import style as S  # palette, rcParams, pretty feature names, save()
STAGES = [("stage1_P1_to_P2", "Phase 1 → Phase 2"), ("stage2_P2_to_P3", "Phase 2 → Phase 3"),
          ("stage3_P3_to_FDA", "Phase 3 → FDA approval")]
COLORS = [S.BLUE, S.ORANGE, S.AQUA]


def main():
    J = json.load(open(RESULTS_DIR / "journey.json"))

    # ---- fig1: AUC per stage with CI
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for i, (k, lab) in enumerate(STAGES):
        f = J[k]["models"]["final"]
        lo, hi = f["roc_auc_ci"]
        ax.plot([lo, hi], [i, i], color=COLORS[i], lw=2.5, solid_capstyle="round")
        ax.plot(f["test_auc"], i, "o", color=COLORS[i], ms=11, markeredgecolor=S.SURFACE, markeredgewidth=2)
        ax.text(hi + 0.008, i, f"{f['test_auc']:.3f}   (n={J[k]['n']:,}, {J[k]['pos_rate']:.0%} pass)",
                va="center", color=S.INK2, fontsize=10)
    ax.axvline(0.5, color=S.INK2, lw=1, ls="--")
    ax.text(0.503, -0.55, "chance", color=S.INK2, fontsize=9, va="center")
    ax.set_yticks(range(3), [f"Stage {i + 1}: {lab}" for i, (_, lab) in enumerate(STAGES)])
    ax.set_ylim(2.5, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.48, 1.08)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Test ROC-AUC (trials started 2015+), 95% bootstrap CI")
    ax.set_title("Test AUC at each gate of a new drug's journey")
    S.save(fig, "fig1_gate_auc.png")

    # ---- fig2: hit rate by predicted quintile (small multiples)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=False)
    for ax, (k, lab), c in zip(axes, STAGES, COLORS):
        q = J[k]["hit_rates"]["by_quintile"]
        ax.bar(range(5), q, width=0.65, color=c)
        for i, v in enumerate(q):
            ax.text(i, v + 0.01, f"{v:.0%}", ha="center", color=S.INK2, fontsize=9,
                    bbox=dict(facecolor=S.SURFACE, edgecolor="none", pad=0.4))
        base = J[k]["pos_rate_test"]
        ax.axhline(base, color=S.INK2, lw=1, ls="--")
        ax.text(4.4, base, f" base {base:.0%}", color=S.INK2, fontsize=8, va="bottom", ha="right")
        ax.set_xticks(range(5), ["top\n20%", "", "middle", "", "bottom\n20%"])
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(q) * 1.2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(lab, fontsize=12)
    axes[0].set_ylabel("Share that actually passed the gate")
    fig.suptitle("Trials ranked by predicted probability (test set)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    S.save(fig, "fig2_hit_rates.png")

    # ---- fig3: top features per stage
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (k, lab), c in zip(axes, STAGES, COLORS):
        imp = pd.read_csv(RESULTS_DIR / f"journey_{k}_importance.csv").head(8)
        y = np.arange(len(imp))
        ax.barh(y, imp.importance, xerr=imp["std"], height=0.6, color=c,
                error_kw={"ecolor": S.INK2, "elinewidth": 1, "capsize": 0})
        ax.set_yticks(y, [S.pretty(f) for f in imp.feature], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=S.INK2, lw=1)
        ax.set_title(lab, fontsize=12)
        ax.set_xlabel("Drop in test AUC when shuffled")
    fig.suptitle("What drives each gate (permutation importance)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    S.save(fig, "fig3_drivers.png")

    # ---- fig4: label sanity, completed vs terminated
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(3)
    comp = [J[k]["label_by_status"].get("Completed", {}).get("rate", np.nan) for k, _ in STAGES]
    term = [J[k]["label_by_status"].get("Terminated", {}).get("rate", np.nan) for k, _ in STAGES]
    b1 = ax.bar(x - 0.19, comp, 0.36, color=S.BLUE, label="Trial completed")
    b2 = ax.bar(x + 0.19, term, 0.36, color=S.MUTED, label="Trial terminated early")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.0%}",
                    ha="center", color=S.INK2, fontsize=10)
    ax.set_xticks(x, [lab for _, lab in STAGES])
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share labelled 'passed'")
    ax.set_ylim(0, max(comp) * 1.25)
    ax.set_title("Label sanity check: completed trials pass far more often")
    ax.legend(loc="upper right", fontsize=9)
    S.save(fig, "fig4_label_check.png")

    # ---- fig5: leakage, random CV vs temporal
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    rc = [J[k]["random_cv_auc"] for k, _ in STAGES]
    tp = [J[k]["temporal_auc_structured"] for k, _ in STAGES]
    b1 = ax.bar(x - 0.19, tp, 0.36, color=S.BLUE, label="Temporal split (honest)")
    b2 = ax.bar(x + 0.19, rc, 0.36, color=S.ORANGE, label="Random 5-fold CV (sees the future)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height():.3f}",
                    ha="center", color=S.INK2, fontsize=10)
    ax.set_xticks(x, [lab for _, lab in STAGES])
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0.5, max(rc + tp) + 0.06)
    ax.set_ylabel("ROC-AUC (structured model)")
    ax.set_title("A random split flatters the later gates (Phase 3: +0.08)")
    ax.legend(loc="upper left", fontsize=9)
    S.save(fig, "fig5_split_check.png")


def slide_versions():
    """Larger-type variants for the slide deck (the deck shows them at ~half size)."""
    J = json.load(open(RESULTS_DIR / "journey.json"))
    with plt.rc_context({"font.size": 17, "axes.titlesize": 19, "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "axes.labelsize": 15, "legend.fontsize": 15}):
        # top features, 5 per gate
        fig, axes = plt.subplots(1, 3, figsize=(20, 4.8))
        short = ["Phase 1 → 2", "Phase 2 → 3", "Phase 3 → FDA"]
        for ax, (k, _), c, lab in zip(axes, STAGES, COLORS, short):
            imp = pd.read_csv(RESULTS_DIR / f"journey_{k}_importance.csv").head(5)
            y = np.arange(len(imp))
            ax.barh(y, imp.importance, height=0.6, color=c)
            ax.set_yticks(y, [S.pretty(f).split(": ", 1)[-1] for f in imp.feature])
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.axvline(0, color=S.INK2, lw=1)
            ax.set_title(lab)
            ax.set_xlabel("AUC drop when shuffled")
        S.save(fig, "slide_fig3_drivers.png")

        x = np.arange(3)
        labels = ["P1 → P2", "P2 → P3", "P3 → FDA"]
        # label sanity
        fig, ax = plt.subplots(figsize=(8, 5))
        comp = [J[k]["label_by_status"]["Completed"]["rate"] for k, _ in STAGES]
        term = [J[k]["label_by_status"]["Terminated"]["rate"] for k, _ in STAGES]
        for off, vals, col, name in [(-0.19, comp, S.BLUE, "Completed"), (0.19, term, S.MUTED, "Terminated early")]:
            bars = ax.bar(x + off, vals, 0.36, color=col, label=name)
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.0%}",
                        ha="center", color=S.INK2)
        ax.set_xticks(x, labels)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 0.58)
        ax.set_ylabel("Share labelled ‘passed’")
        ax.legend(loc="upper right")
        S.save(fig, "slide_fig4_label_check.png")

        # leakage
        fig, ax = plt.subplots(figsize=(8, 5))
        tp = [J[k]["temporal_auc_structured"] for k, _ in STAGES]
        rc = [J[k]["random_cv_auc"] for k, _ in STAGES]
        for off, vals, col, name in [(-0.19, tp, S.BLUE, "Temporal split (honest)"),
                                     (0.19, rc, S.ORANGE, "Random 5-fold CV")]:
            bars = ax.bar(x + off, vals, 0.36, color=col, label=name)
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height():.2f}",
                        ha="center", color=S.INK2)
        ax.set_xticks(x, labels)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0.5, 0.95)
        ax.set_ylabel("ROC-AUC")
        ax.legend(loc="upper left")
        S.save(fig, "slide_fig5_split_check.png")


if __name__ == "__main__":
    main()
    slide_versions()
