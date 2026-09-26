#!/usr/bin/env python3
"""Render the deployment-consequence figure from frozen audit tables."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_tables" / "asb_deployment_consequence_audit"
FINAL = ROOT / "reproducibility" / "output" / "ranking_transfer"
OUT = ROOT / "figures" / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#263238"
BLUE = "#2f6f9f"
RED = "#b64b4b"
AMBER = "#9a6b21"
GREEN = "#3c7752"
GRAY = "#66727c"
GRID = "#d9dee1"

plt.rcParams.update({
    # Match the Times family used by the ICLR manuscript.
    "font.family": "Nimbus Roman",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.0,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    asb = pd.read_csv(FINAL / "heldout_regret.csv")
    asb = asb[asb.stable_development_opposition]
    mixed = pd.read_csv(FINAL / "mixed_family_selection.csv")
    cost = pd.read_csv(FINAL / "mixed_family_cost_sensitive_utility.csv")

    colors = {"first_crossing": BLUE, "two_consecutive": RED, "cumulative_window3": AMBER}
    labels = {"first_crossing": "First crossing", "two_consecutive": "Two consecutive", "cumulative_window3": "Cumulative window-3"}
    policies = list(labels)
    alphas = [0.01, 0.02, 0.05, 0.10]

    fig = plt.figure(figsize=(10.8, 3.45))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.0, 1.42], wspace=0.42)
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[i]) for i in range(3)]

    # Panel A: each dot is one development-stable opposed pair evaluated on
    # the frozen outer test. Zero-regret cases remain visible. Policies are
    # horizontally dodged within each deployment condition.
    rng = np.random.default_rng(20260905)
    offsets = np.linspace(-0.19, 0.19, len(policies))
    for policy, offset in zip(policies, offsets):
        for i, alpha in enumerate(alphas):
            x = i + offset
            vals = asb[(asb.policy == policy) & asb.alpha.eq(alpha)].unified_outer_regret.to_numpy()
            if len(vals):
                jitter = rng.uniform(-0.055, 0.055, len(vals))
                ax_a.scatter(np.full(len(vals), x) + jitter, vals, s=12,
                             color=colors[policy], alpha=0.34, edgecolor="none", zorder=2)
                q25, q75 = np.quantile(vals, [0.25, 0.75])
                ax_a.vlines(x, q25, q75, color=colors[policy], linewidth=1.0,
                            alpha=0.9, zorder=3)
                ax_a.scatter([x], [np.mean(vals)], s=34, color=colors[policy],
                             edgecolor="white", linewidth=0.45, zorder=4)
    ax_a.axhline(0, color=GRAY, linewidth=0.8, zorder=0)
    ax_a.set_xticks(range(4), ["1%", "2%", "5%", "10%"])
    ax_a.set_xlabel("Nominal target FPR")
    ax_a.set_ylabel("Recall regret of predictive choice")
    ax_a.set_ylim(-0.002, max(0.075, float(asb.unified_outer_regret.max()) * 1.12))
    ax_a.text(-0.16, 1.04, "A", transform=ax_a.transAxes, weight="bold", fontsize=10)
    clean(ax_a)

    # Panel B: fold-level observations are faint; lines and larger points are
    # the cell means. The categorical x positions keep deployment spacing
    # identical to Panel A.
    x_positions = np.arange(len(alphas))
    for policy in policies:
        g = mixed[mixed.policy == policy]
        means = g.groupby("alpha", as_index=False).signed_policy_minus_predictive_gap.mean()
        fpr = g.groupby("alpha", as_index=False).outer_fpr_gap_policy_minus_predictive.mean()
        y_means = [float(means.loc[means.alpha.eq(alpha), "signed_policy_minus_predictive_gap"].iloc[0]) for alpha in alphas]
        ax_b.plot(x_positions, y_means, color=colors[policy], linewidth=1.65,
                  marker="o", markersize=4.0, markerfacecolor="white",
                  markeredgecolor=colors[policy], markeredgewidth=0.8,
                  label=labels[policy], zorder=3)
        for i, alpha in enumerate(alphas):
            q = g[g.alpha.eq(alpha)]
            ax_b.scatter(np.full(len(q), i), q.signed_policy_minus_predictive_gap,
                         color=colors[policy], alpha=0.25, s=9, edgecolor="none", zorder=1)
            mean_fpr_gap = float(fpr.loc[fpr.alpha.eq(alpha), "outer_fpr_gap_policy_minus_predictive"].iloc[0])
            if mean_fpr_gap <= 0:
                ax_b.scatter([i], [y_means[i]], s=42, facecolor=colors[policy],
                             edgecolor="white", linewidth=0.45, zorder=4)
    ax_b.axhline(0, color=GRAY, linewidth=0.8)
    ax_b.set_xticks(x_positions, ["1%", "2%", "5%", "10%"])
    ax_b.set_xlabel("Nominal target FPR")
    ax_b.set_ylabel("Recall gap (policy − predictive)")
    ax_b.text(-0.16, 1.04, "B", transform=ax_b.transAxes, weight="bold", fontsize=10)
    clean(ax_b)

    # Panel C: fixed 5% operating point, with a predeclared false-alert cost
    # grid. This is a utility sensitivity curve, not a threshold sweep.
    c = cost[cost.alpha.eq(0.05)]
    for policy in policies:
        q = c[c.policy == policy].sort_values("lambda_false_fpr")
        ax_c.plot(q.lambda_false_fpr, q.mean_utility_gap_policy_minus_predictive, color=colors[policy], linewidth=1.8, marker="o", markersize=4.2, label=labels[policy])
    ax_c.axhline(0, color=GRAY, linewidth=0.8)
    ax_c.set_xticks([0, .25, .5, 1, 2, 4], ["0", "0.25", "0.5", "1", "2", "4"])
    ax_c.set_xlabel(r"False-alert cost $\lambda$")
    ax_c.set_ylabel("Utility gap (policy − predictive)")
    ax_c.tick_params(axis="x", labelsize=6.6, labelrotation=40)
    for tick in ax_c.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax_c.text(-0.16, 1.04, "C", transform=ax_c.transAxes, weight="bold", fontsize=10)
    clean(ax_c)

    # One shared policy legend keeps the color encoding explicit for all
    # panels while leaving the plotting area free of explanatory titles.
    handles, legend_labels = ax_c.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.045),
               ncol=3, frameon=False, fontsize=7.2, handlelength=1.8,
               columnspacing=1.6)
    marker_handles = [
        Line2D([0], [0], marker="o", color=INK, markerfacecolor=INK,
               markeredgecolor=INK, linestyle="None", markersize=4.8,
               label="mean ΔFPR ≤ 0"),
        Line2D([0], [0], marker="o", color=INK, markerfacecolor="white",
               markeredgecolor=INK, linestyle="None", markersize=4.8,
               label="mean ΔFPR > 0"),
    ]
    # This key describes marker fill only in Panel B, so keep it inside that
    # panel rather than making it look like a figure-wide legend.
    ax_b.legend(handles=marker_handles, frameon=False, fontsize=7.2,
                loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=1,
                handletextpad=0.45, labelspacing=0.35, borderaxespad=0.0)
    fig.subplots_adjust(top=0.84)

    fig.savefig(OUT / "figure_6_deployment_consequence.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "figure_6_deployment_consequence.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / 'figure_6_deployment_consequence.pdf'}")


if __name__ == "__main__":
    main()
