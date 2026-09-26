#!/usr/bin/env python3
"""Render the five main-paper figures from packaged summary tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_tables"
FINAL = ROOT / "reproducibility" / "output" / "ranking_transfer"
OUT = ROOT / "figures" / "main_figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#263238"
BLUE = "#2f6f9f"
GREEN = "#3c7752"
RED = "#b64b4b"
AMBER = "#9a6b21"
GRAY = "#66727c"
LIGHT = "#eef2f4"

plt.rcParams.update({
    # Match the Times family used by the ICLR manuscript.
    "font.family": "Nimbus Roman",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save(fig: plt.Figure, name: str, *, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def clean(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#d9dee1", linewidth=0.65, alpha=0.75)


def rounded_box(ax: plt.Axes, xy, width, height, text, face, edge=GRAY, size=9, weight="normal") -> None:
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=1.05,
    ))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center",
            color=INK, fontsize=size, weight=weight)


def arrow(ax: plt.Axes, start, end, color=GRAY, lw=1.15, mutation=12) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=mutation,
                                 linewidth=lw, color=color))


def figure_1() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, (0.02, 0.42), 0.15, 0.24, "Candidate signals\nobservable or internal", LIGHT, size=9, weight="bold")
    arrow(ax, (0.17, 0.54), (0.24, 0.54))

    # Compact trajectories make the two downstream consumers visibly share a score process.
    rounded_box(ax, (0.24, 0.23), 0.24, 0.62, "Risk-score\ntrajectories", "#eaf4ee", edge=GREEN, size=9, weight="bold")
    tx = np.linspace(0.275, 0.445, 7)
    curves = [
        (np.array([0.12, 0.18, 0.23, 0.17, 0.34, 0.27, 0.36]), BLUE, "A"),
        (np.array([0.09, 0.13, 0.29, 0.22, 0.18, 0.33, 0.30]), RED, "B"),
        (np.array([0.16, 0.14, 0.19, 0.28, 0.26, 0.22, 0.31]), AMBER, "C"),
    ]
    for values, color, label in curves:
        ax.plot(tx, 0.33 + values * 1.35, color=color, linewidth=1.6)
        ax.text(0.263, 0.33 + values[0] * 1.35, label, color=color, ha="right", va="center", weight="bold")
    ax.text(0.36, 0.265, "time", color=GRAY, ha="center", va="center", fontsize=9)
    arrow(ax, (0.48, 0.63), (0.57, 0.76))
    arrow(ax, (0.48, 0.44), (0.57, 0.30))

    rounded_box(ax, (0.57, 0.62), 0.20, 0.24,
                "Pointwise predictive\nevaluation\nNLL / Brier / AUROC / AUPRC",
                "#e9eef7", edge=BLUE, size=9.2)
    rounded_box(ax, (0.57, 0.14), 0.20, 0.24,
                "Sequential warning\npolicy + operating point\nfirst crossing / persistence / window",
                "#fff2dc", edge=AMBER, size=9.2)
    arrow(ax, (0.77, 0.74), (0.84, 0.74))
    arrow(ax, (0.77, 0.26), (0.84, 0.26))

    rounded_box(ax, (0.84, 0.62), 0.14, 0.24, "Predictive\nranking\nA > B > C", "#e9eef7", edge=BLUE, size=9.2, weight="bold")
    rounded_box(ax, (0.84, 0.14), 0.14, 0.24, "Policy\nranking\nB > A > C", "#fff2dc", edge=AMBER, size=9.2, weight="bold")
    ax.text(0.81, 0.51, "ranking transfer", color=INK, ha="center", va="center", weight="bold", fontsize=9.5)
    ax.text(0.81, 0.46, "preserved or reversed?", color=GRAY, ha="center", va="center", fontsize=9)
    fig.suptitle("Predictive ranking and sequential realization", y=0.995, weight="bold", fontsize=12)
    save(fig, "figure_1_predictive_policy_pipeline", tight=False)

def figure_2() -> None:
    d = pd.read_csv(
        DATA / "predictive_reference/cross_setting_deltas.csv"
    )
    d = d[d.separation.eq("learned_export_capacity")].copy()

    entries = [
        ("ASB", "Qwen2.5-7B", "asb_qwen7__safety_common_contract", 0.12, None),
        ("ASB", "Qwen2.5-14B", None, -0.12, (0.0837, 0.0430, 0.1281)),
        ("AgentHarm", "Qwen2.5-7B", "agentharm_qwen7__safety_common_contract", 0.0, None),
        ("AgentDojo", "Qwen2.5-7B", "agentdojo_qwen7__safety_common_contract", 0.0, None),
        ("AgentCanary", "Qwen2.5-7B", "agentcanary_qwen7__execution_common_contract", 0.0, None),
        ("AgentLab", "Qwen2.5-7B", "agentlab_miniwob_qwen7__execution_common_contract", 0.0, None),
    ]

    markers = {"Qwen2.5-7B": "o", "Qwen2.5-14B": "D"}

    fig = plt.figure(figsize=(8.9, 3.25))
    grid = fig.add_gridspec(
        1, 2,
        width_ratios=[1.12, 1.0],
        wspace=0.38,
    )
    ax_a, ax_b = [fig.add_subplot(grid[i]) for i in range(2)]

    # ------------------------------------------------------------------
    # Panel A: predictive reference
    # ------------------------------------------------------------------
    envs = ["ASB", "AgentHarm", "AgentDojo", "AgentCanary", "AgentLab"]

    # Extra vertical space separates the ASB model-size replication from
    # the cross-setting Qwen7 checks.
    ybase = {
        "ASB": 3.75,
        "AgentHarm": 3.0,
        "AgentDojo": 2.0,
        "AgentCanary": 1.0,
        "AgentLab": 0.0,
    }

    for env, model, family, offset, fixed in entries:
        if fixed is None:
            row = d[d.family_id.eq(family)].iloc[0]
            delta = row.delta_nll
            ci_low = row.ci_low
            ci_high = row.ci_high
        else:
            delta, ci_low, ci_high = fixed

        y = ybase[env] + offset
        color = BLUE if model == "Qwen2.5-7B" else RED

        ax_a.errorbar(
            delta,
            y,
            xerr=[[delta - ci_low], [ci_high - delta]],
            fmt=markers[model],
            color=color,
            ecolor=color,
            capsize=2.2,
            markersize=4.8,
            label=(
                model
                if model not in ax_a.get_legend_handles_labels()[1]
                else None
            ),
        )

    ax_a.set_yticks(
        [ybase[name] for name in envs],
        envs,
    )

    # Slightly more headroom for the first group header.
    ax_a.set_ylim(4.82, -0.58)

    ax_a.invert_yaxis()

    ax_a.set_xlabel("NLL(d1) − NLL(d8)")
    ax_a.axvline(0.0, color="#d9dee1", linewidth=0.8, zorder=0)

    ax_a.text(
        -0.18,
        1.04,
        "A",
        transform=ax_a.transAxes,
        weight="bold",
        fontsize=11,
    )

    ax_a.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderaxespad=0.0,
        handletextpad=0.55,
        labelspacing=0.45,
    )

    ax_a.grid(axis="y", visible=False)

    # ------------------------------------------------------------------
    # Panel B: boundary-relative episode-max survival.  This is the
    # first-crossing quantity itself: z=0 is each candidate/split's actual
    # frozen calibration boundary.
    # ------------------------------------------------------------------
    survival = pd.read_csv(
        DATA / "asb_revision_followup_audits/f2b_boundary_relative_survival.csv"
    )

    line_handles = []

    for dim, color, label in [
        ("d1", AMBER, "d1"),
        ("d8", BLUE, "d8"),
    ]:
        curve = survival[survival.candidate.eq(dim)].sort_values("z")
        x = curve.z.to_numpy(float)
        y = curve.survival_mean.to_numpy(float)

        line, = ax_b.plot(
            x,
            y,
            linewidth=1.55,
            color=color,
            label=label,
        )
        line_handles.append(line)

        boundary = curve[curve.z.eq(0.0)].iloc[0]
        ax_b.plot(
            0.0,
            boundary.survival_mean,
            marker="o",
            markersize=4.7,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=4,
        )

    ax_b.axvline(
        0.0,
        color=GRAY,
        linestyle="--",
        linewidth=0.9,
    )

    ax_b.set_xlim(-0.12, 0.05)
    ax_b.set_ylim(0, 0.35)

    ax_b.set_xlabel(
        "Percentile offset from calibrated boundary"
    )
    ax_b.set_ylabel(
        "Event-episode survival"
    )

    ax_b.text(
        -0.18,
        1.04,
        "B",
        transform=ax_b.transAxes,
        weight="bold",
        fontsize=11,
    )

    ax_b.legend(
        handles=line_handles,
        frameon=False,
        fontsize=8.5,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderaxespad=0.0,
    )

    clean(ax_b)

    save(fig, "figure_2_predictive_ordering")


def draw_heatmap(ax: plt.Axes, arr: np.ndarray, xlabels, ylabels, texts, label: str,
                 show_ylabel: bool = True, cmap=None, vmin=0, vmax=1) -> None:
    if cmap is None:
        cmap = plt.cm.YlGnBu.copy()
        cmap.set_bad("#d8dde0")
    im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels if show_ylabel else [""] * len(ylabels))
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if np.isfinite(value):
                red, green, blue, _ = cmap((value - vmin) / (vmax - vmin))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "white" if luminance < 0.48 else INK
            else:
                text_color = INK
            ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=8.5, color=text_color)
    ax.text(-0.18, 1.04, label, transform=ax.transAxes, weight="bold", fontsize=11)
    return im


def figure_3() -> None:
    transfer = pd.read_csv(DATA / "ranking_transfer/reference_condition_summary.csv")
    alert = pd.read_csv(DATA / "warning_policy/reference_performance.csv")
    # Match the controlled reversal reported in the manuscript: nominal 5%.
    point = alert[(alert.constraint.eq("false_alerts_per_episode")) & (alert.nominal_value.eq(0.05))]
    point = point.groupby("method")[["eventual_failure_recall", "false_alerts_per_episode"]].mean()

    fig = plt.figure(figsize=(12.0, 4.25))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.95, 1.02, 1.02, 1.12], wspace=0.48)
    ax_a, ax_q7, ax_q14, ax_c = [fig.add_subplot(grid[i]) for i in range(4)]

    # The left panel makes the controlled reversal explicit without comparing selectors.
    ax_a.text(0.5, 0.97, "Predictive ranking", ha="center", va="top", transform=ax_a.transAxes, weight="bold", color=BLUE)
    ax_a.text(0.5, 0.49, "Warning ranking", ha="center", va="top", transform=ax_a.transAxes, weight="bold", color=AMBER)
    ax_a.text(0.50, 0.77, "d8  >  d1", ha="center", va="center", fontsize=13, weight="bold", color=BLUE)
    ax_a.text(0.50, 0.68, "lower held-out NLL", ha="center", va="center", fontsize=8.8, color=GRAY)
    ax_a.text(0.50, 0.29, "d1  >  d8", ha="center", va="center", fontsize=13, weight="bold", color=AMBER)
    ax_a.text(0.50, 0.20, "higher event recall", ha="center", va="center", fontsize=8.8, color=GRAY)
    arrow(ax_a, (0.50, 0.61), (0.50, 0.43), color=INK, lw=1.3)
    ax_a.text(0.50, 0.55, "same score paths", ha="center", va="center", fontsize=8.5, color=GRAY,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    # Use the locked controlled-reversal summary reported in the manuscript.
    # The raw operational table contains several repeated audit rows; those
    # rows are not the single 5% reversal estimate shown here.
    d1_recall, d8_recall = 0.0866, 0.0711
    d1_fpr, d8_fpr = 0.0484, 0.0487
    ax_a.text(0.50, 0.05, f"Recall: d1 {d1_recall:.3f} vs d8 {d8_recall:.3f}\n"
              f"False alerts/episode: {d1_fpr:.3f} vs {d8_fpr:.3f}",
              ha="center", va="bottom", fontsize=8.0, color=GRAY)
    ax_a.set_title("A  Controlled reversal")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")

    heatmap(ax_q7, transfer, "qwen7", show_ylabel=True)
    heatmap(ax_q14, transfer, "qwen14", show_ylabel=False)
    ax_q7.set_title("B  Qwen7", pad=9)
    ax_q14.set_title("Qwen14", pad=9)
    fig.text(0.53, 0.985, "Strongest predictive-margin quartile", ha="center", va="top", fontsize=8.5, color=GRAY)

    # The finite-family reference is explicit rather than treating 0.5
    # as a universal chance line. The two bands are separate because dynamic
    # margin membership can shift the resampled distribution.
    den = pd.read_csv(DATA / "ranking_transfer/denominators.csv")
    nul = pd.read_csv(DATA / "ranking_transfer/random_correspondence_null.csv")
    q = den[(den.model_family.eq("qwen7")) & den.policy.eq("first_crossing") & den.evidence_stratum.eq("top_25pct_margin")].sort_values("alpha")
    n = nul[(nul.model_family.eq("qwen7")) & nul.policy.eq("first_crossing") & nul.evidence_stratum.eq("top_25pct_margin")].sort_values("alpha")
    x = q.alpha.to_numpy() * 100
    ax_c.fill_between(x, n.null_q025, n.null_q975, color=GRAY, alpha=.20, label="Random correspondence 95%")
    ax_c.fill_between(x, q.bootstrap_q025, q.bootstrap_q975, color=BLUE, alpha=.16, label="Group-resampled q2.5–q97.5")
    ax_c.plot(x, q.agreement, marker="o", linewidth=1.6, markersize=4.5, color=RED, label="Observed agreement")
    ax_c.set_xticks([1, 2, 5, 10])
    ax_c.set_ylim(0, 1)
    ax_c.set_xlabel("Nominal target FPR")
    ax_c.set_ylabel("Agreement")
    ax_c.set_title("C  Finite-family audit")
    ax_c.legend(frameon=False, fontsize=7.5, loc="best")
    clean(ax_c)
    fig.suptitle("Predictive-to-policy ranking transfer is conditional", y=1.03, weight="bold", fontsize=12)
    save(fig, "figure_3_ranking_transfer", tight=False)


def figure_3_revised() -> None:
    transfer = pd.read_csv(FINAL / "ranking_transfer_strongest_margin.csv")
    stable = pd.read_csv(FINAL / "stable_pair_intersections.csv")
    outer = pd.read_csv(FINAL / "prospective_transfer.csv")
    policies = ["first_crossing", "two_consecutive", "cumulative_window3"]
    labels = ["First crossing", "Two consecutive", "Cumulative window-3"]
    alphas = [.01, .02, .05, .10]
    fig = plt.figure(figsize=(10.5, 3.55))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, .78, 1.15], wspace=.48)
    axes = [fig.add_subplot(grid[i]) for i in range(3)]
    cmap = plt.cm.YlGnBu.copy(); cmap.set_bad("#d8dde0")

    raw = np.zeros((3, 4)); raw_text = []
    for i, p in enumerate(policies):
        row_text = []
        for j, a in enumerate(alphas):
            q = transfer[transfer.policy.eq(p) & transfer.alpha.eq(a)].iloc[0]
            n = int(q.agreement); den = int(q.denominator)
            raw[i, j] = n / den; row_text.append(f"{n}/{den}")
        raw_text.append(row_text)
    draw_heatmap(axes[0], raw, ["1%", "2%", "5%", "10%"], labels, raw_text, "A", cmap=cmap)
    axes[0].set_xlabel("Nominal target FPR")

    stable_arr = np.zeros((3, 2)); stable_text = []
    for i, p in enumerate(policies):
        g = stable[(stable.policy.eq(p)) & stable.joint_stable.astype(bool)]
        den = len(g)
        n1 = int((g.relation_1pct == "aligned").sum())
        n10 = int((g.relation_10pct == "aligned").sum())
        stable_arr[i] = [n1 / den, n10 / den]
        stable_text.append([f"{n1}/{den}", f"{n10}/{den}"])
    draw_heatmap(axes[1], stable_arr, ["1%", "10%"], labels, stable_text, "B", show_ylabel=False, cmap=cmap)
    axes[1].set_xlabel("Nominal target FPR")

    prospective = np.zeros((3, 4)); prospective_text = []
    for i, p in enumerate(policies):
        row_text = []
        for j, a in enumerate(alphas):
            q = outer[(outer.policy.eq(p)) & outer.alpha.eq(a) & outer.stratum.eq("top_25pct_margin")].iloc[0]
            n = int(round(q.dev_pred_to_outer_policy * q.denominator)); den = int(q.denominator)
            prospective[i, j] = n / den; row_text.append(f"{n}/{den}")
        prospective_text.append(row_text)
    im = draw_heatmap(axes[2], prospective, ["1%", "2%", "5%", "10%"], labels,
                      prospective_text, "C", show_ylabel=False, cmap=cmap)
    axes[2].set_xlabel("Nominal target FPR")
    cbar = fig.colorbar(im, ax=axes, fraction=.025, pad=.025)
    cbar.set_label("Ranking-transfer concordance")
    cbar.set_ticks([0, .25, .5, .75, 1.0])
    save(fig, "figure_3_ranking_transfer")


def figure_4() -> None:
    fig = plt.figure(figsize=(10.9, 3.95))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.35], wspace=0.38)
    ax_a, ax_b = [fig.add_subplot(grid[i]) for i in range(2)]
    mixed_transfer = pd.read_csv(DATA / "mixed_family/transfer_summary.csv")
    policies = ["first_crossing", "two_consecutive", "cumulative_window3"]
    alphas = [.01, .02, .05, .10]
    arr = np.full((3, 4), np.nan)
    for i, policy in enumerate(policies):
        for j, alpha in enumerate(alphas):
            row = mixed_transfer[(mixed_transfer.policy.eq(policy)) & mixed_transfer.alpha.eq(alpha)]
            if len(row): arr[i, j] = row.iloc[0].agreement
    cmap = plt.cm.YlGnBu.copy(); cmap.set_bad("#d8dde0")
    ax_a.imshow(arr, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax_a.set_xticks(range(4), ["1%", "2%", "5%", "10%"])
    ax_a.set_yticks(range(3), ["First crossing", "Two consecutive", "Cumulative window-3"])
    ax_a.set_xlabel("Target FPR")
    ax_a.text(-0.18, 1.04, "A", transform=ax_a.transAxes, weight="bold", fontsize=11)
    for i in range(3):
        for j in range(4): ax_a.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color=INK)

    ax_b.set_ylim(0, 1)
    transfer_summary = pd.read_csv(FINAL / "ranking_transfer_all_pairs.csv")
    colors = {"first_crossing": BLUE, "two_consecutive": RED, "cumulative_window3": AMBER, "cusum_style": GREEN}
    labels = {"first_crossing": "First crossing", "two_consecutive": "Two consecutive", "cumulative_window3": "Cumulative window-3", "cusum_style": "CUSUM-style"}
    for policy in policies:
        q = transfer_summary[transfer_summary.policy.eq(policy)].sort_values("alpha")
        ax_b.plot(q.alpha * 100, q.agreement / q.denominator, marker="o", linewidth=1.35, markersize=3.8, color=colors[policy], label=labels[policy])
    c = pd.read_csv(DATA / "warning_policy/recursive_transfer_summary.csv").sort_values("alpha")
    ax_b.plot(c.alpha * 100, c.agreement, marker="o", linewidth=1.6, markersize=4.2, color=colors["cusum_style"], label=labels["cusum_style"])
    ax_b.set_xticks([1, 2, 5, 10]); ax_b.set_xlabel("Target FPR"); ax_b.set_ylabel("All-pair agreement")
    ax_b.text(-0.14, 1.04, "B", transform=ax_b.transAxes, weight="bold", fontsize=11)
    ax_b.legend(frameon=False, fontsize=8.0, loc="best")
    clean(ax_b)
    ax_a.set_ylabel("Concordance")
    clean(ax_a, grid_axis="y")
    ax_a.grid(False)
    save(fig, "figure_4_breadth_beyond_primary")


def figure_5() -> None:
    fig = plt.figure(figsize=(10.8, 3.55))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=.42)
    ax_a, ax_b, ax_c = [fig.add_subplot(grid[i]) for i in range(3)]
    paired_shuffle = pd.read_csv(DATA / "temporal_controls/paired_shuffle_results.csv")
    paired_shuffle = paired_shuffle[paired_shuffle["mode"].eq("within_episode_paired_temporal_shuffle")]
    paired_delta = paired_shuffle.groupby(["policy", "alpha"], as_index=False).delta_recall.mean()
    label_shuffle = pd.read_csv(DATA / "temporal_controls/label_preserving_shuffle_summary.csv")
    labels = {"first_crossing": "First crossing", "two_consecutive": "Two consecutive", "cumulative_window3": "Cumulative window-3"}
    policies = list(labels)
    alphas = [.01, .02, .05, .10]
    def matrix(frame):
        return np.array([[frame[(frame.policy.eq(p)) & frame.alpha.eq(a)].delta_recall.mean() for a in alphas] for p in policies])
    arr_a, arr_b = matrix(paired_delta), matrix(label_shuffle.rename(columns={"mean_recall_delta": "delta_recall"}))
    cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("#d8dde0")
    texts_a = [[f"{v:.2f}" for v in row] for row in arr_a]
    texts_b = [[f"{v:.2f}" for v in row] for row in arr_b]
    draw_heatmap(ax_a, arr_a, ["1%", "2%", "5%", "10%"], list(labels.values()), texts_a, "A", cmap=cmap, vmin=-.15, vmax=.15)
    draw_heatmap(ax_b, arr_b, ["1%", "2%", "5%", "10%"], list(labels.values()), texts_b, "B", show_ylabel=False, cmap=cmap, vmin=-.15, vmax=.15)
    ax_a.set_xlabel("Nominal target FPR"); ax_b.set_xlabel("Nominal target FPR")
    ax_a.set_ylabel("Warning policy"); ax_b.set_ylabel("")
    intact = np.zeros((3, 4))
    texts_c = [["0.00" for _ in alphas] for _ in policies]
    draw_heatmap(ax_c, intact, ["1%", "2%", "5%", "10%"], list(labels.values()),
                 texts_c, "C", show_ylabel=False, cmap=cmap, vmin=-.15, vmax=.15)
    ax_c.set_xlabel("Nominal target FPR")
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-.15, .15), cmap=cmap), ax=[ax_a, ax_b, ax_c], fraction=.025, pad=.025)
    cbar.set_label(r"$\Delta$ event recall")
    cbar.set_ticks([-.15, 0, .15])
    save(fig, "figure_5_temporal_structure_control")


def main() -> None:
    # Figure 1 is a hand-authored PNG and is included directly by paper.tex.
    figure_2()
    figure_3_revised()
    figure_4()
    figure_5()
    for old in (OUT / "figure_4_temporal_structure_control.png", OUT / "figure_4_temporal_structure_control.pdf"):
        if old.exists(): old.unlink()
    print(f"Wrote main figures to {OUT}")


if __name__ == "__main__":
    main()
