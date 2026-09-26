#!/usr/bin/env python3
"""Compute ranking-transfer tables from normalized release inputs.

Input files:
``predictive_losses.csv``: ``fold,candidate,loss``;
``policy_metrics.csv``: ``fold,policy,alpha,candidate,event_recall,feasible``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ranking_transfer import concordance, pairwise_preferences, policy_preference, strongest_margin_pairs, top1_from_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Directory containing normalized CSV inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    losses = pd.read_csv(args.input / "predictive_losses.csv")
    metrics = pd.read_csv(args.input / "policy_metrics.csv")
    required_loss = {"fold", "candidate", "loss"}
    required_metrics = {"fold", "policy", "alpha", "candidate", "event_recall", "feasible"}
    if not required_loss <= set(losses) or not required_metrics <= set(metrics):
        raise ValueError("input columns do not match the documented schema")

    predictive_rows = []
    transfer_rows = []
    top1_rows = []
    for fold, group in losses.groupby("fold", sort=True):
        loss_map = {int(row.candidate): float(row.loss) for row in group.itertuples()}
        pairs = pairwise_preferences(loss_map, fold=int(fold))
        predictive_rows.append(pairs)
        predictive_top1 = top1_from_losses(loss_map)
        for (policy, alpha), condition in metrics[metrics.fold.eq(fold)].groupby(["policy", "alpha"], sort=True):
            rows = {int(row.candidate): {"event_recall": row.event_recall, "feasible": row.feasible}
                    for row in condition.itertuples()}
            policy_top1 = min(rows, key=lambda candidate: (
                not bool(rows[candidate]["feasible"]), -float(rows[candidate]["event_recall"]), candidate,
            ))
            pair_values = []
            for pair in pairs.itertuples(index=False):
                winner = policy_preference(rows[int(pair.left_candidate)], rows[int(pair.right_candidate)],
                                           int(pair.left_candidate), int(pair.right_candidate))
                pair_values.append({**pair._asdict(), "policy": policy, "alpha": alpha,
                                    "policy_preference": winner,
                                    "agreement": int(winner == int(pair.predictive_preference))})
            transfer_rows.extend(pair_values)
            top1_rows.append({"fold": fold, "policy": policy, "alpha": alpha,
                              "predictive_top1": predictive_top1, "policy_top1": policy_top1,
                              "agreement": int(predictive_top1 == policy_top1)})

    predictive_pairs = pd.concat(predictive_rows, ignore_index=True)
    pairs = pd.DataFrame(transfer_rows)
    pairs.to_csv(args.output / "ranking_transfer_pairs.csv", index=False)
    predictive_pairs.to_csv(args.output / "predictive_pairs.csv", index=False)
    top1 = pd.DataFrame(top1_rows)
    top1.to_csv(args.output / "top1_selection.csv", index=False)

    pairs = pairs.merge(
        predictive_pairs[["fold", "left_candidate", "right_candidate", "absolute_margin"]],
        on=["fold", "left_candidate", "right_candidate"], how="left", suffixes=("", "_predictive"),
    )
    pairs["top_25pct"] = False
    selected_indices = strongest_margin_pairs(predictive_pairs, 3).index
    selected_keys = set(map(tuple, predictive_pairs.loc[selected_indices, ["fold", "left_candidate", "right_candidate"]].to_numpy()))
    pairs["top_25pct"] = pairs.apply(
        lambda row: (row.fold, row.left_candidate, row.right_candidate) in selected_keys, axis=1,
    )
    summary_rows = []
    for (policy, alpha), group in pairs.groupby(["policy", "alpha"], sort=True):
        for stratum, selected in (("all_pairs", group), ("top_25pct", group[group.top_25pct])):
            result = concordance(selected.predictive_preference, selected.policy_preference)
            summary_rows.append({"policy": policy, "alpha": alpha, "stratum": stratum, **result})
    pd.DataFrame(summary_rows).to_csv(args.output / "ranking_transfer_summary.csv", index=False)


if __name__ == "__main__":
    main()
