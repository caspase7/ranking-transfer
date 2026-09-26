#!/usr/bin/env python3
"""Compute matched-coverage selective acceptance summaries.

The input table must contain one row per fold/policy/operating-point/pair with
development predictive and policy directions and their grouped stability
frequencies. The script never uses held-out outcomes to form the accepted set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    required = {"fold", "policy", "alpha", "left_candidate", "right_candidate",
                "predictive_preference", "policy_preference", "predictive_stability",
                "policy_stability", "heldout_policy_preference"}
    if not required <= set(data):
        raise ValueError(f"input is missing columns: {sorted(required - set(data))}")

    rows = []
    for (policy, alpha), group in data.groupby(["policy", "alpha"], sort=True):
        accepted = group[
            group.predictive_preference.ne(0)
            & group.policy_preference.ne(0)
            & group.predictive_preference.eq(group.policy_preference)
            & group.predictive_stability.ge(args.threshold)
            & group.policy_stability.ge(args.threshold)
        ]
        for row in group.itertuples(index=False):
            pair = (row.fold, row.left_candidate, row.right_candidate)
            keep = pair in set(zip(accepted.fold, accepted.left_candidate, accepted.right_candidate))
            decisive = int(row.heldout_policy_preference != 0)
            direction = int(row.predictive_preference)
            rows.append({"fold": row.fold, "policy": policy, "alpha": alpha,
                         "left_candidate": row.left_candidate, "right_candidate": row.right_candidate,
                         "accepted": int(keep), "heldout_decisive": decisive,
                         "correct": int(keep and decisive and direction == row.heldout_policy_preference),
                         "wrong": int(keep and decisive and direction != row.heldout_policy_preference),
                         "heldout_tie": int(not decisive)})
    records = pd.DataFrame(rows)
    summary = records.groupby(["policy", "alpha"], as_index=False).agg(
        total_pairs=("accepted", "size"), accepted_pairs=("accepted", "sum"),
        correct=("correct", "sum"), wrong=("wrong", "sum"), heldout_ties=("heldout_tie", "sum"),
    )
    summary["decisive"] = summary.correct + summary.wrong
    summary["coverage"] = summary.accepted_pairs / summary.total_pairs
    summary["accuracy"] = summary.correct / summary.decisive.where(summary.decisive > 0)
    records.to_csv(args.output / "selective_acceptance_records.csv", index=False)
    summary.to_csv(args.output / "selective_acceptance_summary.csv", index=False)


if __name__ == "__main__":
    main()
