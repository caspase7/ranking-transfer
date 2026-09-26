"""Finite-family predictive-to-policy ranking-transfer calculations."""
from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def predictive_preference(left_loss: float, right_loss: float, left_id: int, right_id: int) -> int:
    """Return the preferred candidate; lower loss wins and smaller id breaks ties."""
    if left_loss < right_loss:
        return left_id
    if right_loss < left_loss:
        return right_id
    return min(left_id, right_id)


def policy_preference(
    left: Mapping[str, float | bool], right: Mapping[str, float | bool],
    left_id: int, right_id: int, feasibility_first: bool = True,
) -> int:
    """Return the policy winner using feasibility, recall, then id ordering."""
    def key(row: Mapping[str, float | bool], candidate_id: int) -> tuple[bool, float, int]:
        feasible = bool(row["feasible"]) if feasibility_first else True
        return (not feasible, -float(row["event_recall"]), candidate_id)
    return left_id if key(left, left_id) < key(right, right_id) else right_id


def pairwise_preferences(losses: Mapping[int, float], fold: int | None = None) -> pd.DataFrame:
    rows = []
    for left, right in combinations(sorted(losses), 2):
        winner = predictive_preference(losses[left], losses[right], left, right)
        rows.append({"fold": fold, "left_candidate": left, "right_candidate": right,
                     "left_loss": losses[left], "right_loss": losses[right],
                     "signed_margin": losses[right] - losses[left],
                     "predictive_preference": winner,
                     "absolute_margin": abs(losses[right] - losses[left])})
    return pd.DataFrame(rows)


def strongest_margin_pairs(pairs: pd.DataFrame, pairs_per_fold: int = 3) -> pd.DataFrame:
    """Select a fixed number of largest-margin pairs in each fold."""
    ordered = pairs.sort_values(
        ["fold", "absolute_margin", "left_candidate", "right_candidate"],
        ascending=[True, False, True, True], kind="mergesort",
    )
    return ordered.groupby("fold", sort=False, group_keys=False).head(pairs_per_fold).copy()


def concordance(predictive: Sequence[int], policy: Sequence[int]) -> dict[str, int | float]:
    pred = np.asarray(predictive, dtype=int)
    pol = np.asarray(policy, dtype=int)
    if pred.shape != pol.shape:
        raise ValueError("predictive and policy preference arrays must have equal shape")
    agreement = int(np.sum(pred == pol))
    return {"agreement": agreement, "disagreement": int(len(pred) - agreement),
            "denominator": int(len(pred)), "concordance": float(agreement / len(pred)) if len(pred) else np.nan}


def grouped_resample(groups: Sequence[str], seed: int, replicate: int) -> np.ndarray:
    """Sample scenario/task groups with replacement, preserving group multiplicity."""
    values = np.asarray(sorted(set(groups)), dtype=object)
    rng = np.random.default_rng(seed + replicate)
    return rng.choice(values, size=len(values), replace=True)


def top1_from_losses(losses: Mapping[int, float]) -> int:
    return min(losses, key=lambda candidate: (float(losses[candidate]), int(candidate)))
