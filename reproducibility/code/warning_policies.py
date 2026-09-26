"""Sequential warning rules, threshold calibration, and operating metrics.

The functions operate on episode-level score sequences. Calibration uses only
eligible non-event episodes supplied by the caller; evaluation is performed on
a separate episode table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


POLICIES = ("first_crossing", "two_consecutive", "cumulative_window3")


def policy_statistic(scores: Sequence[float], policy: str) -> float:
    values = np.asarray(scores, dtype=float)
    if policy == "first_crossing":
        return float(values.max()) if values.size else -np.inf
    if policy == "two_consecutive":
        return float(np.minimum(values[:-1], values[1:]).max()) if values.size > 1 else -np.inf
    if policy == "cumulative_window3":
        return float(np.convolve(values, np.ones(3), mode="valid").max()) if values.size >= 3 else -np.inf
    raise ValueError(f"unknown policy: {policy}")


def episode_statistics(episodes: Iterable[Sequence[float]], policy: str) -> np.ndarray:
    """Return one policy statistic for each episode in input order."""
    return np.asarray([policy_statistic(scores, policy) for scores in episodes], dtype=float)


def calibrate_threshold(non_event_statistics: Sequence[float], target_fpr: float) -> float:
    """Choose the highest observed threshold with empirical FPR <= target."""
    values = np.asarray(non_event_statistics, dtype=float)
    if values.size == 0:
        return np.inf
    allowed = int(np.floor(target_fpr * values.size + 1e-12))
    for threshold in np.sort(np.unique(values))[::-1]:
        if int(np.sum(values >= threshold)) <= allowed:
            return float(threshold)
    return np.inf


@dataclass(frozen=True)
class OperatingMetrics:
    event_recall: float
    false_positive_rate: float
    event_count: int
    non_event_count: int


def evaluate_threshold(statistics: Sequence[float], event_labels: Sequence[int], threshold: float) -> OperatingMetrics:
    values = np.asarray(statistics, dtype=float)
    events = np.asarray(event_labels, dtype=int).astype(bool)
    alerts = values >= threshold
    event_n = int(events.sum())
    non_event_n = int((~events).sum())
    return OperatingMetrics(
        event_recall=float((alerts & events).sum() / max(event_n, 1)),
        false_positive_rate=float((alerts & ~events).sum() / max(non_event_n, 1)),
        event_count=event_n,
        non_event_count=non_event_n,
    )


def calibrate_and_evaluate(
    calibration_statistics: Sequence[float],
    evaluation_statistics: Sequence[float],
    evaluation_events: Sequence[int],
    target_fpr: float,
) -> tuple[float, OperatingMetrics]:
    threshold = calibrate_threshold(calibration_statistics, target_fpr)
    return threshold, evaluate_threshold(evaluation_statistics, evaluation_events, threshold)
