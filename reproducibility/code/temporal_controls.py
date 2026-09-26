"""Score-process rearrangement controls used for the structural analysis."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def paired_record_permutation(frame: pd.DataFrame, episode_col: str, seed: int) -> pd.DataFrame:
    """Permute score/target records jointly within each episode."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    pieces = []
    for _, episode in out.groupby(episode_col, sort=False):
        pieces.append(episode.iloc[rng.permutation(len(episode))].copy())
    return pd.concat(pieces, ignore_index=True)


def label_preserving_score_shuffle(
    frame: pd.DataFrame, episode_col: str, score_col: str, label_col: str, seed: int,
) -> pd.DataFrame:
    """Shuffle scores within an episode and target label, preserving labels."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for _, indices in out.groupby(episode_col, sort=False).groups.items():
        for _, label_indices in out.loc[indices].groupby(label_col, sort=False).groups.items():
            positions = np.asarray(label_indices)
            if len(positions) > 1:
                out.loc[positions, score_col] = out.loc[rng.permutation(positions), score_col].to_numpy()
    return out


def intact_path_reassignment(
    frame: pd.DataFrame, episode_col: str, path_length_col: str, status_col: str, seed: int,
) -> pd.DataFrame:
    """Exchange complete score paths only within status and path-length strata."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    groups = frame.groupby([status_col, path_length_col], sort=False)
    for _, members in groups:
        episodes = list(members[episode_col].drop_duplicates())
        shuffled = episodes.copy()
        rng.shuffle(shuffled)
        mapping = dict(zip(episodes, shuffled))
        for source, destination in mapping.items():
            source_rows = frame[frame[episode_col].eq(source)].sort_values(path_length_col)
            destination_rows = out[episode_col].eq(destination)
            out.loc[destination_rows, "score"] = source_rows["score"].to_numpy()
    return out
