#!/usr/bin/env python3
"""Apply score-process rearrangement controls to a normalized score table.

Input columns: ``episode``, ``position``, ``score``, ``target``, ``event`` and
optionally ``path_length``. The output contains one CSV per control. Warning
metrics are intentionally evaluated by downstream analysis, keeping this
script focused on the transformations themselves.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from temporal_controls import intact_path_reassignment, label_preserving_score_shuffle, paired_record_permutation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    required = {"episode", "position", "score", "target", "event"}
    if not required <= set(frame):
        raise ValueError(f"input is missing columns: {sorted(required - set(frame))}")
    frame = frame.sort_values(["episode", "position"]).reset_index(drop=True)
    paired_record_permutation(frame, "episode", args.seed).to_csv(args.output / "paired_record_permutation.csv", index=False)
    label_preserving_score_shuffle(frame, "episode", "score", "target", args.seed + 1).to_csv(
        args.output / "label_preserving_score_shuffle.csv", index=False,
    )
    if "path_length" in frame:
        intact_path_reassignment(frame, "episode", "path_length", "event", args.seed + 2).to_csv(
            args.output / "intact_path_reassignment.csv", index=False,
        )


if __name__ == "__main__":
    main()
