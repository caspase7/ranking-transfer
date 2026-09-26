# Reproducibility Code

This directory contains the small, publication-facing code layer for the
paper. It intentionally excludes exploratory runners, experiment-history
identifiers, raw trajectory banks, and local filesystem paths.

The public code covers:

- sequential warning rules and false-positive-rate calibration;
- predictive-to-policy pairwise preferences, margins, concordance, and top-1 selection;
- grouped scenario/task resampling;
- held-out selective acceptance summaries;
- score-process rearrangement controls;
- main-paper figure and deployment-consequence rendering entry points.

## Layout

`code/` contains reusable functions and command-line entry points. `config/`
contains the declared protocol. `data/` is intentionally absent from the
source snapshot: an anonymous release can provide normalized, de-identified
tables there without changing the code. `output/` is ignored and is created
by the commands below.

## Normalized inputs

The ranking-transfer runner expects the following files in the directory given
by `PFS_REPRO_DATA` (or `reproducibility/data` by default):

`predictive_losses.csv`: `fold,candidate,loss`;
`policy_metrics.csv`: `fold,policy,alpha,candidate,event_recall,feasible`.

The selective-acceptance runner expects one row per fold/policy/operating-point
and candidate-pair in `selection_records.csv`; the required columns are listed
in the script's error message and include development directions, grouped
stability frequencies, and held-out policy direction.

## Commands

```bash
python reproducibility/code/run_ranking_transfer.py \
  --input reproducibility/data \
  --output reproducibility/output/ranking_transfer

python reproducibility/code/run_selective_acceptance.py \
  --input reproducibility/data/selection_records.csv \
  --output reproducibility/output/selective_acceptance
```

The reusable policy and temporal-control functions can be imported directly.
The rendering scripts are kept separate from the estimand code and consume
machine-readable summary tables. Run them from the repository root after the
corresponding normalized tables are available:

```bash
python reproducibility/code/render_main_figures.py
python reproducibility/code/render_deployment_consequence.py
```

## Release hygiene

Before public release, replace local source paths with the normalized input
directory, remove raw trajectory banks and generated logs, and run the smoke
test with `PYTHONPATH=. python -m unittest discover tests`. The manuscript's reported numbers should be generated from
the same normalized tables that accompany the release.
