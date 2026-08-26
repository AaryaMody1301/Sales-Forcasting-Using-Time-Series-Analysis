# Canonical Forecasting Architecture

## Core rules

1. Observed timestamps and observed targets only.
2. Explicit time frequency and transaction aggregation.
3. Missing periods are never silently filled by the data-contract layer.
4. Backtesting owns train/test separation; individual models receive training data only.
5. Every challenger is evaluated against a deterministic naive baseline on the same folds.
6. ML target-history features must stop at `t-1` when predicting `t`.
7. Multi-step autoregressive ML forecasts are recursive and may use earlier predictions, not hidden future actuals.
8. Generated metrics are benchmark evidence only when they come from the canonical evaluator.
9. A completed experiment is trusted only through its manifest and checksum-verified artifacts.
10. The dashboard consumes explicit manifest paths; it never infers model identity from filenames.

## Canonical package

```text
src/sales_forecasting/
├── data/
│   ├── schema.py
│   ├── prepare.py
│   └── catalog.py
├── evaluation/
│   ├── metrics.py
│   ├── backtesting.py
│   └── leaderboard.py
├── features/
│   └── lags.py
├── models/
│   ├── base.py
│   ├── naive.py
│   ├── statistical.py
│   └── ml.py
├── artifacts/
│   ├── fingerprints.py
│   ├── manifest.py
│   └── store.py
└── dashboard/
    ├── data.py
    └── app.py
```

## Evaluation boundary

The evaluator slices the complete series at a forecast origin and constructs a `PreparedSeries` containing only the training portion. Models never receive the held-out target window during `fit()`.

For ML row `t`, lag and rolling values are derived from `series[:t]`. Calendar features can describe timestamp `t` because the timestamp is known before its target is observed.

At inference, autoregressive ML forecasts are recursive:

```text
history -> predict t+1
history + prediction(t+1) -> predict t+2
...
```

## Run identity

`record_experiment()` fingerprints two independent inputs:

1. **Dataset fingerprint** - SHA-256 over the prepared series schema, timestamp sequence, and target values.
2. **Configuration fingerprint** - SHA-256 over evaluation settings, model implementation/configuration metadata, package version, and code revision.

The run ID is derived from those fingerprints:

```text
<dataset-name>-<dataset-hash-prefix>-<config-hash-prefix>
```

Given the same prepared data, effective configuration, package version, and code revision, the run ID is stable.

## Manifest contract

Each completed run contains `manifest.json` with schema version 1. It records:

- run ID and completion timestamp
- package/Python/dependency versions
- source revision
- dataset schema, range, row counts, and SHA-256 fingerprint
- evaluation horizon, step, train size, baseline, and config fingerprint
- model implementation/configuration, ranks, and aggregate metrics
- explicit paths to leaderboard/fold/forecast artifacts
- SHA-256 checksum and byte size for every referenced CSV

The manifest is written only after the other artifacts are staged successfully.

## Artifact write boundary

Runs are assembled in a temporary sibling directory first. Only after all CSVs and the manifest have been written successfully is the staging directory moved into the deterministic final run path. If the same run is repeated, the prior completed directory is temporarily backed up and restored if replacement fails.

Generated run directories live under `artifacts/` and are ignored by Git.

## Dashboard trust boundary

The Streamlit dashboard discovers only:

```text
artifacts/runs/*/manifest.json
```

Before reading a CSV, the dashboard:

1. validates the manifest schema/version/status;
2. rejects absolute or path-traversal artifact paths;
3. verifies the file SHA-256 against the manifest;
4. only then loads the CSV.

A missing, modified, or malformed artifact is displayed as an integrity error, not replaced with synthetic fallback values.

## CI

GitHub Actions installs the canonical package (including the dashboard extra), runs the test suite, and imports the dashboard entry point on supported Python versions. This provides a repository-level quality gate for future phases.

## Future work

Hyperparameter tuning must be nested inside chronological training/validation folds. Final test windows cannot choose hyperparameters. Known-future regressors and Prophet require an explicit future-covariate contract. Ensembles must derive weights from validation data rather than final test results.
