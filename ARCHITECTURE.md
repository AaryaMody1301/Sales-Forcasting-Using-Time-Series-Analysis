# Canonical Forecasting Architecture

## Core rules

1. Observed timestamps and observed targets only.
2. Explicit time frequency and transaction aggregation.
3. The data-contract layer never silently imputes gaps.
4. Backtesting owns train/test separation; models receive training data only.
5. Missing-value handling is applied independently to training folds only.
6. Held-out target values are never imputed for evaluation.
7. Every challenger is compared against a deterministic naive baseline on identical folds.
8. ML target-history features stop at `t-1` when predicting `t`.
9. Multi-step autoregressive forecasts may use earlier predictions, never hidden future actuals.
10. Hyperparameter selection must be nested inside chronological training-side validation.
11. Generated metrics are benchmark evidence only through checksum-verified manifests.
12. The CLI, artifact store, and dashboard use the same canonical evaluation code.

## Canonical package

```text
src/sales_forecasting/
├── cli.py
├── data/
│   ├── schema.py
│   ├── prepare.py
│   ├── missing.py
│   └── catalog.py
├── evaluation/
│   ├── metrics.py
│   ├── backtesting.py
│   ├── leaderboard.py
│   └── tuning.py
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

## Missing-period boundary

`prepare_time_series()` can expose gaps after regularization, but it does not resolve them. The evaluator applies the selected policy after slicing each training fold.

Supported policies:

```text
error        -> fail when training contains a gap
forward_fill -> each missing training value may use only an earlier observed value
```

No backward fill is allowed. No interpolation that depends on later timestamps is allowed. A missing value in the outer test window always fails evaluation because the true target is required for scoring.

The selected policy is part of the experiment configuration fingerprint.

## Nested tuning boundary

`NestedTunedForecaster.fit(outer_training)` performs this sequence:

```text
outer training history
  -> inner expanding-window folds
  -> evaluate every parameter candidate
  -> select best candidate by inner metric
  -> refit selected candidate on all outer training history
  -> forecast outer test window
```

The outer test window is not available during parameter selection. Tuning metadata is attached to each forecast result and persisted in the fold-metrics artifact as strict JSON.

## CLI boundary

The installed command is:

```text
sales-forecast
```

Subcommands:

- `inspect`: validate and summarize the prepared dataset contract.
- `run`: evaluate a reproducible leaderboard and write a manifest-backed run.
- `tune`: evaluate one nested-tuned ML challenger against the same naive baseline.

The root `run_forecasting.py` is only a compatibility shim to this CLI. It contains no independent forecasting logic.

## Run identity

`record_experiment()` fingerprints:

1. prepared dataset schema/timestamps/values;
2. evaluation settings, including missing policy;
3. model configuration and metadata;
4. package version and code revision.

This keeps causal preprocessing and tuning choices inside the reproducibility boundary.

## Artifact trust boundary

Each completed run contains a manifest plus checksummed leaderboard, fold metrics, and forecasts. The dashboard verifies those checksums before reading artifacts.

## CI

GitHub Actions runs the complete test suite on Python 3.10 and 3.13, imports the dashboard entry point, and smoke-tests the installed CLI help command.

## Next

Known-future regressors require an explicit train/future covariate contract before Prophet can be added. Ensemble weights must be learned from training-side validation rather than outer test results.
