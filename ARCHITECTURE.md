# Canonical Forecasting Architecture

## Core rules

1. Targets must be observed values on an explicit regular time axis.
2. Train/test separation is owned by the evaluator, never individual models.
3. Missing-target handling is causal and training-only.
4. Lag/rolling features stop at `t-1` when predicting `t`.
5. Hyperparameters are selected only on chronological validation inside outer training history.
6. A known-future regressor is a separate covariate whose value genuinely exists before the target is observed.
7. Future covariates are exposed only through the current evaluation horizon; later covariate rows remain hidden from that fold.
8. Prophet cannot guess a missing declared future regressor.
9. Ensemble weights are learned from training-side validation, never the outer test results.
10. Every challenger uses identical outer folds and is compared with the deterministic naive baseline.
11. Benchmark artifacts are trusted only through a completed manifest and checksum verification.

## Package layout

```text
src/sales_forecasting/
├── cli.py
├── data/
│   ├── schema.py
│   ├── prepare.py
│   ├── missing.py
│   ├── regressors.py
│   └── catalog.py
├── evaluation/
│   ├── metrics.py
│   ├── backtesting.py
│   ├── leaderboard.py
│   ├── tuning.py
│   └── ensemble.py
├── features/
│   └── lags.py
├── models/
│   ├── base.py
│   ├── naive.py
│   ├── statistical.py
│   ├── ml.py
│   └── prophet.py
├── artifacts/
└── dashboard/
```

## Known-future covariate boundary

`PreparedSeries.values` contains observed targets only. `PreparedSeries.future_regressors` is a separate regular dataframe that may extend beyond the final target timestamp.

```text
observed target:       t1 ... t100
known covariates:      t1 ... t100 t101 ... t107
forecast origin:                  ^
outer horizon:                       t101 ... t107
```

During an outer fold ending at `t107`, the training object may contain regressor values through `t107`, but target values stop at the forecast origin. Values after `t107` are not exposed to that model fit.

This is materially different from leaking targets: calendar plans, promotions, contracted prices, and similar covariates can be known before their associated sales outcome occurs. A variable whose future value is not known must not be declared through this contract.

## Prophet boundary

The Prophet adapter registers only explicitly declared regressors. Training data contains target history plus the regressor history for the same timestamps. Prediction data contains only future timestamps and the corresponding known regressor values.

A missing future covariate is an error. Constant training regressors are rejected because they contain no estimable signal.

Prophet model persistence uses `prophet.serialize.model_to_json` / `model_from_json`. The future-regressor frame is persisted separately in the adapter payload so a restored fitted model retains the exact forecast-time covariates it was given.

## Ensemble boundary

`ValidationWeightedEnsemble.fit(outer_training)` performs:

```text
outer training history
  -> inner expanding-window evaluation for each member
  -> member validation score
  -> inverse-error weights
  -> refit every member on all outer training history
  -> weighted forecast for outer holdout
```

Zero-error members receive all available ensemble weight, shared equally if more than one member has zero validation error. Otherwise weights are proportional to `1 / error^weight_power` and normalized to sum to one.

The member scores, weights, metric, and inner validation settings are attached to forecast metadata and therefore written into fold artifacts.

## Reproducibility

The Phase 6 dataset fingerprint is versioned as `series-v2` and hashes:

- semantic dataset schema;
- every target timestamp/value;
- every known-future-regressor timestamp, column name, and value.

The manifest records regressor names, coverage range, row count, and the number of future regressor periods beyond the final target.

## Phase 7

Final hardening will remove/archive the remaining legacy implementations, execute the canonical pipeline against the real car dataset, review model/runtime tradeoffs, decide whether an LSTM adds measurable value, polish portfolio evidence, and prepare v1.0.
