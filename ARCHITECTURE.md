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
└── models/
    ├── base.py
    ├── naive.py
    ├── statistical.py
    └── ml.py
```

## Leakage boundary

The evaluator slices the complete series at a forecast origin and constructs a `PreparedSeries` containing only the training portion. An ML model then creates supervised features only inside that training object.

For row `t`, lag and rolling values are derived from `series[:t]`. Calendar features can describe timestamp `t` because the timestamp is known before its target is observed.

At inference, ML forecasts are recursive:

```text
history -> predict t+1
history + prediction(t+1) -> predict t+2
...
```

This matches a true fixed-origin multi-step forecast and avoids the previous mistake of generating test lag features from actual holdout values.

## Leaderboard

`build_leaderboard()` accepts model factories and applies the same expanding-window configuration to every model. It records each full `BacktestResult` and returns a sortable metric table with RMSE delta versus `naive_last_value`.

Model hyperparameter tuning is intentionally not part of this phase. When added, tuning must occur inside training folds or a nested chronological validation procedure; the final test windows cannot be used to choose hyperparameters.

## Phase 4

The next layer will add run manifests and artifact storage so the dashboard consumes explicit model/run metadata rather than relying on filename patterns or old generated result folders.
