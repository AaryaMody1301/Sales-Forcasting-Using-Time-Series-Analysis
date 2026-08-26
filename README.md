# Sales Forecasting Using Time Series Analysis

A forecasting project being rebuilt around explicit data contracts, chronological backtesting, and reproducible model comparisons.

## Current status: Phase 3

The canonical implementation lives under `src/sales_forecasting/`. Legacy modules directly under `src/` and the old dashboard remain migration-only code.

### Phase 1 — data + architecture
- explicit timestamp / target / frequency contracts
- no synthetic time axes
- no silent imputation
- one `ForecastModel` interface

### Phase 2 — evaluation + classical baselines
- deterministic last-value baseline
- expanding-window backtesting
- MAE, RMSE, sMAPE, MASE, WAPE
- working ARIMA and state-space ETS adapters

### Phase 3 — leakage-safe ML
- lag features built strictly from observations before the predicted timestamp
- rolling mean/std/min/max built from historical values only
- calendar features derived only from the known forecast timestamp
- recursive multi-step forecasts: previous predictions feed future lag windows, never hidden actuals
- Random Forest, Gradient Boosting, and XGBoost adapters
- one leaderboard that evaluates every model on identical chronological folds

## Benchmark policy

The required baseline is `LastValueNaiveModel`. A more complicated model is an improvement only when it beats the baseline on the **same folds and horizon**.

```python
from sales_forecasting import (
    FeatureSpec,
    LastValueNaiveModel,
    RandomForestForecaster,
    build_leaderboard,
)

spec = FeatureSpec(lags=(1, 7, 14, 28), rolling_windows=(7, 14, 28))

leaderboard = build_leaderboard(
    prepared_series,
    {
        "naive_last_value": LastValueNaiveModel,
        "random_forest": lambda: RandomForestForecaster(feature_spec=spec),
    },
    initial_train_size=180,
    horizon=7,
)

print(leaderboard.table)
```

The table includes aggregate metrics, rank, RMSE percentage difference versus the naive baseline, and whether each challenger actually beats that baseline.

## ML feature policy

For a target at time `t`, a feature may use:

```text
y[t-1], y[t-2], ...
rolling(y through t-1)
calendar(t)
```

It may **not** use:

```text
y[t]
y[t+1]
rolling windows that include y[t]
values from the held-out forecast horizon
```

During a 7-step forecast, step 2 can use the prediction from step 1 as a lag. It cannot use the real step-1 target because that value is still unknown at the original forecast origin.

## Built-in dataset status

`data/car_prices.csv` is currently registered as a daily median selling-price series. The bundled Amazon product/review CSV remains excluded because it has no observed daily-sales timeline.

## Install

Python 3.10+:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
```

## Roadmap

### Phase 4 — experiment artifacts + dashboard migration
- run IDs and JSON manifests
- dataset/config fingerprints
- deterministic metric/forecast artifacts
- dashboard loads manifests instead of guessing files
- clear distinction between failed runs, demos, and verified benchmarks

### Later
- Prophet only with explicitly supplied future regressors
- LSTM only after classical and tree baselines are reproducible
- validation-derived ensemble weights

## License

MIT. See `LICENSE`.
