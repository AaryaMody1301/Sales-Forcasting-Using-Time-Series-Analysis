# Forecasting Architecture

This document defines the canonical architecture introduced by the repository refactor.

## Principles

1. **Observed time only.** Forecasting datasets must contain an observed timestamp column. Code must never manufacture a date range from row order.
2. **Observed targets only.** A forecast target must represent the quantity being modeled. A proxy such as `price * rating_count` cannot be labeled daily sales.
3. **Explicit frequency.** Every forecasting dataset declares the period it represents (`D`, `W`, `MS`, and so on).
4. **Explicit aggregation.** Transaction/event data with many rows per period must declare how rows become one target value per period.
5. **No silent imputation.** Missing periods remain missing at the dataset-contract layer. Preprocessing must learn any imputation rule from training data only.
6. **One model interface.** All canonical model implementations use `ForecastModel`.
7. **Backtesting is separate from future forecasting.** The evaluator owns chronological splits; models only fit the training series they receive and forecast the requested horizon.
8. **Every serious model must beat a baseline.** Phase 2 introduces `LastValueNaiveModel` as the minimum benchmark.

## Canonical source tree

```text
src/
└── sales_forecasting/
    ├── data/
    │   ├── schema.py
    │   ├── prepare.py
    │   └── catalog.py
    ├── evaluation/
    │   ├── metrics.py
    │   └── backtesting.py
    └── models/
        ├── base.py
        ├── naive.py
        └── statistical.py
```

The older modules directly under `src/` are legacy code. They remain temporarily so behavior can be migrated deliberately. New implementation work must go into `src/sales_forecasting/`.

## Dataset contracts

A `DatasetSchema` declares the timestamp column, target column, frequency, optional aggregation, optional timezone, and explicitly known future regressors.

`prepare_time_series()` validates these requirements, sorts observations, performs only the requested aggregation, regularizes the index, and reports missing periods. It does **not** fill those gaps.

### Car prices

The first built-in car contract represents daily median selling price:

```text
dataset: car_prices_daily_median
timestamp: saledate
target: sellingprice
frequency: daily
aggregation: median
```

### Amazon product/review data

The bundled Amazon file is not registered as a forecasting dataset because it lacks an observed daily sales time axis and observed daily sales/revenue target.

## Model contract

Every canonical model implements:

```text
fit(training_series)
forecast(horizon)
save(path)
load(path)
```

Phase 2 includes:

- `LastValueNaiveModel`
- `ARIMAForecaster`
- `ETSForecaster`

All use the same `ForecastResult` shape.

## Evaluation contract

`expanding_window_backtest()` creates a fresh model for every fold. A fold can only train on observations strictly before its test interval. By default, the next fold advances by the forecast horizon, producing non-overlapping test windows.

The evaluator rejects unresolved missing periods rather than filling them with future information. A later preprocessing phase can introduce causal, training-only gap handling.

Phase 2 reports the same metrics for every model:

- MAE
- RMSE
- sMAPE
- MASE
- WAPE

MAPE is deliberately not canonical because zero or near-zero actual values make it undefined or unstable.

## Statistical adapters

ARIMA uses `statsmodels.tsa.arima.model.ARIMA`. ETS uses the state-space `statsmodels.tsa.exponential_smoothing.ets.ETSModel`. Model-fitting warnings remain visible; the package does not globally suppress convergence or specification warnings.

## Generated artifacts

`results/`, `models/`, `forecasts/`, `data/processed/`, compiled Python files, and serialized model files are generated artifacts and should not be committed.

## Next phase

Phase 3 will add leakage-safe lag/rolling features, Random Forest/Gradient Boosting/XGBoost adapters, explicit known-future regressor handling, and model comparison against the naive baseline.
