# Phase 1 Architecture

This document defines the canonical architecture introduced during the repository refactor.

## Principles

1. **Observed time only.** Forecasting datasets must contain an observed timestamp column. Code must never manufacture a date range from row order.
2. **Observed targets only.** A forecast target must represent the quantity being modeled. A proxy such as `price * rating_count` cannot be labeled daily sales.
3. **Explicit frequency.** Every forecasting dataset declares the period it represents (`D`, `W`, `MS`, and so on).
4. **Explicit aggregation.** Transaction/event data with many rows per period must declare how rows become one target value per period.
5. **No silent imputation.** Missing periods remain missing at the dataset-contract layer. Later preprocessing must learn any imputation rule from training data only.
6. **One model interface.** All Phase 2+ model implementations use `ForecastModel`.
7. **Backtesting is separate from future forecasting.** Phase 2 will build a time-aware evaluator around the canonical contracts instead of allowing each model to define its own test logic.

## Canonical source tree

```text
src/
└── sales_forecasting/
    ├── __init__.py
    ├── data/
    │   ├── schema.py
    │   ├── prepare.py
    │   └── catalog.py
    └── models/
        └── base.py
```

The older modules directly under `src/` are legacy code. They remain temporarily so the refactor can migrate behavior deliberately rather than performing a risky one-shot rewrite. New implementation work must go into `src/sales_forecasting/`.

## Dataset contracts

A `DatasetSchema` declares:

- timestamp column
- target column
- frequency
- optional aggregation for event/transaction data
- optional timezone
- explicitly known future regressors

`prepare_time_series()` validates these requirements, sorts observations, performs only the requested aggregation, regularizes the index, and reports missing periods. It does **not** fill those gaps.

### Car prices

The bundled car transaction dataset can support several legitimate forecasting questions. Phase 1 registers one conservative example:

```text
dataset: car_prices_daily_median
timestamp: saledate
target: sellingprice
frequency: daily
aggregation: median
```

This represents daily median selling price. Future phases can add separately named schemas for daily transaction count or revenue.

### Amazon product/review data

The bundled Amazon file is not registered as a forecasting dataset. It lacks an observed daily sales time axis and an observed daily sales/revenue target. The previous behavior that generated dates from row order is intentionally not part of the canonical package.

## Model contract

Every future model must implement:

```text
fit(training_series)
forecast(horizon)
save(path)
load(path)
```

`ForecastResult` standardizes output shape and requires a clean, ordered `DatetimeIndex`.

The evaluator—not the model—will own train/test splitting in Phase 2. This prevents individual models from using different definitions of a test forecast.

## Generated artifacts

`results/`, `models/`, `forecasts/`, `data/processed/`, compiled Python files, and serialized model files are generated artifacts and should not be committed. Historical/demo artifacts are being removed from the canonical branch because they are not reproducible benchmark evidence.

## Next phase

Phase 2 will implement:

1. deterministic naive baseline
2. rolling/expanding backtesting
3. common metrics
4. ETS and ARIMA adapters
5. ML lag-feature pipeline with no look-ahead
6. run manifests for dashboard consumption
