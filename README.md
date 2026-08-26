# Sales Forecasting Using Time Series Analysis

A time-series forecasting project being rebuilt around reproducible data contracts, leakage-aware evaluation, and one consistent model API.

## Current status: Phase 2

Phase 1 established the canonical `src/sales_forecasting/` package and explicit dataset contracts. Phase 2 adds the first complete evaluation layer and working baseline/statistical models.

The legacy modules directly under `src/`, the old root runner, and the current Streamlit dashboard remain temporarily for migration. New development should target the canonical package only.

## Forecastable data rules

A dataset is forecastable only when the project can state explicitly:

- which column is the observed timestamp;
- which numeric column is the observed target;
- what time frequency the target represents;
- how transaction/event rows are aggregated to one value per period;
- which regressors, if any, are genuinely known at forecast time.

The canonical data layer never invents timestamps and never silently fills missing periods.

## Built-in dataset status

### Car prices

`data/car_prices.csv` contains transaction dates. The first reviewed schema converts it to **daily median selling price**.

```python
import pandas as pd
from sales_forecasting import CAR_PRICES_DAILY_MEDIAN, prepare_time_series

car_prices = pd.read_csv("data/car_prices.csv")
prepared = prepare_time_series(car_prices, CAR_PRICES_DAILY_MEDIAN)
```

Other valid business targets—such as daily sale count or daily revenue—should be registered as separate schemas.

### Amazon product/review data

`data/amazon.csv` is **not** registered as a sales forecasting dataset. It does not contain an observed daily-sales timeline. The old approach that generated dates from row order and used `discounted_price * rating_count` as `daily_sales` is intentionally excluded from the canonical package.

## Install

Python 3.10+ is required.

```bash
python -m venv .venv
```

Activate the environment, then install the canonical package and development dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
```

`requirements.txt` remains temporarily for the legacy application. Canonical dependencies are defined in `pyproject.toml`.

## Phase 2 models

Three models now implement the same contract:

```text
LastValueNaiveModel
ARIMAForecaster
ETSForecaster
```

Example:

```python
from sales_forecasting import ARIMAForecaster

model = ARIMAForecaster(order=(1, 1, 1)).fit(training_series)
future = model.forecast(30)
```

A model receives only its training series. It does not own or inspect the holdout window.

## Expanding-window backtesting

The evaluator creates a fresh model for every chronological fold:

```python
from sales_forecasting import LastValueNaiveModel, expanding_window_backtest

result = expanding_window_backtest(
    prepared,
    LastValueNaiveModel,
    initial_train_size=180,
    horizon=30,
)

print(result.aggregate)
```

At each fold, training observations occur strictly before the test interval. The next fold expands its history with observations that would have become known in production.

Missing periods must be resolved by a future training-only preprocessing step; the evaluator refuses to silently impute them.

## Canonical metrics

Every model is compared using:

- MAE
- RMSE
- sMAPE
- MASE
- WAPE

MAPE is not part of the canonical set because zero/near-zero actuals can make it undefined or misleading.

## Architecture

```text
.
├── pyproject.toml
├── ARCHITECTURE.md
├── src/
│   ├── sales_forecasting/
│   │   ├── data/
│   │   ├── evaluation/
│   │   └── models/
│   └── ... legacy modules pending migration
└── tests/
```

See `ARCHITECTURE.md` for the source-of-truth migration rules.

## Benchmark policy

No accuracy, RMSE, MAPE, R², or “best model” result should be presented as a project result unless it is produced by the canonical evaluation pipeline from an identified dataset/configuration and can be reproduced from code.

A complex model should not be promoted as an improvement unless it beats the deterministic naive baseline on the same backtest folds and metric definitions.

## Roadmap

### Phase 1 — foundation
- canonical package and project metadata
- explicit dataset contracts
- no synthetic time axes
- one model interface
- generated/demo artifact cleanup

### Phase 2 — evaluation + first models
- deterministic last-value baseline
- expanding-window backtesting
- standardized metrics
- working ETS and ARIMA adapters

### Phase 3 — ML models
- leakage-safe lag/rolling features
- Random Forest / Gradient Boosting / XGBoost
- model comparison against the baseline
- explicit known-future regressor handling

### Phase 4 — artifacts + dashboard
- run manifest
- deterministic result directories
- dashboard reads manifests instead of guessing filenames
- proper decomposition and explicit error states

## License

MIT. See `LICENSE`.
