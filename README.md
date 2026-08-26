# Sales Forecasting Using Time Series Analysis

A time-series forecasting project currently being refactored for reproducible data preparation, leakage-aware evaluation, and one consistent model API.

## Refactor status

**Phase 1 is the current foundation.** The repository previously accumulated multiple forecasting runners and helper scripts with incompatible model APIs. Some demo artifacts also used synthetic dates or hard-coded/sample metrics. Those outputs are not treated as benchmark evidence.

The new canonical code lives under:

```text
src/sales_forecasting/
```

Legacy modules directly under `src/`, the old root runner, and the current Streamlit dashboard remain temporarily for migration. New development should target the canonical package only.

## Phase 1 rules

A dataset is forecastable only when the project can state, explicitly:

- which column is the observed timestamp;
- which numeric column is the observed target;
- what time frequency the target represents;
- how transaction/event rows are aggregated to one value per period;
- which regressors, if any, are genuinely known at forecast time.

The data layer never invents timestamps and never silently fills missing periods.

Pandas treats resampling as a time-based grouping operation, so transaction-level datasets are converted to a regular series through an explicit aggregation rather than by pretending every row is the next time step.

## Built-in dataset status

### Car prices

`data/car_prices.csv` contains transaction dates. The first reviewed schema converts it to **daily median selling price**:

```python
from sales_forecasting import CAR_PRICES_DAILY_MEDIAN, prepare_time_series

prepared = prepare_time_series(car_prices_df, CAR_PRICES_DAILY_MEDIAN)
series = prepared.values
```

Other valid business targets—such as daily sale count or daily revenue—should be registered as separate schemas instead of being mixed into the same target definition.

### Amazon product/review data

`data/amazon.csv` is **not** registered as a sales forecasting dataset. It does not contain an observed daily-sales timeline. The old approach that generated a date sequence from row order and used `discounted_price * rating_count` as `daily_sales` is intentionally excluded from the new package.

To add an Amazon sales dataset later, use observed records such as:

```text
timestamp | product_id | units_sold | revenue
```

and choose an explicit target/frequency.

## Install the Phase 1 package

Python 3.10+ is required.

```bash
python -m venv .venv
```

Activate the environment, then install the package and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

`requirements.txt` is retained temporarily for the legacy application. New package dependency management starts in `pyproject.toml`.

## Canonical architecture

```text
.
├── pyproject.toml
├── ARCHITECTURE.md
├── src/
│   ├── sales_forecasting/
│   │   ├── data/
│   │   │   ├── schema.py
│   │   │   ├── prepare.py
│   │   │   └── catalog.py
│   │   └── models/
│   │       └── base.py
│   └── ... legacy modules pending migration
└── tests/
```

See `ARCHITECTURE.md` for the migration rules.

## Canonical model contract

Every model added in Phase 2 or later will implement the same interface:

```text
fit(training_series)
forecast(horizon)
save(path)
load(path)
```

Train/test splitting will be owned by a separate evaluator rather than by each individual model. This keeps statistical, ML, and deep-learning models on the same backtesting definition.

## Roadmap

### Phase 1 — foundation
- modern `src` package and `pyproject.toml`
- explicit dataset contracts
- no synthetic time axes
- one model interface
- remove generated/demo artifacts from the canonical source tree

### Phase 2 — evaluation + first models
- naive baseline
- rolling/expanding time-series backtesting
- standardized metrics
- ETS and ARIMA/SARIMA adapters
- leakage-safe lag features

### Phase 3 — advanced models
- Random Forest / Gradient Boosting / XGBoost
- Prophet with explicit future regressors
- LSTM only after baseline and classical models are reproducible
- validation-derived ensemble weights

### Phase 4 — artifacts + dashboard
- run manifest
- deterministic result directories
- dashboard reads manifests instead of guessing filenames
- proper decomposition and error states

## Benchmark policy

No accuracy, RMSE, MAPE, R², or “best model” result should be presented as a project result unless it is produced by the canonical evaluation pipeline from an identified dataset/configuration and can be reproduced from code.

## License

MIT. See `LICENSE`.
