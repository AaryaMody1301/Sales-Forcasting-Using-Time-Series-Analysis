# Sales Forecasting Using Time Series Analysis

A forecasting project rebuilt around explicit data contracts, chronological backtesting, leakage-safe model features, and reproducible experiment artifacts.

## Current status: Phase 4

The canonical implementation lives under `src/sales_forecasting/`. Legacy modules directly under `src/` are migration-only code.

### Phase 1 - data + architecture
- explicit timestamp / target / frequency contracts
- no synthetic time axes
- no silent imputation
- one `ForecastModel` interface

### Phase 2 - evaluation + classical baselines
- deterministic last-value baseline
- expanding-window backtesting
- MAE, RMSE, sMAPE, MASE, WAPE
- working ARIMA and state-space ETS adapters

### Phase 3 - leakage-safe ML
- lag and rolling features stop at `t-1` when predicting `t`
- calendar features use only the known forecast timestamp
- recursive multi-step forecasts never consume hidden holdout targets
- Random Forest, Gradient Boosting, and XGBoost adapters
- one leaderboard evaluated on identical chronological folds

### Phase 4 - reproducible runs + dashboard
- deterministic run IDs derived from dataset, evaluation/model config, and code revision
- SHA-256 dataset and configuration fingerprints
- JSON run manifests with code/dependency versions and model metadata
- checksummed leaderboard, fold-metric, and forecast artifacts
- atomic run-directory replacement so interrupted writes do not become completed runs
- manifest-driven Streamlit dashboard with artifact-integrity checks
- GitHub Actions CI for the canonical package

## Record an experiment

```python
from sales_forecasting import (
    ExperimentSpec,
    FeatureSpec,
    LastValueNaiveModel,
    ModelSpec,
    RandomForestForecaster,
    record_experiment,
)

feature_spec = FeatureSpec(
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 14, 28),
)

run = record_experiment(
    prepared_series,
    (
        ModelSpec("naive_last_value", LastValueNaiveModel),
        ModelSpec(
            "random_forest",
            lambda: RandomForestForecaster(feature_spec=feature_spec),
            metadata={"purpose": "phase-4 benchmark"},
        ),
    ),
    ExperimentSpec(
        initial_train_size=180,
        horizon=7,
    ),
    code_revision="<git-commit-sha>",
)

print(run.run_id)
print(run.run_dir)
```

`ModelSpec` records the model implementation plus the public configuration exposed by a fresh model instance. Optional `metadata` is for experiment context and is also included in the configuration fingerprint.

For exact source reproducibility, pass the Git commit SHA as `code_revision`. In GitHub Actions, `GITHUB_SHA` is used automatically when available.

## Artifact layout

A completed run is written under a deterministic path:

```text
artifacts/
└── runs/
    └── <dataset>-<dataset-hash>-<config-hash>/
        ├── manifest.json
        ├── leaderboard.csv
        └── models/
            └── <model-label>-<label-hash>/
                ├── fold_metrics.csv
                └── forecasts.csv
```

The manifest records:

- dataset schema and SHA-256 fingerprint
- evaluation settings and configuration fingerprint
- package, Python, dependency, and source revision information
- model implementation/configuration metadata
- aggregate metrics and baseline comparison
- relative artifact paths
- SHA-256 checksum and byte size for every CSV artifact

The dashboard verifies each checksum before loading data. A file that was changed after the run was recorded is rejected rather than silently displayed as benchmark evidence.

## Dashboard

Install the optional dashboard dependency:

```bash
python -m pip install -e ".[dashboard]"
```

Then launch:

```bash
streamlit run dashboard.py
```

The dashboard only discovers `artifacts/runs/*/manifest.json`. It no longer guesses model names from filenames or falls back to fake metrics when an artifact is missing.

## Benchmark policy

The required baseline is `LastValueNaiveModel`. A more complicated model is an improvement only when it beats the baseline on the **same folds and horizon**.

No accuracy, RMSE, MAPE, R2, or "best model" claim should be presented as a project result unless it comes from a completed canonical run with a valid manifest and intact artifact checksums.

## Built-in dataset status

`data/car_prices.csv` is registered as a daily median selling-price series. The bundled Amazon product/review CSV remains excluded because it has no observed daily-sales timeline.

## Development

Python 3.10+:

```bash
python -m venv .venv
python -m pip install -e ".[dev,dashboard]"
python -m pytest
```

CI runs the canonical test suite on pull requests and pushes to `main`.

## Next

A later phase can add a canonical CLI, training-only missing-period policies, nested chronological hyperparameter tuning, known-future regressors/Prophet, validation-derived ensembles, and only then reconsider LSTM models.

## License

MIT. See `LICENSE`.
