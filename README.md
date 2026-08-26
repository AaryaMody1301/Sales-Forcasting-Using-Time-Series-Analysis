# Sales Forecasting Using Time Series Analysis

A forecasting project rebuilt around explicit data contracts, chronological backtesting, leakage-safe features, reproducible artifacts, and a canonical CLI.

## Current status: Phase 5

The canonical implementation lives under `src/sales_forecasting/`. The old root runner now forwards to the same CLI so there is one supported execution path.

### Phase 1 - data + architecture
- explicit timestamp / target / frequency contracts
- no synthetic time axes
- no silent imputation
- one `ForecastModel` interface

### Phase 2 - evaluation + classical baselines
- deterministic last-value baseline
- expanding-window backtesting
- MAE, RMSE, sMAPE, MASE, WAPE
- ARIMA and state-space ETS adapters

### Phase 3 - leakage-safe ML
- lag and rolling features stop at `t-1`
- recursive multi-step forecasts never consume hidden holdout targets
- Random Forest, Gradient Boosting, and XGBoost adapters
- one shared leaderboard

### Phase 4 - reproducible runs + dashboard
- deterministic run IDs and SHA-256 fingerprints
- schema-versioned manifests and checksummed artifacts
- manifest-driven Streamlit dashboard
- GitHub Actions CI

### Phase 5 - CLI + causal missing policy + nested tuning
- installed `sales-forecast` command with `inspect`, `run`, and `tune` subcommands
- `run_forecasting.py` is now only a compatibility shim to that CLI
- missing periods default to `error`; optional `forward_fill` uses prior observations only
- test/holdout targets are never imputed
- nested chronological hyperparameter search occurs inside each outer training fold
- selected tuning parameters and inner scores are retained in fold artifact metadata

## Install

Python 3.10+:

```bash
python -m venv .venv
python -m pip install -e ".[dev,dashboard]"
```

## CLI

### Inspect a dataset

```bash
sales-forecast inspect \
  --csv data/car_prices.csv \
  --dataset car_prices
```

For a custom dataset, provide its semantics explicitly:

```bash
sales-forecast inspect \
  --csv data/my_sales.csv \
  --timestamp-col date \
  --target-col sales \
  --frequency D \
  --aggregation sum
```

### Run a benchmark

```bash
sales-forecast run \
  --csv data/car_prices.csv \
  --dataset car_prices \
  --models naive_last_value arima ets random_forest \
  --initial-train-size 180 \
  --horizon 7 \
  --step 7
```

If regularization created gaps and forward filling is semantically appropriate for the target, it must be requested explicitly:

```bash
--missing-policy forward_fill
```

Forward fill is applied independently inside each training fold. Missing targets inside a held-out test window remain an error and are never filled for scoring.

### Nested chronological tuning

Create a JSON grid, for example:

```json
{
  "n_estimators": [100, 300],
  "max_depth": [3, 6]
}
```

Then run:

```bash
sales-forecast tune \
  --csv data/my_sales.csv \
  --timestamp-col date \
  --target-col sales \
  --frequency D \
  --model random_forest \
  --grid rf_grid.json \
  --initial-train-size 180 \
  --horizon 7 \
  --inner-initial-train-size 90 \
  --inner-horizon 7
```

The outer fold is still the benchmark fold. Hyperparameters are selected only from inner expanding-window validation performed inside that outer fold's training history.

## Artifact layout

Canonical CLI runs use the Phase 4 artifact store:

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

`fold_metrics.csv` includes strict JSON model metadata for each fold. For tuned models that includes the selected parameter set, inner metric, best inner score, and candidate scores.

The missing-data policy is part of the experiment configuration, so changing from `error` to `forward_fill` changes the configuration fingerprint and run ID.

## Dashboard

```bash
streamlit run dashboard.py
```

The dashboard discovers only completed manifests and verifies artifact checksums before reading any CSV.

## Benchmark policy

The required baseline is `LastValueNaiveModel`. A more complicated model is an improvement only when it beats that baseline on the same outer folds and horizon.

Hyperparameters may be selected only from training-side validation. Final outer test windows cannot influence model or parameter selection.

No benchmark claim should be presented unless it comes from a completed canonical run with a valid manifest and intact checksums.

## Dataset status

`data/car_prices.csv` is registered as a daily median selling-price series. The bundled Amazon product/review CSV remains excluded because it has no observed daily-sales timeline.

## Development

```bash
python -m pytest
sales-forecast --help
```

CI runs the full canonical suite on Python 3.10 and 3.13.

## Next

The next phase can add explicit known-future-regressor contracts and Prophet, followed by validation-derived ensembles. LSTM should remain deferred until those simpler approaches are reproducible and justified by the same benchmark protocol.

## License

MIT. See `LICENSE`.
