# Sales Forecasting Using Time Series Analysis

A leakage-aware forecasting project with explicit dataset contracts, chronological evaluation, reproducible artifacts, and a manifest-backed dashboard.

## v1.0 status

Phase 7 is the release-candidate hardening pass. The supported implementation is the `sales_forecasting` package under `src/`; the old parallel forecasting implementations have been removed from the release tree.

Core guarantees:

- observed timestamps and observed targets only;
- no silent target imputation;
- expanding-window backtests with held-out future targets;
- nested chronological tuning for ML models;
- known-future regressors exposed only through each requested forecast horizon;
- validation-derived ensemble weights, never test-derived weights;
- deterministic run fingerprints and checksum-verified artifacts;
- every challenger compared against a last-value baseline on identical folds.

## Install

Python 3.10+:

```bash
python -m venv .venv
python -m pip install -e ".[dev,dashboard,prophet]"
pytest
```

The project uses `pyproject.toml` as its dependency and package metadata source.

## Quick benchmark

The repository includes a small reviewed real-data benchmark derived from the public Vehicle Sales Data source. Raw source files are not committed.

```bash
sales-forecast run \
  --csv data/benchmarks/car_prices_weekly_median.csv \
  --dataset car_prices_weekly \
  --models naive_last_value arima ets \
  --initial-train-size 24 \
  --horizon 4 \
  --step 4
```

Run the full release acceptance benchmark with:

```bash
python scripts/release_benchmark.py
```

### Reviewed v1 benchmark

Target: weekly median vehicle selling price. The benchmark uses the longest contiguous observed weekly segment: 32 observations from 2014-12-21 through 2015-07-26, with **no imputed targets**. Evaluation uses 24 initial training observations and two 4-week holdout folds.

| Rank | Model | RMSE | RMSE vs naive |
|---:|---|---:|---:|
| 1 | ARIMA(1,1,1) | 1764.63 | -6.73% |
| 2 | ETS | 1783.29 | -5.74% |
| 3 | Last-value naive | 1891.95 | baseline |
| 4 | Validation-weighted ensemble | 2600.72 | +37.46% |
| 5 | XGBoost | 2694.44 | +42.42% |
| 6 | Gradient Boosting | 3191.44 | +68.69% |
| 7 | Random Forest | 3234.81 | +70.98% |
| 8 | Prophet | 4333.04 | +129.02% |

This is a reproducibility/acceptance benchmark, **not** evidence that ARIMA is universally best. It contains only two outer folds and 32 contiguous weekly observations. See `docs/RELEASE_BENCHMARK.md`.

## LSTM decision

LSTM is intentionally **not included in v1.0**. The reviewed release series has only 32 contiguous weekly observations, and no leakage-safe LSTM implementation has demonstrated an improvement under the same benchmark protocol. Neural models can be reconsidered after a larger genuine time series is available and they earn inclusion against the same simple baselines.

## CLI

```text
sales-forecast inspect   Validate and summarize a dataset
sales-forecast run       Run a reproducible model leaderboard
sales-forecast tune      Nested chronological tuning for an ML model
sales-forecast ensemble  Learn validation-only ensemble weights
```

Supported model adapters include naive last value, ARIMA, ETS, Random Forest, Gradient Boosting, XGBoost, Prophet, and validation-weighted ensembles.

## Known future regressors

For genuinely known future covariates, provide a separate timestamped regressor CSV using `--regressors-csv`, `--regressor-timestamp-col`, and `--regressor-cols`. Missing future covariate rows are an error; the project does not guess them.

## Dashboard

```bash
streamlit run dashboard.py
```

The dashboard only opens completed manifest-backed runs whose referenced artifacts pass checksum verification.

## Data

See `data/README.md`. The raw ~87 MB vehicle-sales file and the former Amazon product/review file are not included in the v1 release tree. Download raw source data separately and regenerate the reviewed benchmark with `scripts/prepare_vehicle_sales.py`.

## Project structure

```text
src/sales_forecasting/   canonical package
scripts/                 reproducible preparation / acceptance commands
data/benchmarks/         small reviewed benchmark input + provenance
examples/                parameter-grid examples
tests/                   contract, model, evaluation, artifact, CLI tests
docs/                    architecture/model/release evidence
```

## Benchmark policy

No accuracy or “best model” claim is a project result unless it comes from the canonical chronological evaluator with identified data/configuration and reproducible artifacts. A more complex model does not earn promotion merely because it is more sophisticated.

## License

MIT. See `LICENSE`.
