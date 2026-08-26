# Sales Forecasting Using Time Series Analysis

A forecasting project rebuilt around explicit data contracts, chronological evaluation, leakage-safe features, reproducible experiment artifacts, and forecast-time covariate contracts.

## Current status: Phase 6

The canonical implementation lives in `src/sales_forecasting/` and is accessed through the installed `sales-forecast` CLI.

### Completed foundations

- **Phase 1:** explicit timestamp/target/frequency contracts and one model API.
- **Phase 2:** naive baseline, expanding-window evaluation, common metrics, ARIMA and ETS.
- **Phase 3:** leakage-safe lag/rolling features plus Random Forest, Gradient Boosting, and XGBoost.
- **Phase 4:** deterministic manifests, checksummed artifacts, dashboard migration, CI.
- **Phase 5:** canonical CLI, causal training-only missing handling, nested chronological tuning.
- **Phase 6:** explicit known-future regressors, Prophet, and validation-derived ensembles.

## Known-future regressor contract

Future covariates are stored separately from the target series. A regressor is accepted only when it is:

- explicitly named in the dataset contract;
- numeric and finite;
- unique and regular on the target frequency;
- available for every observed target timestamp;
- available for every requested future forecast timestamp.

This prevents the pipeline from silently filling or forecasting a supposedly "known" future variable. The exact regressor values are included in the dataset SHA-256 fingerprint.

Example target file:

```text
date,sales
2026-01-01,100
2026-01-02,104
...
```

Separate regressor file:

```text
date,promo,planned_price
2026-01-01,0,99.0
2026-01-02,1,95.0
...
2026-02-07,1,93.0
```

Inspect it with:

```bash
sales-forecast inspect \
  --csv sales.csv \
  --timestamp-col date \
  --target-col sales \
  --frequency D \
  --regressors-csv future_inputs.csv \
  --regressor-timestamp-col date \
  --regressor-cols promo planned_price
```

## Prophet

Prophet is optional so the base package remains lighter:

```bash
python -m pip install -e ".[prophet]"
```

Then compare Prophet against the same baseline/folds:

```bash
sales-forecast run \
  --csv sales.csv \
  --timestamp-col date \
  --target-col sales \
  --frequency D \
  --regressors-csv future_inputs.csv \
  --regressor-timestamp-col date \
  --regressor-cols promo planned_price \
  --models naive_last_value arima ets prophet \
  --initial-train-size 180 \
  --horizon 7
```

The adapter refuses a Prophet forecast if any declared future regressor value is unavailable. Prophet persistence uses its JSON serialization API rather than Python pickle.

## Validation-derived ensemble

Phase 6 adds `ValidationWeightedEnsemble`. Member weights are derived only from chronological validation folds inside each outer training history. The final outer holdout never determines the weights.

```bash
sales-forecast ensemble \
  --csv sales.csv \
  --timestamp-col date \
  --target-col sales \
  --frequency D \
  --members arima ets random_forest \
  --validation-initial-train-size 120 \
  --validation-horizon 7 \
  --initial-train-size 180 \
  --horizon 7
```

Weights are inverse validation error by default and are written into each fold's artifact metadata alongside the member validation scores.

## Reproducible benchmark policy

A benchmark claim is valid only when it comes from the canonical evaluator and a completed manifest-backed run. Run identity covers the target series, known-future regressor values, model/evaluation configuration, package version, and source revision.

The required baseline remains `LastValueNaiveModel`. A more complex model is considered an improvement only when it beats the baseline on identical chronological folds.

## Development

Python 3.10+:

```bash
python -m venv .venv
python -m pip install -e ".[dev,dashboard,prophet]"
python -m pytest
```

## Next: Phase 7

Phase 7 is final hardening: archive/remove remaining legacy modules, run the real car-data benchmark end-to-end, decide whether LSTM earns a place based on reproducible results, polish the dashboard/docs, and prepare the first v1.0 release.

## License

MIT. See `LICENSE`.
