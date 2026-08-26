# v1 Architecture

The v1 package has one supported forecasting path: `src/sales_forecasting/`.

## Boundaries

1. **Source adapters** handle quirks belonging to a named external source and report exclusions explicitly. They never guess malformed targets or timestamps.
2. **Dataset contracts** define timestamp, target, frequency, aggregation, timezone, and known-future regressors.
3. **Preparation** regularizes the requested time grid but does not silently fill target gaps.
4. **Evaluation** owns chronological train/test separation and applies any permitted training-only missing policy after each training fold is sliced.
5. **Models** receive training targets only. Known-future covariates may extend through the requested forecast horizon, never beyond it during a fold.
6. **Tuning/ensembles** learn parameters or weights from inner chronological validation contained inside the outer training history.
7. **Artifacts** fingerprint the dataset/configuration/code revision and store checksum-verified fold forecasts and metrics behind a manifest.
8. **Dashboard** reads only those manifests; it never infers model identity from filenames or invents fallback metrics.

## Canonical tree

```text
src/sales_forecasting/
├── artifacts/
├── dashboard/
├── data/
├── evaluation/
├── features/
├── models/
└── cli.py
```

Legacy parallel runners and model implementations were removed from the v1 release tree. Root `run_forecasting.py` remains only as a compatibility shim to the installed CLI, and root `dashboard.py` remains a small Streamlit launcher.

## Data policy

Raw third-party datasets are not versioned in the v1 release tree. The repository ships only a small reviewed weekly vehicle-price benchmark plus provenance metadata. `clean_vehicle_sales_source()` is the source-specific boundary for the original vehicle-sales CSV; malformed/out-of-era rows are excluded and counted before aggregation.

The reviewed benchmark is the longest contiguous observed weekly segment. Missing auction weeks are not filled.

## Evaluation policy

All leaderboard models are evaluated on the same expanding windows. The deterministic last-value model is always the baseline. Backtesting and future forecasting remain conceptually separate: test targets are used only for scoring after predictions are made.

ML lag/rolling features stop at `t-1`. Recursive multi-step forecasts may consume earlier predictions, not hidden future actuals.

## Model policy

v1 includes naive, ARIMA, ETS, Random Forest, Gradient Boosting, XGBoost, Prophet, nested tuning, and validation-weighted ensembles. Inclusion in the API is not a performance claim.

LSTM is excluded from v1 because the reviewed release series is only 32 contiguous weekly observations and no canonical LSTM has earned inclusion under the same chronological benchmark. See `docs/MODEL_POLICY.md`.

## Release evidence

`python scripts/release_benchmark.py` runs the fixed v1 acceptance benchmark from the reviewed CSV. GitHub Actions runs it independently from the normal unit/integration CI matrix. `docs/RELEASE_BENCHMARK.md` records the accepted source-cleaning and leaderboard evidence.
