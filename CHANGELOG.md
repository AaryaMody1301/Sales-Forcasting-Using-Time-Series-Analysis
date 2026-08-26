# Changelog

## 1.0.0

### Architecture
- Established `src/sales_forecasting/` as the only supported implementation.
- Removed legacy parallel forecasting/model runners from the release tree.
- Added an installed `sales-forecast` CLI and manifest-backed Streamlit dashboard.

### Data correctness
- Added explicit dataset/frequency/aggregation/timezone contracts.
- Rejected the former Amazon product/review file as a sales time-series benchmark.
- Added auditable vehicle-sales source cleaning for malformed/out-of-era rows.
- Added a reviewed weekly vehicle-price benchmark with no target imputation.
- Removed raw third-party CSVs from the release tree.

### Evaluation and models
- Added expanding-window backtesting and common MAE/RMSE/sMAPE/MASE/WAPE metrics.
- Added naive, ARIMA, ETS, Random Forest, Gradient Boosting, XGBoost, and Prophet adapters.
- Added leakage-safe lag/rolling features, nested chronological tuning, known-future regressors, and validation-derived ensembles.
- Explicitly excluded LSTM from v1 pending stronger same-protocol evidence on a larger genuine time series.

### Reproducibility
- Added deterministic dataset/config fingerprints, run IDs, manifests, checksums, environment metadata, and atomic artifact writes.
- Added CI on Python 3.10 and 3.13, package build checks, dashboard/CLI smoke tests, and a separate release benchmark workflow.

### Breaking cleanup
- `pyproject.toml` is the dependency/package source of truth; the legacy `requirements.txt` generator and file were removed.
- Old archived generators, stale notebooks, and duplicate modules directly under `src/` were removed from the release tree. Git history remains available for archaeology.
