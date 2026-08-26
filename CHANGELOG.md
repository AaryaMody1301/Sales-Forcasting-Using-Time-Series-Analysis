# Changelog

## 1.0.1 (unreleased)

### Reproducibility
- Locked the canonical release-benchmark environment to Python 3.13.15 and the exact dependency versions from the accepted v1 run.
- Added an explicit numerical acceptance file and made the release benchmark fail on model-set, ranking, or material mean-fold RMSE drift.
- Captured Python, pip, and `pip freeze` output in uploaded benchmark evidence.
- Clarified that reported aggregate RMSE is the arithmetic mean of per-fold RMSE values, not pooled holdout RMSE.

### Supply-chain and security
- Pinned official GitHub Actions to full immutable commit SHAs and disabled persisted checkout credentials.
- Upgraded benchmark artifact upload to the current pinned `actions/upload-artifact` major.
- Documented the pickle model-artifact trust boundary in `SECURITY.md` and the README.
- Added third-party data provenance/notice documentation.

### Release hygiene
- Updated package metadata and project URLs for the 1.0.1 hardening line.
- Updated README release wording so `v1.0.0` is identified as published while `main` targets `v1.0.1`.
- Added Dependabot configuration for Python and GitHub Actions maintenance.
- Added a tag-triggered release workflow that validates the tag/package version, builds distributions, and attaches them to the GitHub Release.

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
