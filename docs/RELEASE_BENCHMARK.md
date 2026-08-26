# v1 Release Benchmark

This benchmark is an acceptance/reproducibility check for v1.0. It is deliberately not presented as a broad model-performance study.

## Source acceptance

The reviewed weekly series was regenerated from the public Vehicle Sales Data source with the canonical source adapter.

- raw rows: **558,837**
- usable rows: **558,799**
- excluded rows: **38**
- unparsable timestamps: **33**
- parsed but out-of-era timestamps: **5**
- invalid targets: **12**

Some invalid conditions overlap, so category counts need not sum to the excluded-row total.

The full weekly grid contains 82 periods with 45 no-auction/no-observation gaps. v1 does not fill those targets. The benchmark selects the longest contiguous observed block: **32 weeks**, 2014-12-21 through 2015-07-26.

The raw-source acceptance workflow completed successfully in GitHub Actions run `32950969967` at commit `c98a7d324c84b763296dab201838c5a03d390774`. Its uploaded artifact digest is recorded in `data/benchmarks/car_prices_weekly_median.meta.json`.

## Evaluation

- target: weekly median `sellingprice`
- timezone: `America/Los_Angeles`
- initial train size: 24 weeks
- forecast horizon: 4 weeks
- step: 4 weeks
- outer folds: 2
- ranking metric: RMSE
- target imputation: none

| Rank | Model | MAE | RMSE | sMAPE | WAPE | RMSE vs naive |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ARIMA(1,1,1) | 1426.29 | **1764.63** | 10.66 | 10.15 | **-6.73%** |
| 2 | ETS | **1400.45** | 1783.29 | **10.45** | **9.96** | -5.74% |
| 3 | Last-value naive | 1550.00 | 1891.95 | 11.61 | 11.08 | baseline |
| 4 | Validation-weighted ensemble | 2300.52 | 2600.72 | 17.63 | 16.69 | +37.46% |
| 5 | XGBoost | 2177.88 | 2694.44 | 16.55 | 15.80 | +42.42% |
| 6 | Gradient Boosting | 2259.63 | 3191.44 | 18.46 | 16.18 | +68.69% |
| 7 | Random Forest | 2573.79 | 3234.81 | 19.92 | 18.70 | +70.98% |
| 8 | Prophet | 4111.02 | 4333.04 | 34.14 | 30.10 | +129.02% |

Only ARIMA and ETS beat the naive baseline on this acceptance benchmark. That does **not** establish universal superiority: two outer folds and 32 observations are too limited for a broad claim.

## LSTM gate

v1 does not add an LSTM. A future neural adapter must use the same chronological boundaries, deterministic/repeated seed reporting, and reproducible artifact contract, and it must demonstrate useful out-of-sample improvement on a substantially larger genuine time series before becoming a supported release model.
