# Data

Raw external datasets are intentionally not stored in the v1 release tree.

## Vehicle sales benchmark

The reviewed benchmark at `benchmarks/car_prices_weekly_median.csv` is derived from the public Kaggle **Vehicle Sales Data** dataset:

https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data

The original source contains used-vehicle auction transactions. The v1 preparation flow:

1. parses `saledate` with its source timezone offset;
2. rejects malformed or implausibly out-of-era timestamps rather than guessing them;
3. rejects non-numeric/missing `sellingprice` values;
4. aggregates to weekly median selling price in `America/Los_Angeles`;
5. selects the longest contiguous observed weekly segment;
6. does not impute target values.

`benchmarks/car_prices_weekly_median.meta.json` records the exact source-cleaning counts and the GitHub Actions acceptance run that produced the reviewed series.

To regenerate the derived benchmark, download the raw `car_prices.csv` yourself and run:

```bash
python scripts/prepare_vehicle_sales.py --input /path/to/car_prices.csv
```

## Amazon product/review data

The former Amazon product/review CSV is not part of v1. It does not contain an observed daily sales timeline and is therefore not a defensible forecasting benchmark. Use a genuinely timestamped transaction/revenue dataset instead.
