#!/usr/bin/env python3
"""Prepare the reviewed weekly benchmark from a downloaded vehicle-sales CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sales_forecasting.data import clean_vehicle_sales_source
from sales_forecasting.data.catalog import CAR_PRICES_WEEKLY_MEDIAN
from sales_forecasting.data.prepare import prepare_time_series
from sales_forecasting.data.schema import PreparedSeries


def _longest_observed_run(series: PreparedSeries) -> PreparedSeries:
    observed = series.values.notna().to_numpy()
    best_start = best_end = None
    run_start = None
    for position, is_observed in enumerate(observed):
        if is_observed and run_start is None:
            run_start = position
        if run_start is not None and (not is_observed or position == len(observed) - 1):
            run_end = position if is_observed else position - 1
            if best_start is None or run_end - run_start > best_end - best_start:
                best_start, best_end = run_start, run_end
            run_start = None
    if best_start is None or best_end is None:
        raise ValueError("no contiguous observed weekly segment was found")
    values = series.values.iloc[best_start : best_end + 1].copy()
    return PreparedSeries(
        values=values,
        schema=series.schema,
        source_rows=series.source_rows,
        missing_periods=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/car_prices_weekly_median.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/benchmarks/car_prices_weekly_median.meta.local.json"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(
        args.input,
        usecols=["saledate", "sellingprice"],
        low_memory=False,
    )
    cleaned, report = clean_vehicle_sales_source(raw)
    weekly_full = prepare_time_series(cleaned, CAR_PRICES_WEEKLY_MEDIAN)
    selected = _longest_observed_run(weekly_full)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.values.rename("sellingprice").to_csv(args.output, index_label="saledate")

    metadata = {
        "source_cleaning": report.as_dict(),
        "weekly_full_observations": len(weekly_full.values),
        "weekly_full_missing_periods": weekly_full.missing_periods,
        "selected_observations": len(selected.values),
        "selected_start": selected.values.index[0].isoformat(),
        "selected_end": selected.values.index[-1].isoformat(),
        "selection_policy": "longest_contiguous_observed_weekly_segment",
        "imputed_targets": 0,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
