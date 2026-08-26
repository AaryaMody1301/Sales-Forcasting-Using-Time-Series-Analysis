#!/usr/bin/env python3
"""Run the v1 release-candidate benchmark on the real car-auction source data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sales_forecasting.data import clean_vehicle_sales_source
from sales_forecasting.data.catalog import (
    CAR_PRICES_DAILY_MEDIAN,
    CAR_PRICES_WEEKLY_MEDIAN,
)
from sales_forecasting.data.prepare import prepare_time_series
from sales_forecasting.data.schema import PreparedSeries
from sales_forecasting.evaluation import build_leaderboard
from sales_forecasting.evaluation.ensemble import ValidationWeightedEnsemble
from sales_forecasting.features import FeatureSpec
from sales_forecasting.models import (
    ARIMAForecaster,
    ETSForecaster,
    GradientBoostingForecaster,
    LastValueNaiveModel,
    ProphetForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "car_prices.csv"
OUTPUT_DIR = ROOT / "release_benchmark"


def _longest_observed_run(series: PreparedSeries) -> tuple[PreparedSeries, dict[str, object]]:
    """Select the longest contiguous observed block without imputing missing weeks."""

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
        raise RuntimeError("weekly source contains no contiguous observed segment")
    values = series.values.iloc[best_start : best_end + 1].copy()
    if len(values) < 32:
        raise RuntimeError(
            f"longest contiguous weekly segment has only {len(values)} observations; "
            "insufficient for the release benchmark"
        )
    selected = PreparedSeries(
        values=values,
        schema=series.schema,
        source_rows=series.source_rows,
        missing_periods=0,
    )
    metadata = {
        "full_weekly_observations": int(len(series.values)),
        "full_weekly_missing_periods": int(series.missing_periods),
        "selected_observations": int(len(values)),
        "selected_start": values.index[0].isoformat(),
        "selected_end": values.index[-1].isoformat(),
        "selection_policy": "longest_contiguous_observed_weekly_segment",
        "imputed_targets": 0,
    }
    return selected, metadata


def _weekly_feature_spec() -> FeatureSpec:
    return FeatureSpec(
        lags=(1, 2, 4, 8, 13),
        rolling_windows=(4, 8, 13),
        calendar=True,
    )


def _base_factories():
    feature_spec = _weekly_feature_spec()
    return {
        "naive_last_value": LastValueNaiveModel,
        "ets": lambda: ETSForecaster(trend="add"),
        "arima": lambda: ARIMAForecaster(order=(1, 1, 1)),
        "random_forest": lambda: RandomForestForecaster(
            feature_spec=feature_spec,
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=2,
        ),
        "gradient_boosting": lambda: GradientBoostingForecaster(
            feature_spec=feature_spec,
            n_estimators=120,
            max_depth=3,
            learning_rate=0.04,
        ),
        "xgboost": lambda: XGBoostForecaster(
            feature_spec=feature_spec,
            n_estimators=140,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
        ),
        "prophet": lambda: ProphetForecaster(
            uncertainty_samples=0,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality="auto",
        ),
    }


def main() -> int:
    raw = pd.read_csv(
        DATASET,
        usecols=["saledate", "sellingprice"],
        low_memory=False,
    )
    frame, cleaning_report = clean_vehicle_sales_source(raw)

    daily = prepare_time_series(frame, CAR_PRICES_DAILY_MEDIAN)
    weekly_full = prepare_time_series(frame, CAR_PRICES_WEEKLY_MEDIAN)
    weekly, selection = _longest_observed_run(weekly_full)

    horizon = 4
    holdout_periods = min(24, max(12, (len(weekly.values) // 3 // horizon) * horizon))
    initial_train_size = len(weekly.values) - holdout_periods
    if initial_train_size < 24:
        initial_train_size = len(weekly.values) - 2 * horizon
    if initial_train_size < 20 or initial_train_size + horizon > len(weekly.values):
        raise RuntimeError("selected weekly history is too short for release benchmark")

    factories = _base_factories()
    ensemble_validation_start = max(16, initial_train_size - 16)
    if ensemble_validation_start + horizon > initial_train_size:
        ensemble_validation_start = initial_train_size - horizon
    ensemble_members = {
        label: factory
        for label, factory in factories.items()
        if label in {"ets", "arima", "random_forest", "xgboost", "prophet"}
    }
    factories["validation_weighted_ensemble"] = lambda: ValidationWeightedEnsemble(
        ensemble_members,
        validation_initial_train_size=ensemble_validation_start,
        validation_horizon=horizon,
        validation_step=horizon,
        metric="rmse",
    )

    leaderboard = build_leaderboard(
        weekly,
        factories,
        initial_train_size=initial_train_size,
        horizon=horizon,
        step=horizon,
        baseline_model="naive_last_value",
        missing_policy="error",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    leaderboard.table.to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)
    weekly.values.rename("sellingprice").to_csv(
        OUTPUT_DIR / "car_prices_weekly_median.csv",
        index_label="saledate",
    )

    summary = {
        "source_cleaning": cleaning_report.as_dict(),
        "dataset": {
            "daily": {
                "observations": len(daily.values),
                "missing_periods": daily.missing_periods,
                "missing_fraction": daily.missing_periods / len(daily.values),
                "start": daily.values.index[0].isoformat(),
                "end": daily.values.index[-1].isoformat(),
            },
            "weekly_full": {
                "observations": len(weekly_full.values),
                "missing_periods": weekly_full.missing_periods,
                "start": weekly_full.values.index[0].isoformat(),
                "end": weekly_full.values.index[-1].isoformat(),
            },
            "benchmark_segment": selection,
        },
        "evaluation": {
            "initial_train_size": initial_train_size,
            "horizon": horizon,
            "step": horizon,
            "outer_folds": len(next(iter(leaderboard.backtests.values())).folds),
            "ranking_metric": "rmse",
        },
        "leaderboard": json.loads(leaderboard.table.to_json(orient="records")),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
