#!/usr/bin/env python3
"""Run the deterministic v1 acceptance benchmark on reviewed weekly data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sales_forecasting.data.catalog import CAR_PRICES_WEEKLY_MEDIAN
from sales_forecasting.data.prepare import prepare_time_series
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
DATASET = ROOT / "data" / "benchmarks" / "car_prices_weekly_median.csv"
METADATA = ROOT / "data" / "benchmarks" / "car_prices_weekly_median.meta.json"
OUTPUT_DIR = ROOT / "release_benchmark"


def _feature_spec() -> FeatureSpec:
    return FeatureSpec(lags=(1, 2, 4, 8, 13), rolling_windows=(4, 8, 13), calendar=True)


def _factories():
    feature_spec = _feature_spec()
    factories = {
        "naive_last_value": LastValueNaiveModel,
        "ets": lambda: ETSForecaster(trend="add"),
        "arima": lambda: ARIMAForecaster(order=(1, 1, 1)),
        "random_forest": lambda: RandomForestForecaster(
            feature_spec=feature_spec, n_estimators=120, max_depth=8, min_samples_leaf=2
        ),
        "gradient_boosting": lambda: GradientBoostingForecaster(
            feature_spec=feature_spec, n_estimators=120, max_depth=3, learning_rate=0.04
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
    members = {
        label: factory
        for label, factory in factories.items()
        if label in {"ets", "arima", "random_forest", "xgboost", "prophet"}
    }
    factories["validation_weighted_ensemble"] = lambda: ValidationWeightedEnsemble(
        members,
        validation_initial_train_size=16,
        validation_horizon=4,
        validation_step=4,
        metric="rmse",
    )
    return factories


def main() -> int:
    frame = pd.read_csv(DATASET)
    series = prepare_time_series(frame, CAR_PRICES_WEEKLY_MEDIAN)
    if len(series.values) != 32 or series.missing_periods:
        raise RuntimeError("reviewed v1 benchmark must contain exactly 32 complete weekly observations")

    leaderboard = build_leaderboard(
        series,
        _factories(),
        initial_train_size=24,
        horizon=4,
        step=4,
        baseline_model="naive_last_value",
        missing_policy="error",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    leaderboard.table.to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)
    source_metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    summary = {
        "benchmark": source_metadata,
        "evaluation": {
            "initial_train_size": 24,
            "horizon": 4,
            "step": 4,
            "folds": 2,
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
