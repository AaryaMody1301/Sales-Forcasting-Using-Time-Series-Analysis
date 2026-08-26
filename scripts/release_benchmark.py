#!/usr/bin/env python3
"""Run the v1 release-candidate benchmark on the bundled real car-auction data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sales_forecasting.data.catalog import (
    CAR_PRICES_DAILY_MEDIAN,
    CAR_PRICES_WEEKLY_MEDIAN,
)
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
DATASET = ROOT / "data" / "car_prices.csv"
OUTPUT_DIR = ROOT / "release_benchmark"


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
    frame = pd.read_csv(
        DATASET,
        usecols=["saledate", "sellingprice"],
        low_memory=False,
    )

    daily = prepare_time_series(frame, CAR_PRICES_DAILY_MEDIAN)
    weekly = prepare_time_series(frame, CAR_PRICES_WEEKLY_MEDIAN)

    if weekly.missing_periods:
        missing_examples = [
            timestamp.isoformat()
            for timestamp in weekly.values.index[weekly.values.isna()][:5]
        ]
        raise RuntimeError(
            "weekly release benchmark contains missing target periods: "
            + ", ".join(missing_examples)
        )

    horizon = 4
    holdout_periods = 24
    initial_train_size = max(40, len(weekly.values) - holdout_periods)
    if initial_train_size + horizon > len(weekly.values):
        initial_train_size = len(weekly.values) - horizon
    if initial_train_size < 32:
        raise RuntimeError("weekly car-price history is too short for release benchmark")

    factories = _base_factories()
    ensemble_validation_start = max(24, initial_train_size - 20)
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

    summary = {
        "dataset": {
            "raw_rows": len(frame),
            "daily": {
                "observations": len(daily.values),
                "missing_periods": daily.missing_periods,
                "missing_fraction": daily.missing_periods / len(daily.values),
                "start": daily.values.index[0].isoformat(),
                "end": daily.values.index[-1].isoformat(),
            },
            "weekly": {
                "observations": len(weekly.values),
                "missing_periods": weekly.missing_periods,
                "start": weekly.values.index[0].isoformat(),
                "end": weekly.values.index[-1].isoformat(),
            },
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
