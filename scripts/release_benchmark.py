#!/usr/bin/env python3
"""Run and verify the deterministic v1 acceptance benchmark."""

from __future__ import annotations

import json
import math
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
EXPECTED = ROOT / "data" / "benchmarks" / "release_v1_expected.json"
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


def _verify_expected(table: pd.DataFrame) -> dict[str, object]:
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    if expected.get("schema_version") != 1:
        raise RuntimeError("unsupported release benchmark expectation schema")

    expected_models = expected.get("models")
    if not isinstance(expected_models, dict) or not expected_models:
        raise RuntimeError("release benchmark expectation file has no models")

    actual = table.set_index("model")
    expected_names = set(expected_models)
    actual_names = set(actual.index.astype(str))
    if actual_names != expected_names:
        raise RuntimeError(
            "release benchmark model set changed: "
            f"expected {sorted(expected_names)}, got {sorted(actual_names)}"
        )

    tolerance = float(expected.get("rmse_absolute_tolerance", 0.0))
    for model_name, specification in expected_models.items():
        row = actual.loc[model_name]
        expected_rank = int(specification["rank"])
        actual_rank = int(row["rank"])
        if actual_rank != expected_rank:
            raise RuntimeError(
                f"release benchmark rank changed for {model_name}: "
                f"expected {expected_rank}, got {actual_rank}"
            )

        expected_rmse = float(specification["mean_fold_rmse"])
        actual_rmse = float(row["rmse"])
        if not math.isclose(actual_rmse, expected_rmse, rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(
                f"release benchmark RMSE changed for {model_name}: "
                f"expected {expected_rmse:.10f} +/- {tolerance}, got {actual_rmse:.10f}"
            )

    return {
        "expected_file": str(EXPECTED.relative_to(ROOT)),
        "metric_definition": expected["metric_definition"],
        "rmse_absolute_tolerance": tolerance,
        "passed": True,
    }


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
    verification = _verify_expected(leaderboard.table)

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
            "aggregate_metric_definition": "arithmetic mean of per-fold metrics",
        },
        "verification": verification,
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
