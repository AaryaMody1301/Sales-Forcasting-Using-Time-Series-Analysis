"""Canonical package for leakage-aware, reproducible sales forecasting."""

from .artifacts import (
    ExperimentRun,
    ExperimentSpec,
    ManifestError,
    ModelSpec,
    fingerprint_config,
    fingerprint_prepared_series,
    load_run_manifest,
    record_experiment,
)
from .data.catalog import CAR_PRICES_DAILY_MEDIAN, get_builtin_schema
from .data.prepare import prepare_time_series
from .data.schema import DatasetContractError, DatasetSchema, PreparedSeries
from .evaluation import build_leaderboard, calculate_metrics, expanding_window_backtest
from .features import FeatureSpec, build_feature_row, build_supervised_frame
from .models import (
    ARIMAForecaster,
    ETSForecaster,
    GradientBoostingForecaster,
    LastValueNaiveModel,
    RandomForestForecaster,
    XGBoostForecaster,
)

__all__ = [
    "ARIMAForecaster",
    "CAR_PRICES_DAILY_MEDIAN",
    "DatasetContractError",
    "DatasetSchema",
    "ETSForecaster",
    "ExperimentRun",
    "ExperimentSpec",
    "FeatureSpec",
    "GradientBoostingForecaster",
    "LastValueNaiveModel",
    "ManifestError",
    "ModelSpec",
    "PreparedSeries",
    "RandomForestForecaster",
    "XGBoostForecaster",
    "build_feature_row",
    "build_leaderboard",
    "build_supervised_frame",
    "calculate_metrics",
    "expanding_window_backtest",
    "fingerprint_config",
    "fingerprint_prepared_series",
    "get_builtin_schema",
    "load_run_manifest",
    "prepare_time_series",
    "record_experiment",
]

__version__ = "0.4.0"
