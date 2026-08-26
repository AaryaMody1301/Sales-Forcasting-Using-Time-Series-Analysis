"""Canonical package for leakage-aware sales forecasting."""

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
    "FeatureSpec",
    "GradientBoostingForecaster",
    "LastValueNaiveModel",
    "PreparedSeries",
    "RandomForestForecaster",
    "XGBoostForecaster",
    "build_feature_row",
    "build_leaderboard",
    "build_supervised_frame",
    "calculate_metrics",
    "expanding_window_backtest",
    "get_builtin_schema",
    "prepare_time_series",
]

__version__ = "0.3.0"
