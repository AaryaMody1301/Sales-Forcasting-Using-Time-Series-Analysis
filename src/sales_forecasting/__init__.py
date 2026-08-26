"""Canonical package for the sales forecasting refactor."""

from .data.catalog import CAR_PRICES_DAILY_MEDIAN, get_builtin_schema
from .data.prepare import prepare_time_series
from .data.schema import DatasetContractError, DatasetSchema, PreparedSeries
from .evaluation import (
    BacktestFold,
    BacktestResult,
    ForecastMetrics,
    calculate_metrics,
    expanding_window_backtest,
)
from .models import ARIMAForecaster, ETSForecaster, LastValueNaiveModel

__all__ = [
    "ARIMAForecaster",
    "BacktestFold",
    "BacktestResult",
    "CAR_PRICES_DAILY_MEDIAN",
    "DatasetContractError",
    "DatasetSchema",
    "ETSForecaster",
    "ForecastMetrics",
    "LastValueNaiveModel",
    "PreparedSeries",
    "calculate_metrics",
    "expanding_window_backtest",
    "get_builtin_schema",
    "prepare_time_series",
]

__version__ = "0.2.0"
