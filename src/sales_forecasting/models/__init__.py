"""Canonical forecasting model adapters."""

from .base import ForecastModel, ForecastResult
from .ml import GradientBoostingForecaster, RandomForestForecaster, XGBoostForecaster
from .naive import LastValueNaiveModel
from .statistical import ARIMAForecaster, ETSForecaster

__all__ = [
    "ARIMAForecaster",
    "ETSForecaster",
    "ForecastModel",
    "ForecastResult",
    "GradientBoostingForecaster",
    "LastValueNaiveModel",
    "RandomForestForecaster",
    "XGBoostForecaster",
]
