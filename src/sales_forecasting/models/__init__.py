"""Canonical forecasting model adapters."""

from .base import ForecastModel, ForecastResult
from .ml import GradientBoostingForecaster, RandomForestForecaster, XGBoostForecaster
from .naive import LastValueNaiveModel
from .prophet import ProphetForecaster
from .statistical import ARIMAForecaster, ETSForecaster

__all__ = [
    "ARIMAForecaster",
    "ETSForecaster",
    "ForecastModel",
    "ForecastResult",
    "GradientBoostingForecaster",
    "LastValueNaiveModel",
    "ProphetForecaster",
    "RandomForestForecaster",
    "XGBoostForecaster",
]
