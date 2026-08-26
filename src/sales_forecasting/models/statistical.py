"""Statsmodels adapters implementing the canonical ForecastModel contract."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from sales_forecasting.data.schema import PreparedSeries

from .base import ForecastModel, ForecastResult


class ARIMAForecaster(ForecastModel):
    """ARIMA adapter with fitting separated from external backtesting."""

    name = "arima"

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1), trend=None) -> None:
        self.order = order
        self.trend = trend
        self._result = None
        self._frequency: str | None = None
        self._fitted_until: pd.Timestamp | None = None

    def fit(self, series: PreparedSeries) -> "ARIMAForecaster":
        self.validate_training_series(series)
        values = series.values.astype(float).copy()
        values.index.freq = series.schema.frequency

        self._result = ARIMA(
            values,
            order=self.order,
            trend=self.trend,
        ).fit()
        self._frequency = series.schema.frequency
        self._fitted_until = pd.Timestamp(values.index[-1])
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        if self._result is None or self._frequency is None or self._fitted_until is None:
            raise ValueError("model must be fitted before forecasting")

        values = pd.Series(
            self._result.forecast(steps=horizon),
            dtype=float,
            name="forecast",
        )
        return ForecastResult(
            model_name=self.name,
            values=values,
            frequency=self._frequency,
            fitted_until=self._fitted_until,
            metadata={"order": self.order},
        )

    def save(self, path: Path) -> None:
        if self._result is None:
            raise ValueError("model must be fitted before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "ARIMAForecaster":
        return pickle.loads(Path(path).read_bytes())


class ETSForecaster(ForecastModel):
    """Error-trend-seasonal state-space exponential smoothing adapter."""

    name = "ets"

    def __init__(
        self,
        *,
        error: str = "add",
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        damped_trend: bool = False,
    ) -> None:
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self._result = None
        self._frequency: str | None = None
        self._fitted_until: pd.Timestamp | None = None

    def fit(self, series: PreparedSeries) -> "ETSForecaster":
        self.validate_training_series(series)
        values = series.values.astype(float).copy()
        values.index.freq = series.schema.frequency

        self._result = ETSModel(
            values,
            error=self.error,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
        ).fit(disp=False)
        self._frequency = series.schema.frequency
        self._fitted_until = pd.Timestamp(values.index[-1])
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        if self._result is None or self._frequency is None or self._fitted_until is None:
            raise ValueError("model must be fitted before forecasting")

        values = pd.Series(
            self._result.forecast(steps=horizon),
            dtype=float,
            name="forecast",
        )
        return ForecastResult(
            model_name=self.name,
            values=values,
            frequency=self._frequency,
            fitted_until=self._fitted_until,
            metadata={
                "error": self.error,
                "trend": self.trend,
                "seasonal": self.seasonal,
                "seasonal_periods": self.seasonal_periods,
                "damped_trend": self.damped_trend,
            },
        )

    def save(self, path: Path) -> None:
        if self._result is None:
            raise ValueError("model must be fitted before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "ETSForecaster":
        return pickle.loads(Path(path).read_bytes())
