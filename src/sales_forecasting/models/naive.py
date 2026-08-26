"""Deterministic baseline models."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset

from sales_forecasting.data.schema import PreparedSeries

from .base import ForecastModel, ForecastResult


class LastValueNaiveModel(ForecastModel):
    """Forecast every future point as the last observed training value."""

    name = "naive_last_value"

    def __init__(self) -> None:
        self._last_value: float | None = None
        self._last_timestamp: pd.Timestamp | None = None
        self._frequency: str | None = None

    def fit(self, series: PreparedSeries) -> "LastValueNaiveModel":
        self.validate_training_series(series)
        self._last_value = float(series.values.iloc[-1])
        self._last_timestamp = pd.Timestamp(series.values.index[-1])
        self._frequency = series.schema.frequency
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        if (
            self._last_value is None
            or self._last_timestamp is None
            or self._frequency is None
        ):
            raise ValueError("model must be fitted before forecasting")

        offset = to_offset(self._frequency)
        forecast_index = pd.date_range(
            start=self._last_timestamp + offset,
            periods=horizon,
            freq=self._frequency,
        )
        forecast = pd.Series(
            [self._last_value] * horizon,
            index=forecast_index,
            name="forecast",
            dtype=float,
        )

        return ForecastResult(
            model_name=self.name,
            values=forecast,
            frequency=self._frequency,
            fitted_until=self._last_timestamp,
        )

    def save(self, path: Path) -> None:
        if self._last_value is None:
            raise ValueError("model must be fitted before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            pickle.dumps(
                {
                    "last_value": self._last_value,
                    "last_timestamp": self._last_timestamp,
                    "frequency": self._frequency,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> "LastValueNaiveModel":
        state = pickle.loads(Path(path).read_bytes())
        model = cls()
        model._last_value = state["last_value"]
        model._last_timestamp = state["last_timestamp"]
        model._frequency = state["frequency"]
        return model
