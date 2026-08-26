from pathlib import Path

import numpy as np
import pandas as pd

from sales_forecasting import (
    DatasetSchema,
    NestedTunedForecaster,
    PreparedSeries,
    expanding_window_backtest,
)
from sales_forecasting.models.base import ForecastModel, ForecastResult


class DriftForecaster(ForecastModel):
    name = "drift_test"

    def __init__(self, drift: float = 0.0) -> None:
        self.drift = float(drift)
        self._last_value = None
        self._last_timestamp = None
        self._frequency = None

    def fit(self, series: PreparedSeries):
        self.validate_training_series(series)
        self._last_value = float(series.values.iloc[-1])
        self._last_timestamp = pd.Timestamp(series.values.index[-1])
        self._frequency = series.schema.frequency
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        offset = pd.tseries.frequencies.to_offset(self._frequency)
        index = pd.date_range(
            self._last_timestamp + offset,
            periods=horizon,
            freq=self._frequency,
        )
        values = pd.Series(
            [self._last_value + self.drift * step for step in range(1, horizon + 1)],
            index=index,
            dtype=float,
            name="forecast",
        )
        return ForecastResult(
            model_name=self.name,
            values=values,
            frequency=self._frequency,
            fitted_until=self._last_timestamp,
        )

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path):
        raise NotImplementedError


def make_linear_series(length: int = 18) -> PreparedSeries:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    values = pd.Series(np.arange(1, length + 1, dtype=float), index=index, name="sales")
    schema = DatasetSchema("linear", "date", "sales", "D")
    return PreparedSeries(values, schema, source_rows=length, missing_periods=0)


def test_nested_tuning_selects_parameters_inside_training_history():
    model = NestedTunedForecaster(
        DriftForecaster,
        param_grid={"drift": [0.0, 1.0, 2.0]},
        inner_initial_train_size=6,
        inner_horizon=2,
        metric="rmse",
    )
    model.fit(make_linear_series(12))
    result = model.forecast(2)

    assert result.values.tolist() == [13.0, 14.0]
    assert result.metadata["tuning"]["selected_params"] == {"drift": 1.0}
    assert result.metadata["tuning"]["best_score"] == 0.0


def test_nested_tuning_works_inside_outer_backtest_without_test_leakage():
    result = expanding_window_backtest(
        make_linear_series(),
        lambda: NestedTunedForecaster(
            DriftForecaster,
            param_grid={"drift": [0.0, 1.0, 2.0]},
            inner_initial_train_size=6,
            inner_horizon=2,
            metric="rmse",
        ),
        initial_train_size=12,
        horizon=2,
        step=2,
    )

    assert len(result.folds) == 3
    assert result.aggregate.rmse == 0.0
    assert result.folds[0].metadata["tuning"]["selected_params"] == {"drift": 1.0}
