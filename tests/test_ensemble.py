from pathlib import Path

import numpy as np
import pandas as pd

from sales_forecasting import DatasetSchema, PreparedSeries, ValidationWeightedEnsemble
from sales_forecasting.models import ForecastModel, ForecastResult, LastValueNaiveModel


class DriftModel(ForecastModel):
    name = "drift"

    def __init__(self):
        self.last = None
        self.drift = None
        self.last_timestamp = None
        self.frequency = None

    def fit(self, series: PreparedSeries):
        self.validate_training_series(series)
        self.last = float(series.values.iloc[-1])
        self.drift = float(series.values.iloc[-1] - series.values.iloc[-2])
        self.last_timestamp = pd.Timestamp(series.values.index[-1])
        self.frequency = series.schema.frequency
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        index = pd.date_range(
            self.last_timestamp + pd.tseries.frequencies.to_offset(self.frequency),
            periods=horizon,
            freq=self.frequency,
        )
        values = pd.Series(
            [self.last + self.drift * step for step in range(1, horizon + 1)],
            index=index,
            dtype=float,
        )
        return ForecastResult(self.name, values, self.frequency, self.last_timestamp)

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path):
        raise NotImplementedError


def make_linear_series() -> PreparedSeries:
    index = pd.date_range("2024-01-01", periods=30, freq="D")
    return PreparedSeries(
        pd.Series(np.arange(1, 31, dtype=float), index=index),
        DatasetSchema("linear", "date", "sales", "D"),
        source_rows=30,
        missing_periods=0,
    )


def test_validation_weighted_ensemble_learns_only_from_inner_validation():
    model = ValidationWeightedEnsemble(
        {"naive": LastValueNaiveModel, "drift": DriftModel},
        validation_initial_train_size=12,
        validation_horizon=2,
        validation_step=2,
        metric="rmse",
    )
    model.fit(make_linear_series())
    result = model.forecast(2)

    assert np.allclose(result.values.to_numpy(), [31.0, 32.0])
    metadata = result.metadata["ensemble"]
    assert metadata["weights"]["drift"] == 1.0
    assert metadata["weights"]["naive"] == 0.0
    assert metadata["validation_scores"]["drift"] == 0.0
