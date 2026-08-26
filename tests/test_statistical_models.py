import numpy as np
import pandas as pd

from sales_forecasting import (
    ARIMAForecaster,
    DatasetSchema,
    ETSForecaster,
    PreparedSeries,
)


def make_series(length: int = 36) -> PreparedSeries:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    # Smooth deterministic trend keeps smoke tests fast and reproducible.
    values = pd.Series(
        100.0 + np.arange(length, dtype=float) * 0.5,
        index=index,
        name="sales",
    )
    schema = DatasetSchema("example", "date", "sales", "D")
    return PreparedSeries(values, schema, source_rows=length, missing_periods=0)


def test_arima_adapter_returns_canonical_forecast():
    forecast = ARIMAForecaster(order=(1, 1, 0)).fit(make_series()).forecast(3)

    assert len(forecast.values) == 3
    assert forecast.values.index[0] == pd.Timestamp("2024-02-06")
    assert forecast.model_name == "arima"


def test_ets_adapter_returns_canonical_forecast():
    forecast = ETSForecaster(trend="add").fit(make_series()).forecast(3)

    assert len(forecast.values) == 3
    assert forecast.values.index[0] == pd.Timestamp("2024-02-06")
    assert forecast.model_name == "ets"
