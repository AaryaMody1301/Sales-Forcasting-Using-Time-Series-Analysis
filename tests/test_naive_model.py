import numpy as np
import pandas as pd

from sales_forecasting import DatasetSchema, LastValueNaiveModel, PreparedSeries


def make_series() -> PreparedSeries:
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    values = pd.Series(np.arange(1, 6, dtype=float), index=index, name="sales")
    schema = DatasetSchema("example", "date", "sales", "D")
    return PreparedSeries(values, schema, source_rows=5, missing_periods=0)


def test_last_value_baseline_forecasts_from_training_endpoint():
    forecast = LastValueNaiveModel().fit(make_series()).forecast(2)

    assert forecast.values.tolist() == [5.0, 5.0]
    assert forecast.values.index[0] == pd.Timestamp("2024-01-06")
    assert forecast.fitted_until == pd.Timestamp("2024-01-05")


def test_last_value_baseline_round_trips(tmp_path):
    model = LastValueNaiveModel().fit(make_series())
    path = tmp_path / "naive.pkl"
    model.save(path)

    restored = LastValueNaiveModel.load(path)
    assert restored.forecast(2).values.equals(model.forecast(2).values)
