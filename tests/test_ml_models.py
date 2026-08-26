import pandas as pd
import pytest

from sales_forecasting.data.schema import DatasetSchema, PreparedSeries
from sales_forecasting.features import FeatureSpec
from sales_forecasting.models.ml import (
    GradientBoostingForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)


def make_series(n=50):
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    values = pd.Series([float(i) for i in range(n)], index=index, name="sales")
    return PreparedSeries(
        values=values,
        schema=DatasetSchema(
            name="synthetic",
            timestamp_col="date",
            target_col="sales",
            frequency="D",
        ),
        source_rows=n,
        missing_periods=0,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RandomForestForecaster(
            feature_spec=FeatureSpec(lags=(1, 2, 3), rolling_windows=(3,)),
            n_estimators=10,
            n_jobs=1,
        ),
        lambda: GradientBoostingForecaster(
            feature_spec=FeatureSpec(lags=(1, 2, 3), rolling_windows=(3,)),
            n_estimators=10,
        ),
        lambda: XGBoostForecaster(
            feature_spec=FeatureSpec(lags=(1, 2, 3), rolling_windows=(3,)),
            n_estimators=10,
            n_jobs=1,
        ),
    ],
)
def test_ml_model_fit_and_recursive_forecast(factory):
    prepared = make_series()
    model = factory().fit(prepared)
    forecast = model.forecast(3)

    assert len(forecast.values) == 3
    assert forecast.values.index[0] == pd.Timestamp("2024-02-20")
    assert forecast.values.index[-1] == pd.Timestamp("2024-02-22")
    assert forecast.values.notna().all()
    assert forecast.metadata["recursive"] is True
