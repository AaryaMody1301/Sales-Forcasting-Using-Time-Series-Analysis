import pandas as pd
import pytest

from sales_forecasting.data.schema import DatasetSchema, PreparedSeries
from sales_forecasting.models.base import ForecastModel, ForecastResult


def test_forecast_result_requires_clean_datetime_output():
    with pytest.raises(ValueError, match="cannot contain missing"):
        ForecastResult(
            model_name="example",
            values=pd.Series(
                [1.0, None],
                index=pd.date_range("2024-01-01", periods=2, freq="D"),
            ),
            frequency="D",
            fitted_until=pd.Timestamp("2023-12-31"),
        )


def test_model_contract_rejects_missing_training_periods():
    prepared = PreparedSeries(
        values=pd.Series(
            [1.0, None, 3.0],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        ),
        schema=DatasetSchema(
            name="example",
            timestamp_col="date",
            target_col="sales",
            frequency="D",
        ),
        source_rows=3,
        missing_periods=1,
    )

    with pytest.raises(ValueError, match="missing periods"):
        ForecastModel.validate_training_series(prepared)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_model_contract_rejects_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="positive integer"):
        ForecastModel.validate_horizon(horizon)
