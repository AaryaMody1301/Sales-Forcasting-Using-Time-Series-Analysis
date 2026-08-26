import pandas as pd
import pytest

from sales_forecasting import (
    CAR_PRICES_DAILY_MEDIAN,
    DatasetContractError,
    DatasetSchema,
    get_builtin_schema,
    prepare_time_series,
)


def test_car_prices_contract_aggregates_to_daily_median_and_preserves_gap():
    frame = pd.DataFrame(
        {
            "saledate": ["2024-01-01", "2024-01-01", "2024-01-03"],
            "sellingprice": [10_000, 20_000, 30_000],
        }
    )

    prepared = prepare_time_series(frame, CAR_PRICES_DAILY_MEDIAN)

    assert prepared.values.loc["2024-01-01"] == 15_000
    assert pd.isna(prepared.values.loc["2024-01-02"])
    assert prepared.values.loc["2024-01-03"] == 30_000
    assert prepared.missing_periods == 1
    assert prepared.source_rows == 3


def test_amazon_product_dataset_is_not_registered_as_forecasting_dataset():
    with pytest.raises(DatasetContractError, match="not a time-series sales dataset"):
        get_builtin_schema("amazon")


def test_missing_timestamp_column_is_rejected_instead_of_inventing_dates():
    frame = pd.DataFrame({"sales": [1, 2, 3]})
    schema = DatasetSchema(
        name="sales",
        timestamp_col="date",
        target_col="sales",
        frequency="D",
    )

    with pytest.raises(DatasetContractError, match="missing required columns: date"):
        prepare_time_series(frame, schema)


def test_duplicate_timestamps_require_explicit_aggregation():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "sales": [1, 2],
        }
    )
    schema = DatasetSchema(
        name="sales",
        timestamp_col="date",
        target_col="sales",
        frequency="D",
    )

    with pytest.raises(DatasetContractError, match="explicit aggregation"):
        prepare_time_series(frame, schema)


def test_non_numeric_target_is_rejected():
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "sales": ["high", "low"],
        }
    )
    schema = DatasetSchema(
        name="sales",
        timestamp_col="date",
        target_col="sales",
        frequency="D",
    )

    with pytest.raises(DatasetContractError, match="must be numeric"):
        prepare_time_series(frame, schema)
