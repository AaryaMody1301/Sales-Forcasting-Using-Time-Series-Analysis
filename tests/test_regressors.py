import pandas as pd
import pytest

from sales_forecasting import (
    DatasetContractError,
    DatasetSchema,
    PreparedSeries,
    attach_known_future_regressors,
    fingerprint_prepared_series,
    future_regressors_for_horizon,
)


def make_series() -> PreparedSeries:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    return PreparedSeries(
        values=pd.Series([10, 11, 12, 13, 14, 15], index=index, dtype=float),
        schema=DatasetSchema("example", "date", "sales", "D"),
        source_rows=6,
        missing_periods=0,
    )


def regressor_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="D"),
            "promo": [0, 1, 0, 1, 0, 1, 1, 0],
            "price_index": [100, 101, 100, 99, 98, 99, 97, 98],
        }
    )


def test_attach_known_future_regressors_extends_beyond_target():
    prepared = attach_known_future_regressors(
        make_series(),
        regressor_frame(),
        timestamp_col="date",
        regressor_cols=["promo", "price_index"],
    )

    assert prepared.schema.known_future_regressors == ("promo", "price_index")
    assert prepared.regressor_horizon == 2
    future = future_regressors_for_horizon(prepared, 2)
    assert list(future.index) == list(pd.date_range("2024-01-07", periods=2, freq="D"))
    assert future.iloc[0]["promo"] == 1


def test_regressor_grid_rejects_missing_dates():
    frame = regressor_frame().drop(index=3)
    with pytest.raises(DatasetContractError, match="complete regular grid"):
        attach_known_future_regressors(
            make_series(), frame, timestamp_col="date", regressor_cols=["promo"]
        )


def test_future_regressor_values_are_part_of_dataset_fingerprint():
    first = attach_known_future_regressors(
        make_series(), regressor_frame(), timestamp_col="date", regressor_cols=["promo"]
    )
    changed_frame = regressor_frame()
    changed_frame.loc[7, "promo"] = 1
    second = attach_known_future_regressors(
        make_series(), changed_frame, timestamp_col="date", regressor_cols=["promo"]
    )

    assert fingerprint_prepared_series(first) != fingerprint_prepared_series(second)
