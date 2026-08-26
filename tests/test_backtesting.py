import numpy as np
import pandas as pd
import pytest

from sales_forecasting import (
    DatasetContractError,
    DatasetSchema,
    LastValueNaiveModel,
    PreparedSeries,
    expanding_window_backtest,
)


def make_series(length: int = 12) -> PreparedSeries:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    values = pd.Series(
        np.arange(1, length + 1, dtype=float),
        index=index,
        name="sales",
    )
    schema = DatasetSchema(
        name="example",
        timestamp_col="date",
        target_col="sales",
        frequency="D",
    )
    return PreparedSeries(values, schema, source_rows=length, missing_periods=0)


def test_expanding_folds_never_train_on_their_test_window():
    result = expanding_window_backtest(
        make_series(),
        LastValueNaiveModel,
        initial_train_size=6,
        horizon=2,
        step=2,
    )

    assert len(result.folds) == 3
    assert result.folds[0].train_end == pd.Timestamp("2024-01-06")
    assert result.folds[0].test_start == pd.Timestamp("2024-01-07")
    assert result.folds[1].train_end == pd.Timestamp("2024-01-08")
    assert result.folds[1].test_start == pd.Timestamp("2024-01-09")


def test_backtest_rejects_unresolved_missing_periods():
    series = make_series()
    values = series.values.copy()
    values.iloc[4] = np.nan
    incomplete = PreparedSeries(
        values,
        series.schema,
        source_rows=len(values),
        missing_periods=1,
    )

    with pytest.raises(DatasetContractError, match="training-only preprocessing"):
        expanding_window_backtest(
            incomplete,
            LastValueNaiveModel,
            initial_train_size=6,
            horizon=2,
        )


def test_backtest_requires_enough_history():
    with pytest.raises(ValueError, match="too short"):
        expanding_window_backtest(
            make_series(6),
            LastValueNaiveModel,
            initial_train_size=5,
            horizon=2,
        )
