import numpy as np
import pandas as pd
import pytest

from sales_forecasting import (
    DatasetContractError,
    DatasetSchema,
    ExperimentSpec,
    LastValueNaiveModel,
    PreparedSeries,
    apply_training_missing_policy,
    expanding_window_backtest,
)


def make_series(values) -> PreparedSeries:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    series = pd.Series(values, index=index, name="sales", dtype=float)
    schema = DatasetSchema(
        name="example",
        timestamp_col="date",
        target_col="sales",
        frequency="D",
    )
    return PreparedSeries(
        series,
        schema,
        source_rows=len(series),
        missing_periods=int(series.isna().sum()),
    )


def test_forward_fill_uses_only_previous_observation():
    prepared = make_series([1, 2, np.nan, 4, 5])
    filled = apply_training_missing_policy(prepared, "forward_fill")
    assert filled.values.iloc[2] == 2.0
    assert filled.missing_periods == 0


def test_forward_fill_rejects_leading_gap():
    prepared = make_series([np.nan, 2, 3, 4])
    with pytest.raises(DatasetContractError, match="leading missing"):
        apply_training_missing_policy(prepared, "forward_fill")


def test_backtest_forward_fills_training_only():
    prepared = make_series([1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10])
    result = expanding_window_backtest(
        prepared,
        LastValueNaiveModel,
        initial_train_size=6,
        horizon=2,
        step=2,
        missing_policy="forward_fill",
    )
    assert len(result.folds) == 2


def test_backtest_never_imputes_test_targets():
    prepared = make_series([1, 2, 3, 4, 5, 6, np.nan, 8, 9, 10])
    with pytest.raises(DatasetContractError, match="test window contains missing targets"):
        expanding_window_backtest(
            prepared,
            LastValueNaiveModel,
            initial_train_size=6,
            horizon=2,
            missing_policy="forward_fill",
        )


def test_missing_policy_is_part_of_experiment_configuration():
    strict = ExperimentSpec(initial_train_size=6, missing_policy="error").as_dict()
    causal = ExperimentSpec(initial_train_size=6, missing_policy="forward_fill").as_dict()
    assert strict["missing_policy"] == "error"
    assert causal["missing_policy"] == "forward_fill"
    assert strict != causal
