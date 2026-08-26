import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prophet")

from sales_forecasting import (
    DatasetContractError,
    DatasetSchema,
    PreparedSeries,
    ProphetForecaster,
    attach_known_future_regressors,
    expanding_window_backtest,
)


def make_series(target_periods=30, regressor_periods=34):
    index = pd.date_range("2024-01-01", periods=target_periods, freq="D")
    promo = np.asarray([(i % 3) == 0 for i in range(regressor_periods)], dtype=float)
    values = pd.Series(
        10.0 + 0.4 * np.arange(target_periods) + 2.0 * promo[:target_periods],
        index=index,
        dtype=float,
    )
    base = PreparedSeries(
        values,
        DatasetSchema("prophet_example", "date", "sales", "D"),
        source_rows=target_periods,
        missing_periods=0,
    )
    regressors = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=regressor_periods, freq="D"),
            "promo": promo,
        }
    )
    return attach_known_future_regressors(
        base, regressors, timestamp_col="date", regressor_cols=["promo"]
    )


def prophet_factory():
    return ProphetForecaster(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    )


def test_prophet_uses_known_future_regressors_for_forecast():
    model = prophet_factory().fit(make_series())
    result = model.forecast(4)
    assert len(result.values) == 4
    assert np.isfinite(result.values.to_numpy()).all()
    assert result.metadata["known_future_regressors"] == ["promo"]


def test_prophet_refuses_missing_future_regressor_values():
    model = prophet_factory().fit(make_series(regressor_periods=31))
    with pytest.raises(DatasetContractError, match="every forecast date"):
        model.forecast(2)


def test_prophet_backtest_can_see_known_covariates_but_not_holdout_targets():
    prepared = make_series(target_periods=24, regressor_periods=28)
    result = expanding_window_backtest(
        prepared,
        prophet_factory,
        initial_train_size=20,
        horizon=2,
        step=2,
    )
    assert len(result.folds) == 2
    assert all(np.isfinite(fold.forecast.to_numpy()).all() for fold in result.folds)


def test_prophet_official_json_serialization_round_trip(tmp_path):
    prepared = make_series()
    model = prophet_factory().fit(prepared)
    path = tmp_path / "prophet.json"
    model.save(path)
    restored = ProphetForecaster.load(path)
    assert np.allclose(model.forecast(2).values, restored.forecast(2).values)
