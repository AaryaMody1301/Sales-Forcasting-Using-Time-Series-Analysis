import pandas as pd

from sales_forecasting.features import FeatureSpec, build_feature_row, build_supervised_frame


def test_training_features_never_include_current_target():
    values = pd.Series(
        [10.0, 20.0, 30.0, 40.0, 50.0],
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
        name="sales",
    )
    spec = FeatureSpec(lags=(1, 2), rolling_windows=(2,), calendar=False)

    X, y = build_supervised_frame(values, spec)

    first_time = pd.Timestamp("2024-01-03")
    assert y.loc[first_time] == 30.0
    assert X.loc[first_time, "lag_1"] == 20.0
    assert X.loc[first_time, "lag_2"] == 10.0
    assert X.loc[first_time, "rolling_mean_2"] == 15.0


def test_forecast_row_uses_only_supplied_history():
    history = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    spec = FeatureSpec(lags=(1, 3), rolling_windows=(3,), calendar=False)

    row = build_feature_row(history, pd.Timestamp("2024-01-04"), spec)

    assert row["lag_1"] == 3.0
    assert row["lag_3"] == 1.0
    assert row["rolling_mean_3"] == 2.0
