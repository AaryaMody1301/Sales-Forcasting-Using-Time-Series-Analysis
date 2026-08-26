import pandas as pd

from sales_forecasting.data.schema import DatasetSchema, PreparedSeries
from sales_forecasting.evaluation import build_leaderboard
from sales_forecasting.features import FeatureSpec
from sales_forecasting.models import LastValueNaiveModel, RandomForestForecaster


def test_leaderboard_uses_same_number_of_folds_and_marks_baseline():
    index = pd.date_range("2024-01-01", periods=36, freq="D")
    values = pd.Series([10.0 + (i % 4) for i in range(36)], index=index, name="sales")
    prepared = PreparedSeries(
        values=values,
        schema=DatasetSchema(
            name="synthetic",
            timestamp_col="date",
            target_col="sales",
            frequency="D",
        ),
        source_rows=36,
        missing_periods=0,
    )

    result = build_leaderboard(
        prepared,
        {
            "naive_last_value": LastValueNaiveModel,
            "rf": lambda: RandomForestForecaster(
                feature_spec=FeatureSpec(lags=(1, 2, 3), rolling_windows=(3,)),
                n_estimators=10,
                n_jobs=1,
            ),
        },
        initial_train_size=18,
        horizon=2,
    )

    assert set(result.table["model"]) == {"naive_last_value", "rf"}
    assert result.backtests["naive_last_value"].folds
    assert len(result.backtests["naive_last_value"].folds) == len(result.backtests["rf"].folds)
    baseline = result.table[result.table["model"] == "naive_last_value"].iloc[0]
    assert baseline["beats_baseline"] == False
