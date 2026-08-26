"""Autoregressive tree/boosting model adapters using leakage-safe features."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from sales_forecasting.data.schema import DatasetContractError, PreparedSeries
from sales_forecasting.features import FeatureSpec, build_feature_row, build_supervised_frame

from .base import ForecastModel, ForecastResult


class _AutoregressiveRegressorForecaster(ForecastModel):
    """Shared recursive forecast logic for regressors trained on target history."""

    estimator_class: type

    def __init__(
        self,
        *,
        feature_spec: FeatureSpec | None = None,
        random_state: int = 42,
        **estimator_params: Any,
    ) -> None:
        self.feature_spec = feature_spec or FeatureSpec()
        self.random_state = random_state
        self.estimator_params = dict(estimator_params)
        self._model = None
        self._history: pd.Series | None = None
        self._frequency: str | None = None
        self._fitted_until: pd.Timestamp | None = None
        self._feature_columns: tuple[str, ...] | None = None

    def _make_estimator(self):
        params = {"random_state": self.random_state, **self.estimator_params}
        return self.estimator_class(**params)

    def fit(self, series: PreparedSeries):
        self.validate_training_series(series)
        if series.schema.known_future_regressors:
            raise DatasetContractError(
                "Phase 3 autoregressive models do not yet consume known future regressors"
            )

        values = series.values.astype(float).copy()
        X, y = build_supervised_frame(values, self.feature_spec)

        self._model = self._make_estimator()
        self._model.fit(X, y)
        self._history = values.copy()
        self._frequency = series.schema.frequency
        self._fitted_until = pd.Timestamp(values.index[-1])
        self._feature_columns = tuple(X.columns)
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        if (
            self._model is None
            or self._history is None
            or self._frequency is None
            or self._fitted_until is None
            or self._feature_columns is None
        ):
            raise ValueError("model must be fitted before forecasting")

        history = self._history.copy()
        offset = to_offset(self._frequency)
        forecasts: list[float] = []
        forecast_index: list[pd.Timestamp] = []
        next_timestamp = pd.Timestamp(history.index[-1]) + offset

        for _ in range(horizon):
            row = build_feature_row(history, next_timestamp, self.feature_spec)
            row = row.loc[list(self._feature_columns)]
            X_next = row.to_frame().T
            prediction = float(self._model.predict(X_next)[0])

            forecasts.append(prediction)
            forecast_index.append(next_timestamp)
            history.loc[next_timestamp] = prediction
            next_timestamp = next_timestamp + offset

        values = pd.Series(
            forecasts,
            index=pd.DatetimeIndex(forecast_index),
            name="forecast",
            dtype=float,
        )
        return ForecastResult(
            model_name=self.name,
            values=values,
            frequency=self._frequency,
            fitted_until=self._fitted_until,
            metadata={
                "feature_spec": {
                    "lags": self.feature_spec.lags,
                    "rolling_windows": self.feature_spec.rolling_windows,
                    "calendar": self.feature_spec.calendar,
                },
                "recursive": True,
                "random_state": self.random_state,
            },
        )

    def save(self, path: Path) -> None:
        if self._model is None:
            raise ValueError("model must be fitted before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path):
        model = pickle.loads(Path(path).read_bytes())
        if not isinstance(model, cls):
            raise TypeError(f"serialized model is not a {cls.__name__}")
        return model


class RandomForestForecaster(_AutoregressiveRegressorForecaster):
    name = "random_forest"
    estimator_class = RandomForestRegressor

    def __init__(self, *, feature_spec=None, random_state=42, **estimator_params):
        defaults = {"n_estimators": 300, "n_jobs": -1}
        defaults.update(estimator_params)
        super().__init__(
            feature_spec=feature_spec,
            random_state=random_state,
            **defaults,
        )


class GradientBoostingForecaster(_AutoregressiveRegressorForecaster):
    name = "gradient_boosting"
    estimator_class = GradientBoostingRegressor

    def __init__(self, *, feature_spec=None, random_state=42, **estimator_params):
        defaults = {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3}
        defaults.update(estimator_params)
        super().__init__(
            feature_spec=feature_spec,
            random_state=random_state,
            **defaults,
        )


class XGBoostForecaster(_AutoregressiveRegressorForecaster):
    name = "xgboost"
    estimator_class = XGBRegressor

    def __init__(self, *, feature_spec=None, random_state=42, **estimator_params):
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }
        defaults.update(estimator_params)
        super().__init__(
            feature_spec=feature_spec,
            random_state=random_state,
            **defaults,
        )
