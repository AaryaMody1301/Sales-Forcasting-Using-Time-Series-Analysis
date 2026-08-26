"""Prophet adapter with an explicit known-future-regressor contract."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.tseries.frequencies import to_offset

from sales_forecasting.data.schema import DatasetContractError, PreparedSeries
from .base import ForecastModel, ForecastResult


def _prophet_api():
    try:
        from prophet import Prophet
        from prophet.serialize import model_from_json, model_to_json
    except ImportError as exc:
        raise DatasetContractError(
            "Prophet is not installed. Install the optional dependency with "
            "`python -m pip install -e '.[prophet]'`."
        ) from exc
    return Prophet, model_to_json, model_from_json


def _naive_datetimes(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index
    return index.tz_convert("UTC").tz_localize(None)


class ProphetForecaster(ForecastModel):
    """Prophet model that never guesses future regressor values."""

    name = "prophet"

    def __init__(
        self,
        *,
        growth: str = "linear",
        seasonality_mode: str = "additive",
        yearly_seasonality: str | bool | int = "auto",
        weekly_seasonality: str | bool | int = "auto",
        daily_seasonality: str | bool | int = "auto",
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.8,
        uncertainty_samples: int = 1000,
        regressor_standardize: str | bool = "auto",
        regressor_mode: str = "additive",
    ) -> None:
        if growth not in {"linear", "flat"}:
            raise ValueError("Prophet adapter supports growth='linear' or 'flat'")
        if seasonality_mode not in {"additive", "multiplicative"}:
            raise ValueError("seasonality_mode must be additive or multiplicative")
        if regressor_mode not in {"additive", "multiplicative"}:
            raise ValueError("regressor_mode must be additive or multiplicative")
        if not 0 < interval_width < 1:
            raise ValueError("interval_width must be between 0 and 1")
        if uncertainty_samples < 0:
            raise ValueError("uncertainty_samples cannot be negative")
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.seasonality_prior_scale = float(seasonality_prior_scale)
        self.holidays_prior_scale = float(holidays_prior_scale)
        self.interval_width = float(interval_width)
        self.uncertainty_samples = int(uncertainty_samples)
        self.regressor_standardize = regressor_standardize
        self.regressor_mode = regressor_mode
        self._model = None
        self._frequency: str | None = None
        self._fitted_until: pd.Timestamp | None = None
        self._regressor_columns: tuple[str, ...] = ()
        self._future_regressors: pd.DataFrame | None = None

    def _config(self) -> dict[str, Any]:
        return {
            "growth": self.growth,
            "seasonality_mode": self.seasonality_mode,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "interval_width": self.interval_width,
            "uncertainty_samples": self.uncertainty_samples,
            "regressor_standardize": self.regressor_standardize,
            "regressor_mode": self.regressor_mode,
        }

    def fit(self, series: PreparedSeries) -> "ProphetForecaster":
        self.validate_training_series(series)
        Prophet, _, _ = _prophet_api()
        regressor_columns = tuple(series.schema.known_future_regressors)
        if regressor_columns and series.future_regressors is None:
            raise DatasetContractError(
                "Prophet regressors are declared but no known future values were supplied"
            )
        model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            interval_width=self.interval_width,
            uncertainty_samples=self.uncertainty_samples,
        )
        training = pd.DataFrame(
            {"ds": _naive_datetimes(series.values.index), "y": series.values.to_numpy(dtype=float)}
        )
        if regressor_columns:
            history_regressors = series.future_regressors.loc[
                series.values.index, list(regressor_columns)
            ]
            for column in regressor_columns:
                if history_regressors[column].nunique(dropna=False) <= 1:
                    raise DatasetContractError(
                        f"Prophet regressor {column!r} is constant in the training history"
                    )
                model.add_regressor(
                    column,
                    standardize=self.regressor_standardize,
                    mode=self.regressor_mode,
                )
                training[column] = history_regressors[column].to_numpy(dtype=float)
        model.fit(training)
        self._model = model
        self._frequency = series.schema.frequency
        self._fitted_until = pd.Timestamp(series.values.index[-1])
        self._regressor_columns = regressor_columns
        self._future_regressors = None if series.future_regressors is None else series.future_regressors.copy()
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        if self._model is None or self._frequency is None or self._fitted_until is None:
            raise ValueError("model must be fitted before forecasting")
        offset = to_offset(self._frequency)
        future_index = pd.date_range(
            start=self._fitted_until + offset, periods=horizon, freq=self._frequency
        )
        future = pd.DataFrame({"ds": _naive_datetimes(future_index)})
        if self._regressor_columns:
            if self._future_regressors is None:
                raise DatasetContractError("known future regressor values are unavailable")
            missing = future_index.difference(self._future_regressors.index)
            if len(missing):
                sample = ", ".join(str(value) for value in missing[:3])
                raise DatasetContractError(
                    "Prophet requires known regressor values for every forecast date; missing: " + sample
                )
            future_regressors = self._future_regressors.loc[
                future_index, list(self._regressor_columns)
            ]
            for column in self._regressor_columns:
                future[column] = future_regressors[column].to_numpy(dtype=float)
        predicted = self._model.predict(future)
        values = pd.Series(
            predicted["yhat"].to_numpy(dtype=float),
            index=future_index,
            name="forecast",
            dtype=float,
        )
        return ForecastResult(
            model_name=self.name,
            values=values,
            frequency=self._frequency,
            fitted_until=self._fitted_until,
            metadata={"known_future_regressors": list(self._regressor_columns), "prophet": self._config()},
        )

    def save(self, path: Path) -> None:
        if self._model is None or self._frequency is None or self._fitted_until is None:
            raise ValueError("model must be fitted before saving")
        _, model_to_json, _ = _prophet_api()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        regressors_json = None
        if self._future_regressors is not None:
            regressors_json = self._future_regressors.to_json(
                orient="table", date_format="iso", index=True
            )
        payload = {
            "schema_version": 1,
            "config": self._config(),
            "frequency": self._frequency,
            "fitted_until": self._fitted_until.isoformat(),
            "regressor_columns": list(self._regressor_columns),
            "future_regressors": regressors_json,
            "model": model_to_json(self._model),
        }
        path.write_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ProphetForecaster":
        _, _, model_from_json = _prophet_api()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported Prophet serialization version")
        model = cls(**payload["config"])
        model._model = model_from_json(payload["model"])
        model._frequency = str(payload["frequency"])
        model._fitted_until = pd.Timestamp(payload["fitted_until"])
        model._regressor_columns = tuple(payload["regressor_columns"])
        if payload.get("future_regressors") is not None:
            frame = pd.read_json(StringIO(payload["future_regressors"]), orient="table")
            if not isinstance(frame.index, pd.DatetimeIndex):
                frame.index = pd.to_datetime(frame.index)
            model._future_regressors = frame
        return model
