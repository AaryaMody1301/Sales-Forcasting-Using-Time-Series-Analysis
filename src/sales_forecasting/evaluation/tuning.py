"""Nested chronological hyperparameter tuning for canonical forecasters."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Mapping

from sklearn.model_selection import ParameterGrid

from sales_forecasting.data.schema import PreparedSeries
from sales_forecasting.models.base import ForecastModel, ForecastResult

from .backtesting import expanding_window_backtest

_METRICS = {"mae", "rmse", "smape", "mase", "wape"}


class NestedTunedForecaster(ForecastModel):
    """Tune a model only inside the training data supplied by an outer evaluator.

    Each ``fit`` call performs an inner expanding-window search on the supplied
    training series. The selected candidate is then refit on all of that training
    history. An outer backtester can therefore evaluate this wrapper without any
    information from the outer holdout leaking into parameter selection.
    """

    def __init__(
        self,
        model_class: type[ForecastModel],
        *,
        param_grid: Mapping[str, list[Any]],
        inner_initial_train_size: int,
        inner_horizon: int = 1,
        inner_step: int | None = None,
        metric: str = "rmse",
        base_params: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model_class, type) or not issubclass(model_class, ForecastModel):
            raise TypeError("model_class must be a ForecastModel subclass")
        if inner_initial_train_size < 3:
            raise ValueError("inner_initial_train_size must be at least 3")
        if inner_horizon < 1:
            raise ValueError("inner_horizon must be positive")
        if inner_step is not None and inner_step < 1:
            raise ValueError("inner_step must be positive when provided")
        if metric not in _METRICS:
            raise ValueError(f"unsupported tuning metric {metric!r}")

        grid = {str(key): list(values) for key, values in param_grid.items()}
        if not grid:
            raise ValueError("param_grid must contain at least one parameter")
        if any(not values for values in grid.values()):
            raise ValueError("each param_grid entry must contain at least one value")

        self.model_class = model_class
        self.param_grid = grid
        self.inner_initial_train_size = inner_initial_train_size
        self.inner_horizon = inner_horizon
        self.inner_step = inner_step
        self.metric = metric
        self.base_params = dict(base_params or {})

        prototype = self.model_class(**self.base_params)
        self.name = f"tuned_{prototype.name}"
        self._selected_model: ForecastModel | None = None
        self._selected_params: dict[str, Any] | None = None
        self._best_score: float | None = None
        self._candidate_scores: list[dict[str, Any]] = []

    def _make_model(self, params: Mapping[str, Any]) -> ForecastModel:
        merged = {**self.base_params, **dict(params)}
        model = self.model_class(**merged)
        if not isinstance(model, ForecastModel):
            raise TypeError("model_class did not construct a ForecastModel")
        return model

    def fit(self, series: PreparedSeries) -> "NestedTunedForecaster":
        self.validate_training_series(series)
        if self.inner_initial_train_size + self.inner_horizon > len(series.values):
            raise ValueError(
                "training series is too short for the requested inner tuning split"
            )

        best_params: dict[str, Any] | None = None
        best_score = float("inf")
        candidate_scores: list[dict[str, Any]] = []

        for candidate in ParameterGrid(self.param_grid):
            result = expanding_window_backtest(
                series,
                lambda candidate=dict(candidate): self._make_model(candidate),
                initial_train_size=self.inner_initial_train_size,
                horizon=self.inner_horizon,
                step=self.inner_step,
            )
            score = float(getattr(result.aggregate, self.metric))
            candidate_scores.append({"params": dict(candidate), "score": score})
            if math.isfinite(score) and score < best_score:
                best_score = score
                best_params = dict(candidate)

        if best_params is None:
            raise ValueError("no hyperparameter candidate produced a finite tuning score")

        selected = self._make_model(best_params)
        selected.fit(series)
        self._selected_model = selected
        self._selected_params = best_params
        self._best_score = best_score
        self._candidate_scores = candidate_scores
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        if self._selected_model is None or self._selected_params is None:
            raise ValueError("model must be fitted before forecasting")

        result = self._selected_model.forecast(horizon)
        metadata = dict(result.metadata)
        metadata["tuning"] = {
            "metric": self.metric,
            "best_score": self._best_score,
            "selected_params": self._selected_params,
            "inner_initial_train_size": self.inner_initial_train_size,
            "inner_horizon": self.inner_horizon,
            "inner_step": self.inner_horizon if self.inner_step is None else self.inner_step,
            "candidate_scores": self._candidate_scores,
        }
        return ForecastResult(
            model_name=self.name,
            values=result.values.copy(),
            frequency=result.frequency,
            fitted_until=result.fitted_until,
            metadata=metadata,
        )

    def save(self, path: Path) -> None:
        if self._selected_model is None:
            raise ValueError("model must be fitted before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: Path) -> "NestedTunedForecaster":
        model = pickle.loads(Path(path).read_bytes())
        if not isinstance(model, cls):
            raise TypeError(f"serialized model is not a {cls.__name__}")
        return model
