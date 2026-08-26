"""Validation-derived ensembles for leakage-safe forecast combination."""

from __future__ import annotations

import importlib
import json
import math
import re
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from sales_forecasting.data.schema import PreparedSeries
from sales_forecasting.models.base import ForecastModel, ForecastResult

from .backtesting import expanding_window_backtest

ModelFactory = Callable[[], ForecastModel]
_METRICS = {"mae", "rmse", "smape", "mase", "wape"}


def _safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return label or "model"


class ValidationWeightedEnsemble(ForecastModel):
    """Combine base forecasters using weights learned from inner validation only."""

    name = "validation_weighted_ensemble"

    def __init__(
        self,
        model_factories: Mapping[str, ModelFactory],
        *,
        validation_initial_train_size: int,
        validation_horizon: int = 1,
        validation_step: int | None = None,
        metric: str = "rmse",
        weight_power: float = 1.0,
    ) -> None:
        if len(model_factories) < 2:
            raise ValueError("ensemble requires at least two member models")
        labels = [str(label).strip() for label in model_factories]
        if any(not label for label in labels) or len(labels) != len(set(labels)):
            raise ValueError("ensemble member labels must be unique non-empty strings")
        if validation_initial_train_size < 3:
            raise ValueError("validation_initial_train_size must be at least 3")
        if validation_horizon < 1:
            raise ValueError("validation_horizon must be positive")
        if validation_step is not None and validation_step < 1:
            raise ValueError("validation_step must be positive when provided")
        if metric not in _METRICS:
            raise ValueError(f"unsupported ensemble metric {metric!r}")
        if not math.isfinite(weight_power) or weight_power <= 0:
            raise ValueError("weight_power must be a positive finite number")

        self.members = tuple(labels)
        self.validation_initial_train_size = validation_initial_train_size
        self.validation_horizon = validation_horizon
        self.validation_step = validation_step
        self.metric = metric
        self.weight_power = float(weight_power)
        self._model_factories = dict(model_factories)
        self._models: dict[str, ForecastModel] = {}
        self._scores: dict[str, float] = {}
        self._weights: dict[str, float] = {}
        self._frequency: str | None = None
        self._fitted_until: pd.Timestamp | None = None

    @staticmethod
    def _weights_from_scores(scores: Mapping[str, float], power: float) -> dict[str, float]:
        zero_labels = [label for label, score in scores.items() if score == 0.0]
        if zero_labels:
            share = 1.0 / len(zero_labels)
            return {label: (share if label in zero_labels else 0.0) for label in scores}
        raw = {label: 1.0 / (score**power) for label, score in scores.items()}
        total = sum(raw.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("ensemble validation scores cannot produce finite weights")
        return {label: value / total for label, value in raw.items()}

    def fit(self, series: PreparedSeries) -> "ValidationWeightedEnsemble":
        self.validate_training_series(series)
        if self.validation_initial_train_size + self.validation_horizon > len(series.values):
            raise ValueError("training series is too short for ensemble validation")

        scores: dict[str, float] = {}
        fitted: dict[str, ForecastModel] = {}
        for label, factory in self._model_factories.items():
            result = expanding_window_backtest(
                series,
                factory,
                initial_train_size=self.validation_initial_train_size,
                horizon=self.validation_horizon,
                step=self.validation_step,
            )
            score = float(getattr(result.aggregate, self.metric))
            if not math.isfinite(score) or score < 0:
                raise ValueError(
                    f"ensemble member {label!r} produced non-finite {self.metric} validation score"
                )
            model = factory()
            if not isinstance(model, ForecastModel):
                raise TypeError("ensemble model factories must return ForecastModel instances")
            model.fit(series)
            scores[label] = score
            fitted[label] = model

        self._scores = scores
        self._weights = self._weights_from_scores(scores, self.weight_power)
        self._models = fitted
        self._frequency = series.schema.frequency
        self._fitted_until = pd.Timestamp(series.values.index[-1])
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        self.validate_horizon(horizon)
        if not self._models or not self._weights or self._frequency is None or self._fitted_until is None:
            raise ValueError("ensemble must be fitted before forecasting")

        member_results = {label: model.forecast(horizon) for label, model in self._models.items()}
        first = next(iter(member_results.values()))
        index = first.values.index
        for label, result in member_results.items():
            if not result.values.index.equals(index):
                raise ValueError(f"ensemble member {label!r} returned a misaligned forecast index")

        matrix = np.vstack(
            [member_results[label].values.to_numpy(dtype=float) for label in self.members]
        )
        weights = np.asarray([self._weights[label] for label in self.members], dtype=float)
        combined = weights @ matrix
        values = pd.Series(combined, index=index, name="forecast", dtype=float)

        return ForecastResult(
            model_name=self.name,
            values=values,
            frequency=self._frequency,
            fitted_until=self._fitted_until,
            metadata={
                "ensemble": {
                    "members": list(self.members),
                    "validation_metric": self.metric,
                    "validation_scores": dict(self._scores),
                    "weights": dict(self._weights),
                    "weight_power": self.weight_power,
                    "validation_initial_train_size": self.validation_initial_train_size,
                    "validation_horizon": self.validation_horizon,
                    "validation_step": (
                        self.validation_horizon if self.validation_step is None else self.validation_step
                    ),
                }
            },
        )

    def save(self, path: Path) -> None:
        if not self._models:
            raise ValueError("ensemble must be fitted before saving")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        member_entries = {}
        for label, model in self._models.items():
            member_path = Path("members") / _safe_label(label)
            model.save(path / member_path)
            member_entries[label] = {
                "path": str(member_path),
                "module": model.__class__.__module__,
                "class": model.__class__.__qualname__,
            }
        state = {
            "schema_version": 1,
            "members": list(self.members),
            "validation_initial_train_size": self.validation_initial_train_size,
            "validation_horizon": self.validation_horizon,
            "validation_step": self.validation_step,
            "metric": self.metric,
            "weight_power": self.weight_power,
            "scores": self._scores,
            "weights": self._weights,
            "frequency": self._frequency,
            "fitted_until": self._fitted_until.isoformat() if self._fitted_until is not None else None,
            "fitted_members": member_entries,
        }
        (path / "ensemble.json").write_text(
            json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ValidationWeightedEnsemble":
        path = Path(path)
        state = json.loads((path / "ensemble.json").read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError("unsupported ensemble serialization version")
        models: dict[str, ForecastModel] = {}
        for label, entry in state["fitted_members"].items():
            module = importlib.import_module(entry["module"])
            model_class = module
            for part in entry["class"].split("."):
                model_class = getattr(model_class, part)
            model = model_class.load(path / entry["path"])
            if not isinstance(model, ForecastModel):
                raise TypeError("serialized ensemble member is not a ForecastModel")
            models[label] = model
        obj = cls.__new__(cls)
        obj.members = tuple(state["members"])
        obj.validation_initial_train_size = int(state["validation_initial_train_size"])
        obj.validation_horizon = int(state["validation_horizon"])
        obj.validation_step = state["validation_step"]
        obj.metric = str(state["metric"])
        obj.weight_power = float(state["weight_power"])
        obj._model_factories = {}
        obj._models = models
        obj._scores = {str(k): float(v) for k, v in state["scores"].items()}
        obj._weights = {str(k): float(v) for k, v in state["weights"].items()}
        obj._frequency = str(state["frequency"])
        obj._fitted_until = pd.Timestamp(state["fitted_until"])
        return obj
