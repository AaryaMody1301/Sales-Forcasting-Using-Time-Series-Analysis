"""Single model interface used by all model implementations from Phase 2 onward."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sales_forecasting.data.schema import DatasetContractError, PreparedSeries


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Normalized output returned by every forecasting model."""

    model_name: str
    values: pd.Series
    frequency: str
    fitted_until: pd.Timestamp
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.values.empty:
            raise ValueError("forecast values cannot be empty")
        if not isinstance(self.values.index, pd.DatetimeIndex):
            raise ValueError("forecast values must use a DatetimeIndex")
        if self.values.index.has_duplicates:
            raise ValueError("forecast index must be unique")
        if not self.values.index.is_monotonic_increasing:
            raise ValueError("forecast index must be sorted")
        if self.values.isna().any():
            raise ValueError("forecast values cannot contain missing values")


class ForecastModel(ABC):
    """Canonical fit/forecast/persist contract for all model families."""

    name: str

    @abstractmethod
    def fit(self, series: PreparedSeries) -> "ForecastModel":
        """Fit using only the training series supplied by the evaluator."""

    @abstractmethod
    def forecast(self, horizon: int) -> ForecastResult:
        """Forecast exactly ``horizon`` future periods."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist all model state required for a later forecast."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ForecastModel":
        """Restore a model produced by ``save``."""

    @staticmethod
    def validate_training_series(series: PreparedSeries) -> None:
        """Common defensive checks before an implementation fits."""

        if series.values.isna().any():
            raise DatasetContractError(
                "training data contains missing periods; handle them explicitly "
                "inside the training-only preprocessing pipeline"
            )
        if len(series.values) < 3:
            raise DatasetContractError("at least three observations are required")

    @staticmethod
    def validate_horizon(horizon: int) -> None:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
