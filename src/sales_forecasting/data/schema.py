"""Explicit data contracts for time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.tseries.frequencies import to_offset

Aggregation = Literal[
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "first",
    "last",
    "count",
]

_ALLOWED_AGGREGATIONS = {
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "first",
    "last",
    "count",
}


class DatasetContractError(ValueError):
    """Raised when source data cannot satisfy a forecasting contract."""


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Describe how raw rows become one regular target time series.

    ``aggregation`` is required for event/transaction datasets that can contain
    multiple observations inside one forecast period. If it is ``None``, the
    input must already contain at most one observation per timestamp and all
    timestamps must align to ``frequency``.
    """

    name: str
    timestamp_col: str
    target_col: str
    frequency: str
    aggregation: Aggregation | None = None
    timezone: str | None = None
    known_future_regressors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("name", "timestamp_col", "target_col", "frequency"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise DatasetContractError(f"{field_name} must be a non-empty string")

        if self.timestamp_col == self.target_col:
            raise DatasetContractError("timestamp_col and target_col must be different")

        if self.aggregation not in _ALLOWED_AGGREGATIONS and self.aggregation is not None:
            raise DatasetContractError(
                f"Unsupported aggregation {self.aggregation!r}; "
                f"choose one of {sorted(_ALLOWED_AGGREGATIONS)}"
            )

        try:
            to_offset(self.frequency)
        except (TypeError, ValueError) as exc:
            raise DatasetContractError(
                f"Invalid pandas frequency {self.frequency!r}"
            ) from exc

        duplicate_regressors = {
            col for col in self.known_future_regressors
            if self.known_future_regressors.count(col) > 1
        }
        if duplicate_regressors:
            raise DatasetContractError(
                "known_future_regressors contains duplicates: "
                + ", ".join(sorted(duplicate_regressors))
            )

        forbidden = {self.timestamp_col, self.target_col}
        overlap = forbidden.intersection(self.known_future_regressors)
        if overlap:
            raise DatasetContractError(
                "known_future_regressors cannot repeat timestamp/target columns: "
                + ", ".join(sorted(overlap))
            )


@dataclass(frozen=True, slots=True)
class PreparedSeries:
    """A validated regular series plus provenance needed by later phases."""

    values: pd.Series
    schema: DatasetSchema
    source_rows: int
    missing_periods: int

    def __post_init__(self) -> None:
        if not isinstance(self.values.index, pd.DatetimeIndex):
            raise DatasetContractError("Prepared series must use a DatetimeIndex")
        if not self.values.index.is_monotonic_increasing:
            raise DatasetContractError("Prepared series index must be sorted")
        if self.values.index.has_duplicates:
            raise DatasetContractError("Prepared series index must be unique")
        if self.source_rows < 1:
            raise DatasetContractError("source_rows must be positive")
        if self.missing_periods < 0:
            raise DatasetContractError("missing_periods cannot be negative")
