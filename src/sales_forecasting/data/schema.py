"""Explicit data contracts for time-series forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
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

    ``known_future_regressors`` names covariates whose values are legitimately
    available at forecast time. Their values are carried separately on
    :class:`PreparedSeries` so future covariates can extend beyond the last known
    target without inventing future target rows.
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

        if any(not isinstance(col, str) or not col.strip() for col in self.known_future_regressors):
            raise DatasetContractError("known_future_regressors must contain non-empty strings")

        duplicate_regressors = {
            col
            for col in self.known_future_regressors
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
    """A validated regular target series plus optional known-future covariates.

    ``future_regressors`` may extend beyond ``values.index`` because those future
    covariates are known at forecast time while future targets are not. Declared
    regressors must be numeric, finite, regular, and cover every target timestamp.
    """

    values: pd.Series
    schema: DatasetSchema
    source_rows: int
    missing_periods: int
    future_regressors: pd.DataFrame | None = None

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

        declared = tuple(self.schema.known_future_regressors)
        regressors = self.future_regressors
        if not declared:
            if regressors is not None:
                raise DatasetContractError(
                    "future_regressors were supplied but the schema declares no known future regressors"
                )
            return

        if regressors is None:
            raise DatasetContractError(
                "schema declares known future regressors but no future_regressors frame was supplied"
            )
        if not isinstance(regressors, pd.DataFrame):
            raise DatasetContractError("future_regressors must be a pandas DataFrame")
        if not isinstance(regressors.index, pd.DatetimeIndex):
            raise DatasetContractError("future_regressors must use a DatetimeIndex")
        if not regressors.index.is_monotonic_increasing:
            raise DatasetContractError("future_regressors index must be sorted")
        if regressors.index.has_duplicates:
            raise DatasetContractError("future_regressors index must be unique")

        expected_columns = list(declared)
        if list(regressors.columns) != expected_columns:
            raise DatasetContractError(
                "future_regressors columns must exactly match schema order: "
                + ", ".join(expected_columns)
            )
        if regressors.empty:
            raise DatasetContractError("future_regressors cannot be empty")
        if not self.values.index.isin(regressors.index).all():
            missing = self.values.index[~self.values.index.isin(regressors.index)]
            sample = ", ".join(str(value) for value in missing[:3])
            raise DatasetContractError(
                "future_regressors must cover every target timestamp; missing examples: " + sample
            )

        try:
            numeric = regressors.astype(float)
        except (TypeError, ValueError) as exc:
            raise DatasetContractError("known future regressors must be numeric") from exc
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise DatasetContractError("known future regressors must contain only finite values")

        expected_index = pd.date_range(
            start=regressors.index[0],
            end=regressors.index[-1],
            freq=self.schema.frequency,
        )
        if not regressors.index.equals(expected_index):
            raise DatasetContractError(
                "future_regressors must form a complete regular grid at the declared frequency"
            )

    @property
    def regressor_horizon(self) -> int:
        """Number of complete future covariate rows beyond the final observed target."""

        if self.future_regressors is None:
            return 0
        return int((self.future_regressors.index > self.values.index[-1]).sum())
