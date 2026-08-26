"""Known-future regressor contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd

from .schema import DatasetContractError, PreparedSeries


def _aligned_timestamps(values: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    try:
        timestamps = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DatasetContractError("regressor timestamp column contains invalid dates") from exc

    target_tz = target_index.tz
    timestamp_tz = timestamps.dt.tz
    if target_tz is None and timestamp_tz is not None:
        raise DatasetContractError(
            "regressor timestamps are timezone-aware while the target index is timezone-naive"
        )
    if target_tz is not None:
        try:
            if timestamp_tz is None:
                timestamps = timestamps.dt.tz_localize(target_tz)
            else:
                timestamps = timestamps.dt.tz_convert(target_tz)
        except (TypeError, ValueError) as exc:
            raise DatasetContractError("regressor timestamps cannot be aligned to target timezone") from exc
    return timestamps


def attach_known_future_regressors(
    series: PreparedSeries,
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    regressor_cols: Sequence[str],
) -> PreparedSeries:
    """Attach explicit known-future covariates to a prepared target series.

    The regressor frame is intentionally separate from the target data so it may
    contain dates after the last observed target. Values must be numeric, finite,
    unique by timestamp, and form a complete grid at the target frequency.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DatasetContractError("regressor input must be a non-empty pandas DataFrame")
    if not isinstance(timestamp_col, str) or not timestamp_col.strip():
        raise DatasetContractError("regressor timestamp_col must be a non-empty string")

    columns = tuple(str(col).strip() for col in regressor_cols)
    if not columns or any(not col for col in columns):
        raise DatasetContractError("at least one non-empty regressor column is required")
    if len(columns) != len(set(columns)):
        raise DatasetContractError("regressor columns must be unique")
    if timestamp_col in columns:
        raise DatasetContractError("regressor timestamp column cannot also be a regressor")

    required = {timestamp_col, *columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DatasetContractError("missing regressor columns: " + ", ".join(missing))

    work = frame.loc[:, [timestamp_col, *columns]].copy()
    work[timestamp_col] = _aligned_timestamps(work[timestamp_col], series.values.index)
    if work[timestamp_col].isna().any():
        raise DatasetContractError("regressor timestamp column contains missing values")
    if work[timestamp_col].duplicated().any():
        raise DatasetContractError("regressor timestamps must be unique")

    for col in columns:
        try:
            work[col] = pd.to_numeric(work[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise DatasetContractError(f"regressor column {col!r} must be numeric") from exc

    work = work.sort_values(timestamp_col, kind="stable").set_index(timestamp_col)
    work.index.name = series.schema.timestamp_col
    work = work.loc[work.index >= series.values.index[0], list(columns)]
    if work.empty or work.index[-1] < series.values.index[-1]:
        raise DatasetContractError(
            "known future regressors must cover the complete observed target history"
        )

    expected_index = pd.date_range(
        start=series.values.index[0],
        end=work.index[-1],
        freq=series.schema.frequency,
    )
    if not work.index.equals(expected_index):
        missing_grid = expected_index.difference(work.index)
        extra_grid = work.index.difference(expected_index)
        details = []
        if len(missing_grid):
            details.append("missing " + ", ".join(str(value) for value in missing_grid[:3]))
        if len(extra_grid):
            details.append("off-grid " + ", ".join(str(value) for value in extra_grid[:3]))
        raise DatasetContractError(
            "known future regressors must form a complete regular grid; " + "; ".join(details)
        )

    numeric = work.astype(float)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DatasetContractError("known future regressors must contain only finite values")

    schema = replace(series.schema, known_future_regressors=columns)
    return PreparedSeries(
        values=series.values.copy(),
        schema=schema,
        source_rows=series.source_rows,
        missing_periods=series.missing_periods,
        future_regressors=numeric,
    )


def future_regressors_for_horizon(series: PreparedSeries, horizon: int) -> pd.DataFrame:
    """Return the exact future covariate rows required for a forecast horizon."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not series.schema.known_future_regressors or series.future_regressors is None:
        raise DatasetContractError("forecast requires declared known future regressors")

    offset = pd.tseries.frequencies.to_offset(series.schema.frequency)
    index = pd.date_range(
        start=series.values.index[-1] + offset,
        periods=horizon,
        freq=series.schema.frequency,
    )
    missing = index.difference(series.future_regressors.index)
    if len(missing):
        sample = ", ".join(str(value) for value in missing[:3])
        raise DatasetContractError(
            "known future regressor values are missing for requested forecast dates: " + sample
        )
    return series.future_regressors.loc[index, list(series.schema.known_future_regressors)].copy()
