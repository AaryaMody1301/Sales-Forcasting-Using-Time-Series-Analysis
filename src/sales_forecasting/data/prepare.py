"""Convert raw tabular data into a validated regular target series."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from .schema import DatasetContractError, DatasetSchema, PreparedSeries


def _parse_timestamps(values: pd.Series, timezone: str | None) -> pd.Series:
    try:
        timestamps = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DatasetContractError("timestamp column contains invalid dates") from exc

    if timezone is None:
        if not is_datetime64_any_dtype(timestamps.dtype):
            raise DatasetContractError(
                "timestamp column contains mixed timezone offsets; declare a timezone "
                "in the dataset schema so offsets can be normalized explicitly"
            )
        return timestamps

    try:
        if is_datetime64_any_dtype(timestamps.dtype):
            if timestamps.dt.tz is None:
                return timestamps.dt.tz_localize(timezone)
            return timestamps.dt.tz_convert(timezone)

        # Mixed offsets (for example PST/PDT in the car-auction source) can
        # produce an object dtype. Normalize through UTC, then convert to the
        # declared business timezone so local calendar boundaries stay stable.
        normalized = pd.to_datetime(values, errors="raise", utc=True)
        return normalized.dt.tz_convert(timezone)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DatasetContractError(
            f"could not apply timezone {timezone!r} to timestamp column"
        ) from exc


def _parse_target(values: pd.Series, target_col: str) -> pd.Series:
    try:
        return pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DatasetContractError(
            f"target column {target_col!r} must be numeric"
        ) from exc


def _aggregate(resampler, method: str) -> pd.Series:
    if method == "sum":
        # Empty periods are unknown, not automatically zero.
        return resampler.sum(min_count=1)
    return getattr(resampler, method)()


def prepare_time_series(df: pd.DataFrame, schema: DatasetSchema) -> PreparedSeries:
    """Prepare one regular target series without inventing timestamps or imputing gaps.

    Missing forecast periods remain ``NaN``. Later preprocessing/evaluation code
    must decide how to handle them using training-only information.
    """

    if not isinstance(df, pd.DataFrame):
        raise DatasetContractError("input must be a pandas DataFrame")
    if df.empty:
        raise DatasetContractError("input DataFrame is empty")

    required = {schema.timestamp_col, schema.target_col}
    missing_columns = sorted(required.difference(df.columns))
    if missing_columns:
        raise DatasetContractError(
            "missing required columns: " + ", ".join(missing_columns)
        )

    work = df.loc[:, [schema.timestamp_col, schema.target_col]].copy()
    work[schema.timestamp_col] = _parse_timestamps(
        work[schema.timestamp_col], schema.timezone
    )
    work[schema.target_col] = _parse_target(
        work[schema.target_col], schema.target_col
    )
    work = work.sort_values(schema.timestamp_col, kind="stable")

    if work[schema.timestamp_col].isna().any():
        raise DatasetContractError("timestamp column contains missing values")
    if work[schema.target_col].isna().all():
        raise DatasetContractError("target column contains no observed values")

    indexed = work.set_index(schema.timestamp_col)[schema.target_col]
    indexed.name = schema.target_col

    if schema.aggregation is not None:
        resampler = indexed.resample(schema.frequency)
        values = _aggregate(resampler, schema.aggregation)
    else:
        if indexed.index.has_duplicates:
            raise DatasetContractError(
                "duplicate timestamps require an explicit aggregation method"
            )

        expected_index = pd.date_range(
            start=indexed.index.min(),
            end=indexed.index.max(),
            freq=schema.frequency,
        )
        off_grid = indexed.index.difference(expected_index)
        if len(off_grid):
            sample = ", ".join(str(value) for value in off_grid[:3])
            raise DatasetContractError(
                "timestamps do not align to the declared frequency "
                f"{schema.frequency!r}; examples: {sample}"
            )
        values = indexed.reindex(expected_index)
        values.index.name = schema.timestamp_col

    values = values.sort_index()
    values.name = schema.target_col

    return PreparedSeries(
        values=values,
        schema=schema,
        source_rows=len(df),
        missing_periods=int(values.isna().sum()),
    )
