"""Source-specific normalization for the public vehicle-sales auction dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from .schema import DatasetContractError

DEFAULT_BUSINESS_TIMEZONE = "America/Los_Angeles"
# The source is documented as 2014-2015 auction transactions. A small guard
# band protects legitimate timezone-boundary rows while rejecting malformed
# far-future dates that can still parse as valid datetimes.
DEFAULT_EARLIEST_UTC = pd.Timestamp("2013-12-01", tz="UTC")
DEFAULT_LATEST_EXCLUSIVE_UTC = pd.Timestamp("2016-01-01", tz="UTC")


@dataclass(frozen=True, slots=True)
class VehicleSalesCleaningReport:
    raw_rows: int
    invalid_timestamp_rows: int
    out_of_range_timestamp_rows: int
    invalid_target_rows: int
    excluded_rows: int
    usable_rows: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def clean_vehicle_sales_source(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "saledate",
    target_col: str = "sellingprice",
    timezone: str = DEFAULT_BUSINESS_TIMEZONE,
    earliest_utc: pd.Timestamp = DEFAULT_EARLIEST_UTC,
    latest_exclusive_utc: pd.Timestamp = DEFAULT_LATEST_EXCLUSIVE_UTC,
    max_excluded_fraction: float = 0.01,
) -> tuple[pd.DataFrame, VehicleSalesCleaningReport]:
    """Return valid vehicle-sale rows and an auditable exclusion report.

    Invalid timestamps/targets and timestamps outside the documented source era
    are dropped rather than guessed or imputed. The generic time-series layer
    remains strict; this function is an explicit adapter for this known source.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DatasetContractError("vehicle-sales source must be a non-empty DataFrame")
    missing = [column for column in (timestamp_col, target_col) if column not in frame.columns]
    if missing:
        raise DatasetContractError(
            "vehicle-sales source is missing required columns: " + ", ".join(missing)
        )
    if not 0 <= max_excluded_fraction <= 1:
        raise ValueError("max_excluded_fraction must be between 0 and 1")

    timestamps_utc = pd.to_datetime(
        frame[timestamp_col],
        errors="coerce",
        utc=True,
        format="mixed",
    )
    targets = pd.to_numeric(frame[target_col], errors="coerce")

    invalid_timestamp = timestamps_utc.isna()
    out_of_range = (~invalid_timestamp) & (
        (timestamps_utc < earliest_utc) | (timestamps_utc >= latest_exclusive_utc)
    )
    invalid_target = targets.isna()
    valid = ~(invalid_timestamp | out_of_range | invalid_target)

    cleaned = pd.DataFrame(
        {
            timestamp_col: timestamps_utc.loc[valid].dt.tz_convert(timezone),
            target_col: targets.loc[valid].astype(float),
        }
    ).reset_index(drop=True)

    report = VehicleSalesCleaningReport(
        raw_rows=int(len(frame)),
        invalid_timestamp_rows=int(invalid_timestamp.sum()),
        out_of_range_timestamp_rows=int(out_of_range.sum()),
        invalid_target_rows=int(invalid_target.sum()),
        excluded_rows=int((~valid).sum()),
        usable_rows=int(valid.sum()),
    )
    if report.usable_rows == 0:
        raise DatasetContractError("vehicle-sales source contains no usable rows")
    if report.excluded_rows / report.raw_rows > max_excluded_fraction:
        raise DatasetContractError(
            "vehicle-sales source exceeds the allowed invalid-row fraction; "
            "investigate the source before forecasting"
        )
    return cleaned, report
