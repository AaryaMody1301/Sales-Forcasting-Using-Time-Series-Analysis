"""Canonical package for the sales forecasting refactor."""

from .data.catalog import CAR_PRICES_DAILY_MEDIAN, get_builtin_schema
from .data.prepare import prepare_time_series
from .data.schema import (
    DatasetContractError,
    DatasetSchema,
    PreparedSeries,
)

__all__ = [
    "CAR_PRICES_DAILY_MEDIAN",
    "DatasetContractError",
    "DatasetSchema",
    "PreparedSeries",
    "get_builtin_schema",
    "prepare_time_series",
]

__version__ = "0.1.0"
