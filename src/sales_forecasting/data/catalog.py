"""Built-in dataset contracts approved for forecasting."""

from __future__ import annotations

from .schema import DatasetContractError, DatasetSchema

CAR_PRICES_DAILY_MEDIAN = DatasetSchema(
    name="car_prices_daily_median",
    timestamp_col="saledate",
    target_col="sellingprice",
    frequency="D",
    aggregation="median",
    timezone="America/Los_Angeles",
)

CAR_PRICES_WEEKLY_MEDIAN = DatasetSchema(
    name="car_prices_weekly_median",
    timestamp_col="saledate",
    target_col="sellingprice",
    frequency="W-SUN",
    aggregation="median",
    timezone="America/Los_Angeles",
)

_BUILTIN_SCHEMAS = {
    # v1 defaults to weekly because the public auction source is too sparse for
    # a complete daily target series. The daily contract remains explicit.
    "car_prices": CAR_PRICES_WEEKLY_MEDIAN,
    "car_prices_daily": CAR_PRICES_DAILY_MEDIAN,
    "car_prices_weekly": CAR_PRICES_WEEKLY_MEDIAN,
}


def get_builtin_schema(name: str) -> DatasetSchema:
    """Return a reviewed built-in schema."""

    normalized = name.strip().lower()

    if normalized == "amazon":
        raise DatasetContractError(
            "The Amazon product/review dataset is not a time-series sales dataset. "
            "Provide observed timestamps and an observed sales or revenue target."
        )

    try:
        return _BUILTIN_SCHEMAS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_BUILTIN_SCHEMAS))
        raise DatasetContractError(
            f"Unknown built-in dataset {name!r}. Supported: {supported}"
        ) from exc
