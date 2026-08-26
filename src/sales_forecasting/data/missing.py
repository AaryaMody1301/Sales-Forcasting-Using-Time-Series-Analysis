"""Causal missing-period policies for model training."""

from __future__ import annotations

from typing import Literal

from .schema import DatasetContractError, PreparedSeries

MissingPolicy = Literal["error", "forward_fill"]
_ALLOWED_POLICIES = {"error", "forward_fill"}


def normalize_missing_policy(policy: str) -> MissingPolicy:
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized not in _ALLOWED_POLICIES:
        raise ValueError(
            f"unsupported missing policy {policy!r}; choose one of {sorted(_ALLOWED_POLICIES)}"
        )
    return normalized  # type: ignore[return-value]


def _copy_with_values(series: PreparedSeries, values) -> PreparedSeries:
    return PreparedSeries(
        values=values,
        schema=series.schema,
        source_rows=series.source_rows,
        missing_periods=int(values.isna().sum()),
        future_regressors=(
            None if series.future_regressors is None else series.future_regressors.copy()
        ),
    )


def apply_training_missing_policy(
    series: PreparedSeries,
    policy: str = "error",
) -> PreparedSeries:
    """Resolve missing training periods without using future target observations."""

    policy = normalize_missing_policy(policy)
    values = series.values.astype(float).copy()

    if not values.isna().any():
        return _copy_with_values(series, values)

    if policy == "error":
        raise DatasetContractError(
            "training data contains missing periods; resolve them with an explicit "
            "training-only preprocessing policy"
        )

    filled = values.ffill()
    if filled.isna().any():
        raise DatasetContractError(
            "forward_fill cannot resolve leading missing periods because no earlier "
            "observation exists"
        )

    return _copy_with_values(series, filled)
