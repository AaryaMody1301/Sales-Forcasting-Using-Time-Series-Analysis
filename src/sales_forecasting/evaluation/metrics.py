"""Common forecast metrics used by every model family."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """A common, model-independent metric set."""

    mae: float
    rmse: float
    smape: float
    mase: float
    wape: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite_arrays(actual, forecast) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)

    if actual_array.shape != forecast_array.shape:
        raise ValueError("actual and forecast must have the same shape")
    if actual_array.size == 0:
        raise ValueError("actual and forecast cannot be empty")
    if not np.isfinite(actual_array).all() or not np.isfinite(forecast_array).all():
        raise ValueError("metrics require finite actual and forecast values")

    return actual_array, forecast_array


def calculate_metrics(actual, forecast, insample) -> ForecastMetrics:
    """Calculate the standard Phase 2 metric set.

    MASE uses a one-step naive in-sample scale. MAPE is deliberately excluded
    because it is undefined or unstable around zero. sMAPE and WAPE remain
    percentage-like summaries, while MAE/RMSE retain the target's native unit.
    """

    actual_array, forecast_array = _finite_arrays(actual, forecast)
    insample_array = np.asarray(insample, dtype=float)

    if insample_array.size < 2 or not np.isfinite(insample_array).all():
        raise ValueError("insample must contain at least two finite values")

    error = actual_array - forecast_array
    absolute_error = np.abs(error)

    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(error**2)))

    smape_denominator = np.abs(actual_array) + np.abs(forecast_array)
    smape_terms = np.divide(
        2.0 * absolute_error,
        smape_denominator,
        out=np.zeros_like(smape_denominator),
        where=smape_denominator != 0,
    )
    smape = float(100.0 * np.mean(smape_terms))

    naive_scale = float(np.mean(np.abs(np.diff(insample_array))))
    mase = float(mae / naive_scale) if naive_scale > 0 else float("nan")

    actual_scale = float(np.sum(np.abs(actual_array)))
    wape = (
        float(100.0 * np.sum(absolute_error) / actual_scale)
        if actual_scale > 0
        else float("nan")
    )

    return ForecastMetrics(
        mae=mae,
        rmse=rmse,
        smape=smape,
        mase=mase,
        wape=wape,
    )
