import math

import pytest

from sales_forecasting import calculate_metrics


def test_perfect_forecast_has_zero_error():
    metrics = calculate_metrics(
        actual=[1.0, 2.0],
        forecast=[1.0, 2.0],
        insample=[0.0, 1.0, 2.0],
    )

    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.smape == 0.0
    assert metrics.mase == 0.0
    assert metrics.wape == 0.0


def test_smape_handles_zero_zero_without_dividing_by_zero():
    metrics = calculate_metrics(
        actual=[0.0, 10.0],
        forecast=[0.0, 5.0],
        insample=[1.0, 2.0, 3.0],
    )

    assert math.isfinite(metrics.smape)


def test_metric_shapes_must_match():
    with pytest.raises(ValueError, match="same shape"):
        calculate_metrics([1.0], [1.0, 2.0], [0.0, 1.0])
