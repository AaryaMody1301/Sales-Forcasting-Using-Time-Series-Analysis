"""Expanding-window evaluation shared by every forecasting model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from sales_forecasting.data.schema import DatasetContractError, PreparedSeries
from sales_forecasting.models.base import ForecastModel

from .metrics import ForecastMetrics, calculate_metrics

ModelFactory = Callable[[], ForecastModel]


@dataclass(frozen=True, slots=True)
class BacktestFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: ForecastMetrics
    forecast: pd.Series


@dataclass(frozen=True, slots=True)
class BacktestResult:
    model_name: str
    folds: tuple[BacktestFold, ...]
    aggregate: ForecastMetrics


def _safe_nanmean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _aggregate_metrics(folds: list[BacktestFold]) -> ForecastMetrics:
    return ForecastMetrics(
        mae=_safe_nanmean([fold.metrics.mae for fold in folds]),
        rmse=_safe_nanmean([fold.metrics.rmse for fold in folds]),
        smape=_safe_nanmean([fold.metrics.smape for fold in folds]),
        mase=_safe_nanmean([fold.metrics.mase for fold in folds]),
        wape=_safe_nanmean([fold.metrics.wape for fold in folds]),
    )


def expanding_window_backtest(
    series: PreparedSeries,
    model_factory: ModelFactory,
    *,
    initial_train_size: int,
    horizon: int = 1,
    step: int | None = None,
) -> BacktestResult:
    """Evaluate fresh model instances across expanding chronological folds.

    Each fold fits only on observations strictly earlier than its test window.
    The next fold may include observations revealed by previous folds, matching
    a walk-forward production process. The default ``step`` equals ``horizon``
    so test windows do not overlap.
    """

    values = series.values

    if values.isna().any():
        raise DatasetContractError(
            "backtesting requires a complete series; missing periods must be "
            "handled using training-only preprocessing"
        )
    if initial_train_size < 3:
        raise ValueError("initial_train_size must be at least 3")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    step = horizon if step is None else step
    if step < 1:
        raise ValueError("step must be positive")
    if initial_train_size + horizon > len(values):
        raise ValueError(
            "series is too short for the requested initial train size and horizon"
        )

    folds: list[BacktestFold] = []
    train_end = initial_train_size
    fold_number = 0
    model_name: str | None = None

    while train_end + horizon <= len(values):
        train_values = values.iloc[:train_end].copy()
        test_values = values.iloc[train_end : train_end + horizon].copy()

        training = PreparedSeries(
            values=train_values,
            schema=series.schema,
            source_rows=len(train_values),
            missing_periods=0,
        )

        model = model_factory()
        if not isinstance(model, ForecastModel):
            raise TypeError("model_factory must return a ForecastModel")

        model.fit(training)
        predicted = model.forecast(horizon)

        if not predicted.values.index.equals(test_values.index):
            raise ValueError(
                "model forecast index does not match the backtest test window"
            )

        metrics = calculate_metrics(
            actual=test_values.values,
            forecast=predicted.values.values,
            insample=train_values.values,
        )

        if model_name is None:
            model_name = model.name
        elif model.name != model_name:
            raise ValueError("model_factory returned inconsistent model types")

        folds.append(
            BacktestFold(
                fold=fold_number,
                train_start=pd.Timestamp(train_values.index[0]),
                train_end=pd.Timestamp(train_values.index[-1]),
                test_start=pd.Timestamp(test_values.index[0]),
                test_end=pd.Timestamp(test_values.index[-1]),
                metrics=metrics,
                forecast=predicted.values.copy(),
            )
        )

        train_end += step
        fold_number += 1

    return BacktestResult(
        model_name=model_name or "unknown",
        folds=tuple(folds),
        aggregate=_aggregate_metrics(folds),
    )
