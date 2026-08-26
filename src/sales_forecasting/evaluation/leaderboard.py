"""Compare model factories on one shared chronological backtest definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from sales_forecasting.data.missing import MissingPolicy
from sales_forecasting.data.schema import PreparedSeries

from .backtesting import BacktestResult, ModelFactory, expanding_window_backtest


@dataclass(frozen=True, slots=True)
class LeaderboardResult:
    table: pd.DataFrame
    backtests: Mapping[str, BacktestResult]
    baseline_model: str


def build_leaderboard(
    series: PreparedSeries,
    model_factories: Mapping[str, ModelFactory],
    *,
    initial_train_size: int,
    horizon: int = 1,
    step: int | None = None,
    baseline_model: str = "naive_last_value",
    missing_policy: MissingPolicy | str = "error",
) -> LeaderboardResult:
    """Backtest every model identically and rank by aggregate RMSE."""

    if baseline_model not in model_factories:
        raise ValueError(f"baseline model {baseline_model!r} is required")
    if len(model_factories) < 2:
        raise ValueError("leaderboard requires a baseline and at least one challenger")

    backtests: dict[str, BacktestResult] = {}
    rows: list[dict[str, float | str | int | bool]] = []

    for label, factory in model_factories.items():
        result = expanding_window_backtest(
            series,
            factory,
            initial_train_size=initial_train_size,
            horizon=horizon,
            step=step,
            missing_policy=missing_policy,
        )
        backtests[label] = result
        metrics = result.aggregate
        rows.append(
            {
                "model": label,
                "implementation": result.model_name,
                "folds": len(result.folds),
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "smape": metrics.smape,
                "mase": metrics.mase,
                "wape": metrics.wape,
            }
        )

    table = pd.DataFrame(rows)
    baseline_rmse = float(table.loc[table["model"] == baseline_model, "rmse"].iloc[0])
    table["rmse_vs_baseline_pct"] = (
        (table["rmse"] - baseline_rmse) / baseline_rmse * 100.0
        if baseline_rmse != 0
        else float("nan")
    )
    table["beats_baseline"] = table["rmse"] < baseline_rmse
    table.loc[table["model"] == baseline_model, "beats_baseline"] = False
    table = table.sort_values(["rmse", "mae", "model"], kind="stable").reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))

    return LeaderboardResult(
        table=table,
        backtests=backtests,
        baseline_model=baseline_model,
    )
