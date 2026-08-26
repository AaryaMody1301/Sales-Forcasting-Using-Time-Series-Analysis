"""Canonical command-line interface for reproducible forecasting experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from sales_forecasting.artifacts import ExperimentSpec, ModelSpec, record_experiment
from sales_forecasting.data import attach_known_future_regressors
from sales_forecasting.data.catalog import get_builtin_schema
from sales_forecasting.data.prepare import prepare_time_series
from sales_forecasting.data.schema import DatasetContractError, DatasetSchema, PreparedSeries
from sales_forecasting.evaluation import NestedTunedForecaster, ValidationWeightedEnsemble
from sales_forecasting.models import (
    ARIMAForecaster,
    ETSForecaster,
    GradientBoostingForecaster,
    LastValueNaiveModel,
    ProphetForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

_MODEL_CLASSES = {
    "naive_last_value": LastValueNaiveModel,
    "arima": ARIMAForecaster,
    "ets": ETSForecaster,
    "prophet": ProphetForecaster,
    "random_forest": RandomForestForecaster,
    "gradient_boosting": GradientBoostingForecaster,
    "xgboost": XGBoostForecaster,
}
_TUNABLE_MODEL_CLASSES = {
    "random_forest": RandomForestForecaster,
    "gradient_boosting": GradientBoostingForecaster,
    "xgboost": XGBoostForecaster,
}
_AGGREGATIONS = ["sum", "mean", "median", "min", "max", "first", "last", "count"]
_METRICS = ["mae", "rmse", "smape", "mase", "wape"]


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", required=True, type=Path, help="Input target CSV file")
    parser.add_argument(
        "--dataset",
        default="custom",
        help="Built-in dataset name (for example car_prices) or 'custom'",
    )
    parser.add_argument("--timestamp-col")
    parser.add_argument("--target-col")
    parser.add_argument("--frequency")
    parser.add_argument("--aggregation", choices=_AGGREGATIONS)
    parser.add_argument("--timezone")
    parser.add_argument(
        "--regressors-csv",
        type=Path,
        help="Separate CSV containing historical and genuinely known future covariates",
    )
    parser.add_argument(
        "--regressor-timestamp-col",
        help="Timestamp column in --regressors-csv",
    )
    parser.add_argument(
        "--regressor-cols",
        nargs="+",
        help="Regressor columns whose future values are known at forecast time",
    )


def _schema_from_args(args: argparse.Namespace) -> DatasetSchema:
    if args.dataset != "custom":
        return get_builtin_schema(args.dataset)

    required = {
        "--timestamp-col": args.timestamp_col,
        "--target-col": args.target_col,
        "--frequency": args.frequency,
    }
    missing = [flag for flag, value in required.items() if not value]
    if missing:
        raise ValueError("custom datasets require " + ", ".join(missing))

    return DatasetSchema(
        name=f"custom_{Path(args.csv).stem}",
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        frequency=args.frequency,
        aggregation=args.aggregation,
        timezone=args.timezone,
    )


def _load_prepared(args: argparse.Namespace) -> PreparedSeries:
    frame = pd.read_csv(args.csv)
    prepared = prepare_time_series(frame, _schema_from_args(args))

    regressor_settings = (
        args.regressors_csv,
        args.regressor_timestamp_col,
        args.regressor_cols,
    )
    if any(value is not None for value in regressor_settings):
        if not all(value is not None for value in regressor_settings):
            raise ValueError(
                "known future regressors require --regressors-csv, "
                "--regressor-timestamp-col, and --regressor-cols together"
            )
        regressor_frame = pd.read_csv(args.regressors_csv)
        prepared = attach_known_future_regressors(
            prepared,
            regressor_frame,
            timestamp_col=args.regressor_timestamp_col,
            regressor_cols=args.regressor_cols,
        )
    return prepared


def _experiment_from_args(args: argparse.Namespace) -> ExperimentSpec:
    return ExperimentSpec(
        initial_train_size=args.initial_train_size,
        horizon=args.horizon,
        step=args.step,
        baseline_model="naive_last_value",
        missing_policy=args.missing_policy,
    )


def _print_run(run) -> None:
    print(f"run_id: {run.run_id}")
    print(f"run_dir: {run.run_dir}")
    print(run.leaderboard.table.to_string(index=False))


def _cmd_inspect(args: argparse.Namespace) -> int:
    prepared = _load_prepared(args)
    summary = {
        "dataset": prepared.schema.name,
        "observations": len(prepared.values),
        "source_rows": prepared.source_rows,
        "missing_periods": prepared.missing_periods,
        "start": prepared.values.index[0].isoformat(),
        "end": prepared.values.index[-1].isoformat(),
        "frequency": prepared.schema.frequency,
        "target": prepared.schema.target_col,
        "known_future_regressors": list(prepared.schema.known_future_regressors),
        "future_regressor_horizon": prepared.regressor_horizon,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    prepared = _load_prepared(args)
    labels = list(dict.fromkeys(args.models))
    if "naive_last_value" not in labels:
        labels.insert(0, "naive_last_value")

    unknown = sorted(set(labels).difference(_MODEL_CLASSES))
    if unknown:
        raise ValueError("unknown models: " + ", ".join(unknown))

    model_specs = [
        ModelSpec(label=label, factory=_MODEL_CLASSES[label], metadata={"source": "cli"})
        for label in labels
    ]
    if len(model_specs) < 2:
        raise ValueError("run requires at least one challenger in addition to the baseline")

    run = record_experiment(
        prepared,
        model_specs,
        _experiment_from_args(args),
        artifact_root=args.artifact_root,
        code_revision=_git_revision(),
    )
    _print_run(run)
    return 0


def _read_grid(path: Path) -> dict[str, list[Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("grid JSON must be a non-empty object")
    grid: dict[str, list[Any]] = {}
    for key, values in data.items():
        if not isinstance(key, str) or not isinstance(values, list) or not values:
            raise ValueError("grid entries must map parameter names to non-empty JSON arrays")
        grid[key] = values
    return grid


def _cmd_tune(args: argparse.Namespace) -> int:
    prepared = _load_prepared(args)
    if args.inner_initial_train_size + args.inner_horizon > args.initial_train_size:
        raise ValueError(
            "inner tuning window must fit entirely inside the earliest outer training fold"
        )

    model_class = _TUNABLE_MODEL_CLASSES[args.model]
    grid = _read_grid(args.grid)
    tuned_label = f"tuned_{args.model}"

    def tuned_factory():
        return NestedTunedForecaster(
            model_class,
            param_grid=grid,
            inner_initial_train_size=args.inner_initial_train_size,
            inner_horizon=args.inner_horizon,
            inner_step=args.inner_step,
            metric=args.metric,
        )

    model_specs = [
        ModelSpec("naive_last_value", LastValueNaiveModel, metadata={"source": "cli"}),
        ModelSpec(
            tuned_label,
            tuned_factory,
            metadata={"source": "cli", "grid_file": str(args.grid)},
        ),
    ]
    run = record_experiment(
        prepared,
        model_specs,
        _experiment_from_args(args),
        artifact_root=args.artifact_root,
        code_revision=_git_revision(),
    )
    _print_run(run)
    return 0


def _cmd_ensemble(args: argparse.Namespace) -> int:
    prepared = _load_prepared(args)
    members = list(dict.fromkeys(args.members))
    if len(members) < 2:
        raise ValueError("ensemble requires at least two distinct member models")
    unknown = sorted(set(members).difference(_MODEL_CLASSES))
    if unknown:
        raise ValueError("unknown ensemble members: " + ", ".join(unknown))
    if args.validation_initial_train_size + args.validation_horizon > args.initial_train_size:
        raise ValueError(
            "ensemble validation window must fit inside the earliest outer training fold"
        )

    member_factories = {label: _MODEL_CLASSES[label] for label in members}

    def ensemble_factory():
        return ValidationWeightedEnsemble(
            member_factories,
            validation_initial_train_size=args.validation_initial_train_size,
            validation_horizon=args.validation_horizon,
            validation_step=args.validation_step,
            metric=args.metric,
            weight_power=args.weight_power,
        )

    model_specs = [
        ModelSpec("naive_last_value", LastValueNaiveModel, metadata={"source": "cli"})
    ]
    for label in members:
        if label != "naive_last_value":
            model_specs.append(
                ModelSpec(label, _MODEL_CLASSES[label], metadata={"source": "cli", "ensemble_member": True})
            )
    model_specs.append(
        ModelSpec(
            "validation_weighted_ensemble",
            ensemble_factory,
            metadata={"source": "cli", "members": members},
        )
    )

    run = record_experiment(
        prepared,
        model_specs,
        _experiment_from_args(args),
        artifact_root=args.artifact_root,
        code_revision=_git_revision(),
    )
    _print_run(run)
    return 0


def _add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-train-size", required=True, type=int)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--step", type=int)
    parser.add_argument(
        "--missing-policy",
        choices=["error", "forward_fill"],
        default="error",
        help="Training-only target missing-period policy",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sales-forecast",
        description="Leakage-aware time-series forecasting experiments",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Validate and summarize a dataset")
    _add_dataset_arguments(inspect_parser)
    inspect_parser.set_defaults(handler=_cmd_inspect)

    run_parser = subparsers.add_parser("run", help="Run a reproducible model leaderboard")
    _add_dataset_arguments(run_parser)
    run_parser.add_argument(
        "--models",
        nargs="+",
        default=["naive_last_value", "arima", "ets"],
        help="Models to evaluate; Prophet is available as 'prophet' when its optional dependency is installed",
    )
    _add_experiment_arguments(run_parser)
    run_parser.set_defaults(handler=_cmd_run)

    tune_parser = subparsers.add_parser(
        "tune",
        help="Nested chronological tuning for one ML challenger",
    )
    _add_dataset_arguments(tune_parser)
    tune_parser.add_argument("--model", required=True, choices=sorted(_TUNABLE_MODEL_CLASSES))
    tune_parser.add_argument("--grid", required=True, type=Path, help="JSON parameter grid")
    tune_parser.add_argument("--metric", default="rmse", choices=_METRICS)
    tune_parser.add_argument("--inner-initial-train-size", required=True, type=int)
    tune_parser.add_argument("--inner-horizon", type=int, default=1)
    tune_parser.add_argument("--inner-step", type=int)
    _add_experiment_arguments(tune_parser)
    tune_parser.set_defaults(handler=_cmd_tune)

    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help="Learn ensemble weights from chronological validation folds",
    )
    _add_dataset_arguments(ensemble_parser)
    ensemble_parser.add_argument(
        "--members",
        nargs="+",
        required=True,
        help="At least two model labels, for example arima ets prophet",
    )
    ensemble_parser.add_argument("--validation-initial-train-size", required=True, type=int)
    ensemble_parser.add_argument("--validation-horizon", type=int, default=1)
    ensemble_parser.add_argument("--validation-step", type=int)
    ensemble_parser.add_argument("--metric", choices=_METRICS, default="rmse")
    ensemble_parser.add_argument("--weight-power", type=float, default=1.0)
    _add_experiment_arguments(ensemble_parser)
    ensemble_parser.set_defaults(handler=_cmd_ensemble)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (DatasetContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
