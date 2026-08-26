"""Run orchestration and deterministic local artifact storage."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from sales_forecasting.data.missing import MissingPolicy, normalize_missing_policy
from sales_forecasting.data.schema import PreparedSeries
from sales_forecasting.evaluation import build_leaderboard
from sales_forecasting.evaluation.leaderboard import LeaderboardResult
from sales_forecasting.models.base import ForecastModel

from .fingerprints import (
    fingerprint_config,
    fingerprint_prepared_series,
    normalize_json_value,
    sha256_file,
)
from .manifest import MANIFEST_SCHEMA_VERSION

ModelFactory = Callable[[], ForecastModel]
_PROJECT_NAME = "sales-forecasting-time-series"
_DEPENDENCIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "statsmodels",
    "xgboost",
    "streamlit",
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug.lower() or "run"


def _package_version() -> str:
    try:
        return importlib.metadata.version(_PROJECT_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "0+unknown"


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _timestamp(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).isoformat()


def _artifact_dir_for_model(label: str) -> str:
    suffix = fingerprint_config(label)[:6]
    return f"models/{_slug(label)}-{suffix}"


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _metric_dict(metrics) -> dict[str, float | None]:
    return {
        "mae": _finite_or_none(metrics.mae),
        "rmse": _finite_or_none(metrics.rmse),
        "smape": _finite_or_none(metrics.smape),
        "mase": _finite_or_none(metrics.mase),
        "wape": _finite_or_none(metrics.wape),
    }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    label: str
    factory: ModelFactory = field(repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("model label must be non-empty")
        normalize_json_value(self.metadata)

    def describe(self) -> tuple[str, dict[str, Any]]:
        model = self.factory()
        if not isinstance(model, ForecastModel):
            raise TypeError("model factory must return a ForecastModel")
        if not getattr(model, "name", "").strip():
            raise ValueError("forecast model must expose a non-empty name")

        configuration: dict[str, Any] = {}
        for key, value in getattr(model, "__dict__", {}).items():
            if key.startswith("_"):
                continue
            try:
                configuration[key] = normalize_json_value(value)
            except TypeError:
                continue
        return model.name, configuration


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    initial_train_size: int
    horizon: int = 1
    step: int | None = None
    baseline_model: str = "naive_last_value"
    missing_policy: MissingPolicy | str = "error"

    def __post_init__(self) -> None:
        if self.initial_train_size < 3:
            raise ValueError("initial_train_size must be at least 3")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.step is not None and self.step < 1:
            raise ValueError("step must be positive when provided")
        if not self.baseline_model.strip():
            raise ValueError("baseline_model must be non-empty")
        object.__setattr__(self, "missing_policy", normalize_missing_policy(self.missing_policy))

    @property
    def effective_step(self) -> int:
        return self.horizon if self.step is None else self.step

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_train_size": self.initial_train_size,
            "horizon": self.horizon,
            "step": self.effective_step,
            "baseline_model": self.baseline_model,
            "missing_policy": self.missing_policy,
            "ranking_metric": "rmse",
        }


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    run_id: str
    run_dir: Path
    manifest: Mapping[str, Any]
    leaderboard: LeaderboardResult


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _fold_metrics_frame(backtest) -> pd.DataFrame:
    rows = []
    for fold in backtest.folds:
        rows.append(
            {
                "fold": fold.fold,
                "train_start": _timestamp(fold.train_start),
                "train_end": _timestamp(fold.train_end),
                "test_start": _timestamp(fold.test_start),
                "test_end": _timestamp(fold.test_end),
                **_metric_dict(fold.metrics),
                "metadata_json": json.dumps(
                    normalize_json_value(fold.metadata),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ),
            }
        )
    return pd.DataFrame(rows)


def _forecast_frame(series: PreparedSeries, backtest) -> pd.DataFrame:
    rows = []
    for fold in backtest.folds:
        for timestamp, forecast in fold.forecast.items():
            rows.append(
                {
                    "fold": fold.fold,
                    "timestamp": _timestamp(pd.Timestamp(timestamp)),
                    "actual": float(series.values.loc[timestamp]),
                    "forecast": float(forecast),
                }
            )
    return pd.DataFrame(rows)


def _artifact_metadata(run_dir: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for relative in relative_paths:
        path = run_dir / relative
        result[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return result


def _replace_run_directory(staging: Path, destination: Path) -> None:
    backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    had_previous = destination.exists()
    if had_previous:
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if had_previous and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def record_experiment(
    series: PreparedSeries,
    model_specs: Sequence[ModelSpec],
    experiment: ExperimentSpec,
    *,
    artifact_root: Path = Path("artifacts"),
    code_revision: str | None = None,
    package_version: str | None = None,
) -> ExperimentRun:
    """Backtest configured models and persist one self-describing run directory."""

    if series.values.empty:
        raise ValueError("cannot record an experiment for an empty series")
    labels = [spec.label for spec in model_specs]
    if len(labels) != len(set(labels)):
        raise ValueError("model labels must be unique")
    if experiment.baseline_model not in labels:
        raise ValueError("experiment baseline_model must be present in model_specs")
    if len(model_specs) < 2:
        raise ValueError("an experiment requires a baseline and at least one challenger")

    descriptions = {spec.label: spec.describe() for spec in model_specs}
    implementations = {label: value[0] for label, value in descriptions.items()}
    configurations = {label: value[1] for label, value in descriptions.items()}
    factories = {spec.label: spec.factory for spec in model_specs}
    leaderboard = build_leaderboard(
        series,
        factories,
        initial_train_size=experiment.initial_train_size,
        horizon=experiment.horizon,
        step=experiment.step,
        baseline_model=experiment.baseline_model,
        missing_policy=experiment.missing_policy,
    )

    package_version = package_version or _package_version()
    code_revision = code_revision or os.getenv("GITHUB_SHA") or f"package:{package_version}"
    dataset_fingerprint = fingerprint_prepared_series(series)
    config_payload = {
        "evaluation": experiment.as_dict(),
        "models": [
            {
                "label": spec.label,
                "implementation": implementations[spec.label],
                "configuration": configurations[spec.label],
                "metadata": normalize_json_value(spec.metadata),
            }
            for spec in model_specs
        ],
        "package_version": package_version,
        "code_revision": code_revision,
    }
    config_fingerprint = fingerprint_config(config_payload)
    run_id = (
        f"{_slug(series.schema.name)}-"
        f"{dataset_fingerprint[:10]}-{config_fingerprint[:10]}"
    )

    runs_root = Path(artifact_root) / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / run_id
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=runs_root))

    try:
        relative_paths: list[str] = []
        leaderboard_path = "leaderboard.csv"
        _write_dataframe(staging / leaderboard_path, leaderboard.table)
        relative_paths.append(leaderboard_path)

        model_entries: dict[str, Any] = {}
        leaderboard_rows = leaderboard.table.set_index("model")
        for spec in model_specs:
            backtest = leaderboard.backtests[spec.label]
            model_dir = _artifact_dir_for_model(spec.label)
            fold_metrics_path = f"{model_dir}/fold_metrics.csv"
            forecasts_path = f"{model_dir}/forecasts.csv"
            _write_dataframe(staging / fold_metrics_path, _fold_metrics_frame(backtest))
            _write_dataframe(staging / forecasts_path, _forecast_frame(series, backtest))
            relative_paths.extend([fold_metrics_path, forecasts_path])

            row = leaderboard_rows.loc[spec.label]
            model_entries[spec.label] = {
                "implementation": implementations[spec.label],
                "configuration": configurations[spec.label],
                "metadata": normalize_json_value(spec.metadata),
                "folds": len(backtest.folds),
                "aggregate_metrics": _metric_dict(backtest.aggregate),
                "rank": int(row["rank"]),
                "rmse_vs_baseline_pct": _finite_or_none(row["rmse_vs_baseline_pct"]),
                "beats_baseline": bool(row["beats_baseline"]),
                "fold_metrics_path": fold_metrics_path,
                "forecasts_path": forecasts_path,
            }

        artifacts = _artifact_metadata(staging, relative_paths)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "code": {
                "project": _PROJECT_NAME,
                "package_version": package_version,
                "revision": code_revision,
                "python": platform.python_version(),
                "dependencies": _dependency_versions(),
            },
            "dataset": {
                "name": series.schema.name,
                "fingerprint_sha256": dataset_fingerprint,
                "timestamp_col": series.schema.timestamp_col,
                "target_col": series.schema.target_col,
                "frequency": series.schema.frequency,
                "aggregation": series.schema.aggregation,
                "timezone": series.schema.timezone,
                "known_future_regressors": list(series.schema.known_future_regressors),
                "source_rows": series.source_rows,
                "observations": len(series.values),
                "missing_periods": series.missing_periods,
                "start": _timestamp(pd.Timestamp(series.values.index[0])),
                "end": _timestamp(pd.Timestamp(series.values.index[-1])),
            },
            "evaluation": {
                **experiment.as_dict(),
                "config_fingerprint_sha256": config_fingerprint,
                "folds": len(next(iter(leaderboard.backtests.values())).folds),
            },
            "leaderboard": {
                "path": leaderboard_path,
                "rows": len(leaderboard.table),
            },
            "models": model_entries,
            "artifacts": artifacts,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        _replace_run_directory(staging, run_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return ExperimentRun(
        run_id=run_id,
        run_dir=run_dir,
        manifest=manifest,
        leaderboard=leaderboard,
    )
