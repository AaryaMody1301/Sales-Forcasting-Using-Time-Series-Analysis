"""Manifest-driven dashboard data access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sales_forecasting.artifacts.fingerprints import sha256_file
from sales_forecasting.artifacts.manifest import ManifestError, load_run_manifest


@dataclass(frozen=True, slots=True)
class RunHandle:
    run_dir: Path
    manifest: Mapping[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.manifest["run_id"])


@dataclass(frozen=True, slots=True)
class RunCatalog:
    runs: tuple[RunHandle, ...]
    errors: tuple[str, ...]


def discover_runs(artifact_root: Path = Path("artifacts")) -> RunCatalog:
    runs_root = Path(artifact_root) / "runs"
    if not runs_root.exists():
        return RunCatalog(runs=(), errors=())

    runs: list[RunHandle] = []
    errors: list[str] = []
    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        try:
            manifest = load_run_manifest(manifest_path)
        except ManifestError as exc:
            errors.append(f"{manifest_path}: {exc}")
            continue
        runs.append(RunHandle(run_dir=manifest_path.parent, manifest=manifest))

    runs.sort(
        key=lambda run: str(run.manifest.get("created_at_utc", "")),
        reverse=True,
    )
    return RunCatalog(runs=tuple(runs), errors=tuple(errors))


def _safe_artifact_path(run: RunHandle, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ManifestError("artifact paths must be relative to the run directory")

    root = run.run_dir.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ManifestError("artifact path escapes the run directory")
    if not path.is_file():
        raise ManifestError(f"artifact does not exist: {relative_path}")
    return path


def _verified_artifact_path(run: RunHandle, relative_path: str) -> Path:
    metadata = run.manifest.get("artifacts", {}).get(relative_path)
    if not isinstance(metadata, Mapping) or "sha256" not in metadata:
        raise ManifestError(f"manifest has no checksum for artifact {relative_path!r}")
    path = _safe_artifact_path(run, relative_path)
    actual = sha256_file(path)
    expected = str(metadata["sha256"])
    if actual != expected:
        raise ManifestError(
            f"artifact checksum mismatch for {relative_path!r}: expected {expected}, got {actual}"
        )
    return path


def _read_verified_csv(run: RunHandle, relative_path: str) -> pd.DataFrame:
    return pd.read_csv(_verified_artifact_path(run, relative_path))


def load_leaderboard(run: RunHandle) -> pd.DataFrame:
    relative = str(run.manifest["leaderboard"]["path"])
    return _read_verified_csv(run, relative)


def _model_entry(run: RunHandle, model_label: str) -> Mapping[str, Any]:
    models = run.manifest["models"]
    if model_label not in models:
        raise ManifestError(f"model {model_label!r} is not present in run {run.run_id}")
    entry = models[model_label]
    if not isinstance(entry, Mapping):
        raise ManifestError(f"model entry {model_label!r} is invalid")
    return entry


def load_model_forecasts(run: RunHandle, model_label: str) -> pd.DataFrame:
    relative = str(_model_entry(run, model_label)["forecasts_path"])
    frame = _read_verified_csv(run, relative)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame


def load_fold_metrics(run: RunHandle, model_label: str) -> pd.DataFrame:
    relative = str(_model_entry(run, model_label)["fold_metrics_path"])
    return _read_verified_csv(run, relative)
