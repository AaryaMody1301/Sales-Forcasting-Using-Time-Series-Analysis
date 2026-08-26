"""Run-manifest loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

MANIFEST_SCHEMA_VERSION = 1


class ManifestError(ValueError):
    """Raised when a run manifest or referenced artifact is invalid."""


def validate_manifest(data: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "run_id",
        "created_at_utc",
        "status",
        "code",
        "dataset",
        "evaluation",
        "leaderboard",
        "models",
        "artifacts",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ManifestError("manifest is missing required keys: " + ", ".join(missing))
    if data["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"unsupported manifest schema version {data['schema_version']!r}"
        )
    if data["status"] != "completed":
        raise ManifestError("only completed runs can be loaded as benchmark artifacts")
    if not isinstance(data["models"], Mapping) or not data["models"]:
        raise ManifestError("manifest must contain at least one model")
    if not isinstance(data["artifacts"], Mapping):
        raise ManifestError("manifest artifacts must be a mapping")
    return dict(data)


def load_run_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read manifest at {path}") from exc
    if not isinstance(data, Mapping):
        raise ManifestError("manifest root must be a JSON object")
    manifest = validate_manifest(data)
    if path.name == "manifest.json" and path.parent.name != manifest["run_id"]:
        raise ManifestError("manifest run_id does not match its run directory")
    return manifest
