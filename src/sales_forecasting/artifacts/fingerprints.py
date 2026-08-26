"""Stable fingerprints for experiment inputs and configuration."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sales_forecasting.data.schema import PreparedSeries


def normalize_json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): normalize_json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return normalize_json_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration values must be finite JSON numbers")
        return value
    raise TypeError(f"unsupported manifest value type: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    normalized = normalize_json_value(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def fingerprint_config(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def fingerprint_prepared_series(series: PreparedSeries) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(asdict(series.schema)))
    digest.update(b"\0series-v2\0")
    for timestamp, raw_value in series.values.items():
        digest.update(struct.pack(">q", pd.Timestamp(timestamp).value))
        if pd.isna(raw_value):
            digest.update(b"N")
        else:
            digest.update(b"V")
            digest.update(struct.pack(">d", float(raw_value)))
    digest.update(b"\0known-future-regressors\0")
    if series.future_regressors is None:
        digest.update(b"none")
    else:
        for timestamp, row in series.future_regressors.iterrows():
            digest.update(struct.pack(">q", pd.Timestamp(timestamp).value))
            for column in series.future_regressors.columns:
                digest.update(column.encode("utf-8"))
                digest.update(b"\0")
                digest.update(struct.pack(">d", float(row[column])))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
