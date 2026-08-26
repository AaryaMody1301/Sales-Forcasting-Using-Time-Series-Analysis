"""Lag, rolling, and calendar features that never read the target being predicted."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Feature configuration shared by training and recursive forecasting."""

    lags: tuple[int, ...] = (1, 7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 14, 28)
    calendar: bool = True

    def __post_init__(self) -> None:
        if not self.lags:
            raise ValueError("at least one lag is required")
        if any(not isinstance(lag, int) or isinstance(lag, bool) or lag < 1 for lag in self.lags):
            raise ValueError("lags must be positive integers")
        if any(
            not isinstance(window, int) or isinstance(window, bool) or window < 2
            for window in self.rolling_windows
        ):
            raise ValueError("rolling windows must be integers >= 2")
        if len(set(self.lags)) != len(self.lags):
            raise ValueError("lags must be unique")
        if len(set(self.rolling_windows)) != len(self.rolling_windows):
            raise ValueError("rolling windows must be unique")

    @property
    def minimum_history(self) -> int:
        return max((*self.lags, *self.rolling_windows), default=max(self.lags))


def _calendar_features(timestamp: pd.Timestamp) -> dict[str, float]:
    day_of_year = timestamp.dayofyear
    return {
        "month": float(timestamp.month),
        "day_of_week": float(timestamp.dayofweek),
        "day_of_year": float(day_of_year),
        "month_sin": float(np.sin(2 * np.pi * timestamp.month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * timestamp.month / 12.0)),
        "dow_sin": float(np.sin(2 * np.pi * timestamp.dayofweek / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * timestamp.dayofweek / 7.0)),
        "doy_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "doy_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
    }


def build_feature_row(
    history: pd.Series,
    timestamp: pd.Timestamp,
    spec: FeatureSpec,
) -> pd.Series:
    """Build features for ``timestamp`` using only values observed before it."""

    if len(history) < spec.minimum_history:
        raise ValueError(
            f"need at least {spec.minimum_history} historical observations, got {len(history)}"
        )
    if history.isna().any():
        raise ValueError("history cannot contain missing values")

    features: dict[str, float] = {}
    for lag in spec.lags:
        features[f"lag_{lag}"] = float(history.iloc[-lag])

    for window in spec.rolling_windows:
        recent = history.iloc[-window:].astype(float)
        features[f"rolling_mean_{window}"] = float(recent.mean())
        features[f"rolling_std_{window}"] = float(recent.std(ddof=0))
        features[f"rolling_min_{window}"] = float(recent.min())
        features[f"rolling_max_{window}"] = float(recent.max())

    if spec.calendar:
        features.update(_calendar_features(pd.Timestamp(timestamp)))

    return pd.Series(features, dtype=float)


def build_supervised_frame(
    values: pd.Series,
    spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create X/y rows where X[t] contains target history only through t-1."""

    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("values must use a DatetimeIndex")
    if values.isna().any():
        raise ValueError("values cannot contain missing observations")
    if len(values) <= spec.minimum_history:
        raise ValueError("series is too short for the configured features")

    rows: list[pd.Series] = []
    targets: list[float] = []
    indices: list[pd.Timestamp] = []

    for position in range(spec.minimum_history, len(values)):
        timestamp = pd.Timestamp(values.index[position])
        history = values.iloc[:position]
        rows.append(build_feature_row(history, timestamp, spec))
        targets.append(float(values.iloc[position]))
        indices.append(timestamp)

    X = pd.DataFrame(rows, index=pd.DatetimeIndex(indices))
    y = pd.Series(targets, index=X.index, name=values.name, dtype=float)
    return X, y
