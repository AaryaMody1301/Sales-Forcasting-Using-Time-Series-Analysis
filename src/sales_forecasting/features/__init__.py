"""Leakage-safe feature generation for autoregressive ML models."""

from .lags import FeatureSpec, build_feature_row, build_supervised_frame

__all__ = ["FeatureSpec", "build_feature_row", "build_supervised_frame"]
