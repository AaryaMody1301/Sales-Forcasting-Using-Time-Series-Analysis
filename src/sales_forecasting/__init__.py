"""Canonical package for leakage-aware, reproducible sales forecasting."""

from .artifacts import (
    ExperimentRun,
    ExperimentSpec,
    ManifestError,
    ModelSpec,
    fingerprint_config,
    fingerprint_prepared_series,
    load_run_manifest,
    record_experiment,
)
from .data import (
    CAR_PRICES_DAILY_MEDIAN,
    DatasetContractError,
    DatasetSchema,
    MissingPolicy,
    PreparedSeries,
    apply_training_missing_policy,
    attach_known_future_regressors,
    future_regressors_for_horizon,
    get_builtin_schema,
    normalize_missing_policy,
    prepare_time_series,
)
from .evaluation import (
    NestedTunedForecaster,
    ValidationWeightedEnsemble,
    build_leaderboard,
    calculate_metrics,
    expanding_window_backtest,
)
from .features import FeatureSpec, build_feature_row, build_supervised_frame
from .models import (
    ARIMAForecaster,
    ETSForecaster,
    GradientBoostingForecaster,
    LastValueNaiveModel,
    ProphetForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)

__all__ = [
    "ARIMAForecaster",
    "CAR_PRICES_DAILY_MEDIAN",
    "DatasetContractError",
    "DatasetSchema",
    "ETSForecaster",
    "ExperimentRun",
    "ExperimentSpec",
    "FeatureSpec",
    "GradientBoostingForecaster",
    "LastValueNaiveModel",
    "ManifestError",
    "MissingPolicy",
    "ModelSpec",
    "NestedTunedForecaster",
    "PreparedSeries",
    "ProphetForecaster",
    "RandomForestForecaster",
    "ValidationWeightedEnsemble",
    "XGBoostForecaster",
    "apply_training_missing_policy",
    "attach_known_future_regressors",
    "build_feature_row",
    "build_leaderboard",
    "build_supervised_frame",
    "calculate_metrics",
    "expanding_window_backtest",
    "fingerprint_config",
    "fingerprint_prepared_series",
    "future_regressors_for_horizon",
    "get_builtin_schema",
    "load_run_manifest",
    "normalize_missing_policy",
    "prepare_time_series",
    "record_experiment",
]

__version__ = "0.6.0"
