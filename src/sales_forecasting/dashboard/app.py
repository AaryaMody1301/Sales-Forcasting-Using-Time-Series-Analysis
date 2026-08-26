"""Streamlit dashboard backed only by verified run manifests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sales_forecasting.artifacts import ManifestError
from sales_forecasting.dashboard.data import (
    RunHandle,
    discover_runs,
    load_fold_metrics,
    load_leaderboard,
    load_model_forecasts,
)


def _run_label(run: RunHandle) -> str:
    dataset = run.manifest["dataset"]["name"]
    created = str(run.manifest["created_at_utc"]).replace("T", " ").split("+")[0]
    return f"{created} | {dataset} | {run.run_id}"


def main() -> None:
    st.set_page_config(page_title="Forecasting Runs", layout="wide")
    st.title("Sales Forecasting Experiment Dashboard")
    st.caption("Only completed, checksum-verified canonical runs are displayed.")

    artifact_root = Path(st.sidebar.text_input("Artifact root", value="artifacts"))
    if st.sidebar.button("Refresh runs"):
        st.rerun()

    catalog = discover_runs(artifact_root)
    for error in catalog.errors:
        st.warning(f"Skipped invalid run: {error}")

    if not catalog.runs:
        st.info(
            "No canonical experiment runs were found. Record one with "
            "sales_forecasting.record_experiment(), then refresh this page."
        )
        return

    run = st.selectbox("Experiment run", catalog.runs, format_func=_run_label)
    manifest = run.manifest
    dataset = manifest["dataset"]
    evaluation = manifest["evaluation"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", dataset["name"])
    col2.metric("Horizon", evaluation["horizon"])
    col3.metric("Folds", evaluation["folds"])
    col4.metric("Frequency", dataset["frequency"])

    st.caption(
        f"Run ID: `{run.run_id}`  |  "
        f"Dataset SHA-256: `{dataset['fingerprint_sha256']}`  |  "
        f"Config SHA-256: `{evaluation['config_fingerprint_sha256']}`"
    )

    try:
        leaderboard = load_leaderboard(run)
    except ManifestError as exc:
        st.error(f"Leaderboard integrity check failed: {exc}")
        return

    st.subheader("Leaderboard")
    st.dataframe(leaderboard, width="stretch", hide_index=True)

    model_labels = list(manifest["models"])
    model_label = st.selectbox("Model details", model_labels)
    model_entry = manifest["models"][model_label]

    metric_cols = st.columns(5)
    for column, metric_name in zip(
        metric_cols,
        ("mae", "rmse", "smape", "mase", "wape"),
    ):
        value = model_entry["aggregate_metrics"][metric_name]
        display = "n/a" if value is None else f"{value:.4g}"
        column.metric(metric_name.upper(), display)

    try:
        fold_metrics = load_fold_metrics(run, model_label)
        forecasts = load_model_forecasts(run, model_label)
    except ManifestError as exc:
        st.error(f"Model artifact integrity check failed: {exc}")
        return

    st.subheader("Fold metrics")
    st.dataframe(fold_metrics, width="stretch", hide_index=True)

    if forecasts.empty:
        st.info("This run has no forecast rows for the selected model.")
        return

    fold_options = sorted(int(value) for value in forecasts["fold"].unique())
    selected_fold = st.selectbox("Forecast fold", fold_options)
    fold_forecast = forecasts.loc[
        forecasts["fold"] == selected_fold,
        ["timestamp", "actual", "forecast"],
    ].copy()
    fold_forecast["timestamp"] = pd.to_datetime(fold_forecast["timestamp"])

    st.subheader("Actual vs forecast")
    st.line_chart(
        fold_forecast,
        x="timestamp",
        y=["actual", "forecast"],
        width="stretch",
    )
    st.dataframe(fold_forecast, width="stretch", hide_index=True)
