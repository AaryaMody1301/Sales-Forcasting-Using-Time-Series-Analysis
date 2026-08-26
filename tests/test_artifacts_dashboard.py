import pandas as pd
import pytest

from sales_forecasting.artifacts import (
    ExperimentSpec,
    ManifestError,
    ModelSpec,
    fingerprint_config,
    fingerprint_prepared_series,
    load_run_manifest,
    record_experiment,
)
from sales_forecasting.dashboard.data import (
    discover_runs,
    load_fold_metrics,
    load_leaderboard,
    load_model_forecasts,
)
from sales_forecasting.data.schema import DatasetSchema, PreparedSeries
from sales_forecasting.evaluation.backtesting import BacktestFold, BacktestResult
from sales_forecasting.evaluation.metrics import ForecastMetrics
from sales_forecasting.evaluation.leaderboard import LeaderboardResult
from sales_forecasting.models.base import ForecastModel


class DummyModel(ForecastModel):
    name = "dummy"

    def fit(self, series):
        return self

    def forecast(self, horizon):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError


def prepared(values=(10.0, 11.0, 12.0, 13.0, 14.0)):
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return PreparedSeries(
        values=pd.Series(values, index=index, name="sales"),
        schema=DatasetSchema(
            name="demo_sales",
            timestamp_col="date",
            target_col="sales",
            frequency="D",
        ),
        source_rows=len(values),
        missing_periods=0,
    )


def fake_leaderboard(series, factories, **kwargs):
    index = series.values.index
    folds_by_model = {}
    rows = []
    for rank, (label, factory) in enumerate(factories.items(), start=1):
        implementation = factory().name
        metrics = ForecastMetrics(
            mae=float(rank),
            rmse=float(rank),
            smape=1.0,
            mase=1.0,
            wape=1.0,
        )
        fold = BacktestFold(
            fold=0,
            train_start=index[0],
            train_end=index[2],
            test_start=index[3],
            test_end=index[4],
            metrics=metrics,
            forecast=pd.Series([12.5, 13.5], index=index[3:5]),
        )
        folds_by_model[label] = BacktestResult(
            model_name=implementation,
            folds=(fold,),
            aggregate=metrics,
        )
        rows.append(
            {
                "rank": rank,
                "model": label,
                "implementation": implementation,
                "folds": 1,
                "mae": metrics.mae,
                "rmse": metrics.rmse,
                "smape": metrics.smape,
                "mase": metrics.mase,
                "wape": metrics.wape,
                "rmse_vs_baseline_pct": 0.0 if rank == 1 else 100.0,
                "beats_baseline": False,
            }
        )
    return LeaderboardResult(
        table=pd.DataFrame(rows),
        backtests=folds_by_model,
        baseline_model=kwargs["baseline_model"],
    )


def specs():
    return (
        ModelSpec("naive_last_value", DummyModel, metadata={"kind": "baseline"}),
        ModelSpec("challenger", DummyModel, metadata={"depth": 3}),
    )


def test_fingerprints_change_with_data_and_config():
    base = prepared()
    changed = prepared((10.0, 11.0, 12.0, 13.0, 99.0))
    assert fingerprint_prepared_series(base) != fingerprint_prepared_series(changed)
    assert fingerprint_config({"horizon": 1}) != fingerprint_config({"horizon": 2})


def test_record_experiment_is_deterministic_and_dashboard_reads_it(tmp_path, monkeypatch):
    monkeypatch.setattr("sales_forecasting.artifacts.store.build_leaderboard", fake_leaderboard)
    series = prepared()
    spec = ExperimentSpec(initial_train_size=3, horizon=2)

    first = record_experiment(
        series,
        specs(),
        spec,
        artifact_root=tmp_path,
        code_revision="abc123",
        package_version="0.4.0",
    )
    second = record_experiment(
        series,
        specs(),
        spec,
        artifact_root=tmp_path,
        code_revision="abc123",
        package_version="0.4.0",
    )

    assert first.run_id == second.run_id
    assert second.run_dir == tmp_path / "runs" / second.run_id
    manifest = load_run_manifest(second.run_dir / "manifest.json")
    assert manifest["dataset"]["fingerprint_sha256"] == fingerprint_prepared_series(series)
    assert manifest["evaluation"]["horizon"] == 2
    assert manifest["models"]["challenger"]["metadata"] == {"depth": 3}

    catalog = discover_runs(tmp_path)
    assert not catalog.errors
    assert [run.run_id for run in catalog.runs] == [second.run_id]
    run = catalog.runs[0]
    assert list(load_leaderboard(run)["model"]) == ["naive_last_value", "challenger"]
    assert len(load_fold_metrics(run, "challenger")) == 1
    forecasts = load_model_forecasts(run, "challenger")
    assert list(forecasts["actual"]) == [13.0, 14.0]
    assert list(forecasts["forecast"]) == [12.5, 13.5]


def test_dashboard_rejects_tampered_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr("sales_forecasting.artifacts.store.build_leaderboard", fake_leaderboard)
    run = record_experiment(
        prepared(),
        specs(),
        ExperimentSpec(initial_train_size=3, horizon=2),
        artifact_root=tmp_path,
        code_revision="abc123",
        package_version="0.4.0",
    )
    leaderboard = run.run_dir / "leaderboard.csv"
    leaderboard.write_text(leaderboard.read_text() + "tampered\n")

    handle = discover_runs(tmp_path).runs[0]
    with pytest.raises(ManifestError, match="checksum mismatch"):
        load_leaderboard(handle)


def test_run_id_changes_with_evaluation_config(tmp_path, monkeypatch):
    monkeypatch.setattr("sales_forecasting.artifacts.store.build_leaderboard", fake_leaderboard)
    series = prepared((1, 2, 3, 4, 5, 6))
    run1 = record_experiment(
        series,
        specs(),
        ExperimentSpec(initial_train_size=3, horizon=1),
        artifact_root=tmp_path,
        code_revision="abc123",
        package_version="0.4.0",
    )
    run2 = record_experiment(
        series,
        specs(),
        ExperimentSpec(initial_train_size=3, horizon=2),
        artifact_root=tmp_path,
        code_revision="abc123",
        package_version="0.4.0",
    )
    assert run1.run_id != run2.run_id
