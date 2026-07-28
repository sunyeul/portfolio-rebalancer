from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    find_spec("qlib") is None,
    reason="Qlib research environment only",
)


def test_tracking_run_uses_disposable_artifact_location(tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient
    from qlib.config import C

    from research.qlib_validation.forecast import _qlib_tracking_run

    monkeypatch.chdir(tmp_path)
    with _qlib_tracking_run() as tracking_root:
        experiment = MlflowClient(
            tracking_uri=C.exp_manager["kwargs"]["uri"]
        ).get_experiment_by_name("research-forecast")
        assert tracking_root is not None
        assert experiment is not None
        assert experiment.artifact_location.startswith(tracking_root.as_uri())
    assert not (tmp_path / "mlruns").exists()


def test_forecast_uses_chronological_holdout_and_emits_latest_scores(
    snapshot_factory,
    tmp_path,
    monkeypatch,
):
    from research.qlib_validation.forecast import forecast_snapshot

    monkeypatch.chdir(tmp_path)
    repository_mlruns = Path(__file__).resolve().parents[2] / "mlruns"
    repository_artifacts_before = set(repository_mlruns.rglob("code_status.txt"))
    snapshot = snapshot_factory(days=360)

    result = forecast_snapshot(
        snapshot,
        horizon_sessions=20,
        test_sessions=80,
        embargo_sessions=20,
        minimum_train_dates=100,
    )

    assert result["qlib_static_loader"] == {"matched": True}
    assert result["forecast_protocol"]["model"] == {
        "family": "lightgbm_regression",
        "backend": "qlib_native",
        "qlib_component": "qlib.contrib.model.gbdt.LGBModel",
        "model_selection": "chronological_inner_validation",
        "num_boost_round_cap": 120,
        "early_stopping_rounds": 20,
        "random_seed": 1729,
        "num_threads": 1,
        "experiment_tracking": "isolated_temporary_sqlite",
    }
    assert result["forecast_protocol"]["target"] == (
        "native_close_return_after_horizon_sessions"
    )
    assert result["forecast_protocol"]["currency_conversion"] == "not_applied"
    assert result["forecast_protocol"]["overlapping_horizon_labels"] is True
    assert (
        result["holdout"]["train_last_session"]
        < result["holdout"]["test_first_session"]
    )
    assert result["holdout"]["row_count"] > 0
    assert result["holdout"]["metrics"]["model_rmse"] >= 0
    assert result["holdout"]["metrics"]["baseline_rmse"] >= 0
    model_selection = result["holdout"]["model_selection"]
    assert model_selection["embargo_sessions"] == 20
    assert model_selection["validation_sessions"] == 60
    assert model_selection["selected_boosting_rounds"] >= 1
    assert (
        model_selection["train_last_session"]
        < model_selection["validation_first_session"]
        <= model_selection["validation_last_session"]
        < result["holdout"]["test_first_session"]
    )
    assert len(result["current_forecasts"]) == len(snapshot.policy_specs)
    assert all(
        item["horizon_sessions"] == 20
        and isinstance(item["predicted_return"], float)
        and item["decision_session"] > result["holdout"]["test_last_session"]
        for item in result["current_forecasts"]
    )

    forbidden = {"buy", "sell", "execute", "order_size", "status"}

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert forbidden.isdisjoint(keys(result))
    assert not (tmp_path / "mlruns").exists()
    assert (
        set(repository_mlruns.rglob("code_status.txt")) == repository_artifacts_before
    )


def test_forecast_rejects_a_holdout_without_a_label_embargo(snapshot_factory):
    from research.qlib_validation.forecast import ForecastError, forecast_snapshot

    with pytest.raises(ForecastError, match="insufficient chronological history"):
        forecast_snapshot(
            snapshot_factory(days=160),
            horizon_sessions=20,
            test_sessions=80,
            embargo_sessions=20,
            minimum_train_dates=100,
        )
