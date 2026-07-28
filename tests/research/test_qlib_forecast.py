from importlib.util import find_spec

import pytest


pytestmark = pytest.mark.skipif(
    find_spec("qlib") is None,
    reason="Qlib research environment only",
)


def test_forecast_uses_chronological_holdout_and_emits_latest_scores(
    snapshot_factory,
):
    from research.qlib_validation.forecast import forecast_snapshot

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
        "family": "ridge_regression",
        "backend": "qlib_native",
        "qlib_component": "qlib.contrib.model.linear.LinearModel",
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
