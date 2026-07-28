from importlib.util import find_spec

import pandas as pd
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
        "model_selection": "fixed_regularization",
        "alpha": 1.0,
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
    assert set(result["explanation"]["feature_coefficients"]) == {
        "return_5",
        "return_20",
        "return_60",
        "volatility_20",
        "drawdown_60",
    }
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


def test_chronological_split_purges_labels_that_reach_the_holdout():
    from research.qlib_validation.forecast import _split

    sessions = pd.date_range("2026-01-01", periods=10, freq="D")
    rows = [
        {
            "key": key,
            "decision_session": decision.date().isoformat(),
            "target_session": sessions[min(index + 1, len(sessions) - 1)]
            .date()
            .isoformat(),
        }
        for index, decision in enumerate(sessions)
        for key in ("KR/KOSPI", "US/SPY")
    ]
    labeled = pd.DataFrame(rows)
    labeled.loc[
        (labeled["key"] == "KR/KOSPI")
        & (labeled["decision_session"] == "2026-01-07"),
        "target_session",
    ] = "2026-01-09"

    train, holdout = _split(
        labeled,
        test_sessions=2,
        embargo_sessions=1,
        minimum_train_dates=3,
    )
    assert train["target_session"].max() < holdout["decision_session"].min()


def test_ridge_uses_full_outer_train_and_all_labeled_rows_for_current_scores(
    snapshot_factory, monkeypatch
):
    from research.qlib_validation import forecast

    snapshot = snapshot_factory(days=360)
    labeled, _ = forecast._feature_frames(snapshot, horizon_sessions=20)
    outer_train, _ = forecast._split(
        labeled,
        test_sessions=80,
        embargo_sessions=20,
        minimum_train_dates=100,
    )
    calls = []

    def fake_ridge_predict(**kwargs):
        calls.append(kwargs)
        return pd.Series(0.0, index=kwargs["test_features"].index), {
            "intercept": 0.0,
            "feature_coefficients": {
                name: 0.0 for name in forecast.FEATURE_NAMES
            },
        }

    monkeypatch.setattr(forecast, "_qlib_native_ridge_predict", fake_ridge_predict)

    result = forecast.forecast_snapshot(
        snapshot,
        horizon_sessions=20,
        test_sessions=80,
        embargo_sessions=20,
        minimum_train_dates=100,
    )

    assert len(calls) == 2
    holdout_call, current_call = calls
    assert len(holdout_call["train_labels"]) == len(outer_train)
    assert holdout_call["alpha"] == 1.0
    assert len(current_call["train_labels"]) == len(labeled)
    assert current_call["alpha"] == 1.0
    assert result["forecast_protocol"]["model"]["alpha"] == 1.0
