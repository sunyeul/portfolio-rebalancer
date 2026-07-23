import json
from datetime import datetime, timedelta, timezone

import pytest

from services.dynamic_allocation import (
    REGIME_TARGETS,
    DynamicAllocationError,
    build_neutral_policy,
    evaluate_dynamic_allocation,
    scale_policy_to_regime,
)
from services.policy_validation import validate_policy


NOW = datetime(2026, 7, 24, tzinfo=timezone.utc)


def _instrument(country, symbol, layer, target):
    return {
        "market_country": country,
        "symbol": symbol,
        "layer": layer,
        "minimum": 0.0,
        "target": target,
        "maximum": max(target, 0.30),
    }


def _active_policy():
    return {
        "cash_reserve": {"minimum": 0.10, "target": 0.15, "maximum": 0.20},
        "performance": {
            "annual_return_target": 0.10,
            "measurement": "ytd_twr",
            "minimum_history_days": 365,
        },
        "risk_review": {
            "lookback_sessions": 252,
            "minimum_history_points": 200,
            "max_data_age_days": 7,
            "max_gap_days": 7,
            "account_drawdown_review": -0.15,
            "instrument_drawdown_review": {
                "core": -0.25,
                "satellite": -0.20,
                "experiment": -0.15,
            },
        },
        "cadence": {"observation": "weekly", "inspection": "monthly"},
        "layers": {
            "core": {"minimum": 0.70, "target": 0.75, "maximum": 0.80},
            "satellite": {"minimum": 0.15, "target": 0.20, "maximum": 0.25},
            "experiment": {"minimum": 0.02, "target": 0.05, "maximum": 0.08},
        },
        "instruments": [
            _instrument("KR", "000660", "satellite", 0.015),
            _instrument("KR", "005930", "satellite", 0.015),
            _instrument("KR", "069500", "core", 0.08),
            _instrument("US", "AMZN", "satellite", 0.025),
            _instrument("US", "GLD", "satellite", 0.025),
            _instrument("US", "GOOGL", "satellite", 0.035),
            _instrument("US", "MU", "satellite", 0.03),
            _instrument("US", "NBIS", "satellite", 0.01),
            _instrument("US", "NVDA", "satellite", 0.02),
            _instrument("US", "QQQ", "core", 0.12),
            _instrument("US", "SCHD", "core", 0.10),
            _instrument("US", "SOXL", "experiment", 0.03),
            _instrument("US", "SPCX", "experiment", 0.02),
            _instrument("US", "SPY", "core", 0.25),
            _instrument("US", "TSLA", "satellite", 0.025),
            _instrument("US", "VOO", "core", 0.20),
        ],
    }


def _by_symbol(policy, symbol):
    return next(item for item in policy["instruments"] if item["symbol"] == symbol)


def _layer_total(policy, layer, bound="target"):
    return sum(item[bound] for item in policy["instruments"] if item["layer"] == layer)


def test_build_neutral_policy_reclassifies_gld_and_preserves_layer_ratios():
    result = build_neutral_policy(_active_policy())

    assert result["cash_reserve"] == {
        "minimum": 0.03,
        "target": 0.05,
        "maximum": 0.10,
    }
    assert result["layers"] == {
        "core": {"minimum": 0.50, "target": 0.60, "maximum": 0.70},
        "satellite": {"minimum": 0.28, "target": 0.38, "maximum": 0.48},
        "experiment": {"minimum": 0.00, "target": 0.02, "maximum": 0.04},
    }
    gld = _by_symbol(result, "GLD")
    assert gld["layer"] == "core"
    assert gld["target"] == pytest.approx(0.10)
    assert _by_symbol(result, "SPY")["target"] / _by_symbol(result, "VOO")[
        "target"
    ] == pytest.approx(0.25 / 0.20)
    assert _by_symbol(result, "GOOGL")["target"] / _by_symbol(result, "MU")[
        "target"
    ] == pytest.approx(0.035 / 0.03)
    assert _by_symbol(result, "SOXL")["target"] / _by_symbol(result, "SPCX")[
        "target"
    ] == pytest.approx(3 / 2)
    for layer in ("core", "satellite", "experiment"):
        assert _layer_total(result, layer) == pytest.approx(
            result["layers"][layer]["target"]
        )
        assert _layer_total(result, layer, "minimum") == pytest.approx(
            result["layers"][layer]["minimum"]
        )
        assert _layer_total(result, layer, "maximum") == pytest.approx(
            result["layers"][layer]["maximum"]
        )
    identities = [
        (item["market_country"], item["symbol"]) for item in result["instruments"]
    ]
    assert validate_policy(result, identities) == result


@pytest.mark.parametrize("regime", ["risk_on", "neutral", "risk_off"])
def test_scale_policy_to_regime_preserves_within_layer_shares(regime):
    neutral = build_neutral_policy(_active_policy())

    result = scale_policy_to_regime(neutral, regime)

    assert result["cash_reserve"]["target"] == REGIME_TARGETS[regime]["cash_target"]
    for layer, target in REGIME_TARGETS[regime]["layers"].items():
        assert result["layers"][layer]["target"] == target
        assert _layer_total(result, layer) == pytest.approx(target)
    assert _by_symbol(result, "SPY")["target"] / _by_symbol(result, "VOO")[
        "target"
    ] == pytest.approx(0.25 / 0.20)


def test_build_neutral_policy_requires_gld():
    policy = _active_policy()
    policy["instruments"] = [
        item for item in policy["instruments"] if item["symbol"] != "GLD"
    ]

    with pytest.raises(DynamicAllocationError, match="GLD"):
        build_neutral_policy(policy)


def test_build_neutral_policy_requires_each_source_layer():
    policy = _active_policy()
    policy["instruments"] = [
        item for item in policy["instruments"] if item["layer"] != "experiment"
    ]

    with pytest.raises(DynamicAllocationError, match="experiment"):
        build_neutral_policy(policy)


def _candles(values, *, end=NOW):
    start = end - timedelta(days=len(values) - 1)
    return [
        {
            "candle_at": (start + timedelta(days=index)).isoformat(),
            "close_price": value,
        }
        for index, value in enumerate(values)
    ]


def _series_map(values):
    return {
        key: _candles(values) for key in ("US/SPY", "US/QQQ", "KR/KOSPI", "KR/KOSDAQ")
    }


def test_broad_positive_trend_proposes_risk_on_without_execution_fields():
    policy = build_neutral_policy(_active_policy())
    values = [100 + index * 0.5 for index in range(220)]

    result = evaluate_dynamic_allocation(
        _series_map(values),
        active_policy=policy,
        last_change_at="2026-05-01T00:00:00+00:00",
        now=NOW,
    )

    assert result["status"] == "Review"
    assert result["candidate_state"] == "candidate"
    assert result["regime"] == "risk_on"
    assert result["proposed_targets"] == REGIME_TARGETS["risk_on"]
    encoded = json.dumps(result).lower()
    assert "buy" not in encoded
    assert "sell" not in encoded


def test_broad_negative_trend_proposes_risk_off():
    policy = build_neutral_policy(_active_policy())
    values = [220 - index * 0.5 for index in range(220)]

    result = evaluate_dynamic_allocation(
        _series_map(values),
        active_policy=policy,
        last_change_at="2026-05-01T00:00:00+00:00",
        now=NOW,
    )

    assert result["regime"] == "risk_off"
    assert result["candidate_state"] == "candidate"


def test_mixed_market_trend_is_neutral_and_aligned_policy_is_ok():
    policy = build_neutral_policy(_active_policy())
    rising = [100 + index * 0.5 for index in range(220)]
    falling = [220 - index * 0.5 for index in range(220)]
    series = {
        "US/SPY": _candles(rising),
        "US/QQQ": _candles(falling),
        "KR/KOSPI": _candles(rising),
        "KR/KOSDAQ": _candles(falling),
    }

    result = evaluate_dynamic_allocation(series, active_policy=policy, now=NOW)

    assert result["regime"] == "neutral"
    assert result["status"] == "OK"
    assert result["candidate_state"] == "observe"


def test_severe_risk_weight_overrides_positive_trend():
    policy = build_neutral_policy(_active_policy())
    calm_rising = [100 + index * 0.2 for index in range(220)]
    volatile = [100 + index * 0.2 for index in range(200)] + [80, 125] * 10
    series = _series_map(calm_rising)
    series["US/SPY"] = _candles(volatile)
    series["US/QQQ"] = _candles(volatile)

    result = evaluate_dynamic_allocation(
        series,
        active_policy=policy,
        last_change_at="2026-05-01T00:00:00+00:00",
        now=NOW,
    )

    assert result["severe_risk_weight"] == pytest.approx(0.60)
    assert result["regime"] == "risk_off"


def test_cooldown_reports_regime_without_candidate():
    policy = build_neutral_policy(_active_policy())
    values = [100 + index * 0.5 for index in range(220)]

    result = evaluate_dynamic_allocation(
        _series_map(values),
        active_policy=policy,
        last_change_at="2026-07-10T00:00:00+00:00",
        now=NOW,
    )

    assert result["regime"] == "risk_on"
    assert result["cooling"] is True
    assert result["status"] == "Watch"
    assert result["candidate_state"] == "observe"
    assert result["proposed_policy"] is not None


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda series: series.update({"KR/KOSDAQ": []}),
            "market_history_insufficient",
        ),
        (
            lambda series: series["KR/KOSDAQ"].__setitem__(
                -1,
                {
                    "candle_at": series["KR/KOSDAQ"][-2]["candle_at"],
                    "close_price": 100.0,
                },
            ),
            "market_data_invalid",
        ),
        (
            lambda series: series.update(
                {"KR/KOSDAQ": _candles([100.0] * 220, end=NOW - timedelta(days=8))}
            ),
            "market_data_stale",
        ),
    ],
)
def test_required_benchmark_quality_failure_fails_closed(mutator, reason):
    policy = build_neutral_policy(_active_policy())
    series = _series_map([100.0] * 220)
    mutator(series)

    result = evaluate_dynamic_allocation(series, active_policy=policy, now=NOW)

    assert result["status"] == "Watch"
    assert result["candidate_state"] == "observe"
    assert result["reason"] == reason
    assert result["proposed_policy"] is None
    assert result["failed_benchmarks"] == ["KR/KOSDAQ"]
