import pytest

from research.qlib_validation.contracts import SourceSnapshot
from research.qlib_validation.metrics import (
    block_bootstrap_effect,
    build_forward_observations,
    downside_metrics,
    parse_signal_rule,
    signal_verdict,
)
from research.qlib_validation.replay import replay_regimes


def test_downside_metrics_measure_drawdown_volatility_worst_day_and_recovery():
    result = downside_metrics([100.0, 90.0, 80.0, 90.0, 100.0])
    assert result["max_drawdown"] == pytest.approx(-0.2)
    assert result["worst_daily_return"] == pytest.approx(-1.0 / 9.0)
    assert result["recovery_sessions"] == 2
    assert result["annualized_volatility"] > 0


def test_block_bootstrap_is_seeded_and_reproducible():
    observations = [
        (
            "risk_off" if index % 3 == 0 else "neutral",
            -0.12 if index % 3 == 0 else -0.03,
        )
        for index in range(18)
    ]
    first = block_bootstrap_effect(observations, samples=200, seed=20260728)
    second = block_bootstrap_effect(observations, samples=200, seed=20260728)
    assert first == second
    assert first["estimate"] < 0.0


def test_signal_supported_requires_three_episodes_and_negative_upper_ci():
    effects = {
        ("benchmarks", 20): {"estimate": -0.05, "ci_low": -0.08, "ci_high": -0.01},
        ("benchmarks", 60): {"estimate": -0.07, "ci_low": -0.10, "ci_high": -0.02},
        ("policy_instruments", 20): {
            "estimate": -0.04,
            "ci_low": -0.07,
            "ci_high": -0.01,
        },
        ("policy_instruments", 60): {
            "estimate": -0.06,
            "ci_low": -0.09,
            "ci_high": -0.02,
        },
    }
    assert (
        signal_verdict(effects, risk_off_episodes=3, complete_coverage=True)["verdict"]
        == "supported"
    )
    assert (
        signal_verdict(effects, risk_off_episodes=2, complete_coverage=True)["verdict"]
        == "inconclusive"
    )


def test_non_negative_effect_is_not_supported_when_data_is_complete():
    effects = {
        ("benchmarks", 20): {"estimate": 0.01, "ci_low": -0.02, "ci_high": 0.03},
        ("benchmarks", 60): {"estimate": -0.02, "ci_low": -0.05, "ci_high": 0.01},
        ("policy_instruments", 20): {
            "estimate": -0.01,
            "ci_low": -0.04,
            "ci_high": 0.02,
        },
        ("policy_instruments", 60): {
            "estimate": -0.03,
            "ci_low": -0.06,
            "ci_high": 0.01,
        },
    }
    assert (
        signal_verdict(effects, risk_off_episodes=3, complete_coverage=True)["verdict"]
        == "not_supported"
    )
    assert (
        signal_verdict(
            effects,
            risk_off_episodes=3,
            complete_coverage=True,
            source_fresh=False,
        )["reason"]
        == "source_stale"
    )
    assert (
        signal_verdict(
            effects,
            risk_off_episodes=3,
            complete_coverage=True,
            replay_complete=False,
        )["reason"]
        == "replay_incomplete"
    )


def test_signal_rule_controls_required_effects_and_thresholds():
    rule = parse_signal_rule(
        {
            "primary_metric": "max_drawdown",
            "comparison": "risk_off_minus_other",
            "required_baskets": ["benchmarks"],
            "supported_upper_ci_below": -0.01,
            "not_supported_point_at_or_above": 0.0,
        },
        (20,),
    )
    effects = {
        ("benchmarks", 20): {
            "estimate": -0.02,
            "ci_low": -0.04,
            "ci_high": -0.005,
        }
    }

    assert rule.required_effects == frozenset({("benchmarks", 20)})
    assert (
        signal_verdict(
            effects,
            risk_off_episodes=3,
            complete_coverage=True,
            required_effects=rule.required_effects,
            supported_upper_ci_below=rule.supported_upper_ci_below,
            not_supported_point_at_or_above=rule.not_supported_point_at_or_above,
        )["verdict"]
        == "inconclusive"
    )


@pytest.mark.parametrize(
    "override",
    [
        {"primary_metric": "total_return"},
        {"comparison": "risk_on_minus_other"},
        {"required_baskets": ["unknown"]},
        {"supported_upper_ci_below": 0.1},
    ],
)
def test_signal_rule_rejects_unsupported_or_ambiguous_protocol(override):
    value = {
        "primary_metric": "max_drawdown",
        "comparison": "risk_off_minus_other",
        "required_baskets": ["benchmarks", "policy_instruments"],
        "supported_upper_ci_below": 0.0,
        "not_supported_point_at_or_above": 0.0,
        **override,
    }

    with pytest.raises(ValueError, match="signal_rule"):
        parse_signal_rule(value, (20, 60))


def test_latest_month_is_right_censored_without_blocking(long_snapshot):
    point = replay_regimes(long_snapshot)[-1]
    result = build_forward_observations(long_snapshot, [point], (60,))
    assert result["missing"]
    assert all(item["reason"] == "right_censored" for item in result["missing"])
    assert all(item["blocking"] is False for item in result["missing"])


def test_missing_policy_instrument_blocks_signal_verdict(long_snapshot):
    points = replay_regimes(long_snapshot)
    point = points[len(points) // 2]
    benchmark_keys = {item.key for item in long_snapshot.benchmark_specs}
    missing_spec = next(
        item for item in long_snapshot.policy_specs if item.key not in benchmark_keys
    )
    shortened = tuple(
        item
        for item in long_snapshot.candles
        if item.key != missing_spec.key or item.available_at <= point.decision_timestamp
    )
    snapshot = SourceSnapshot(
        long_snapshot.policy_record,
        long_snapshot.benchmark_specs,
        long_snapshot.policy_specs,
        shortened,
    )
    result = build_forward_observations(snapshot, [point], (20,))
    blocking = [item for item in result["missing"] if item["blocking"]]
    assert blocking == [
        {
            "month": point.month,
            "key": missing_spec.key,
            "horizon": 20,
            "reason": "forward_history_short",
            "blocking": True,
        }
    ]


def test_non_required_policy_basket_does_not_block_benchmark_signal(long_snapshot):
    points = replay_regimes(long_snapshot)
    point = points[len(points) // 2]
    benchmark_keys = {item.key for item in long_snapshot.benchmark_specs}
    missing_spec = next(
        item for item in long_snapshot.policy_specs if item.key not in benchmark_keys
    )
    shortened = tuple(
        item
        for item in long_snapshot.candles
        if item.key != missing_spec.key or item.available_at <= point.decision_timestamp
    )
    snapshot = SourceSnapshot(
        long_snapshot.policy_record,
        long_snapshot.benchmark_specs,
        long_snapshot.policy_specs,
        shortened,
    )
    result = build_forward_observations(
        snapshot,
        [point],
        (20,),
        required_effects=frozenset({("benchmarks", 20)}),
    )

    assert all(row["basket"] == "benchmarks" for row in result["rows"])
    assert result["missing"] == []


def test_partial_benchmark_truncation_is_not_treated_as_right_censoring(long_snapshot):
    points = replay_regimes(long_snapshot)
    point = points[len(points) // 2]
    missing_spec = long_snapshot.benchmark_specs[0]
    shortened = tuple(
        item
        for item in long_snapshot.candles
        if item.key != missing_spec.key or item.available_at <= point.decision_timestamp
    )
    snapshot = SourceSnapshot(
        long_snapshot.policy_record,
        long_snapshot.benchmark_specs,
        long_snapshot.policy_specs,
        shortened,
    )
    result = build_forward_observations(snapshot, [point], (20,))
    assert any(
        item["key"] == missing_spec.key
        and item["reason"] == "benchmark_forward_history_short"
        and item["blocking"] is True
        for item in result["missing"]
    )
