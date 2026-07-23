import sqlite3
from copy import deepcopy

import pytest

from storage.database import initialize_database
from storage.policy_store import DEFAULT_POLICY, get_active_policy, policy_hash


def _risk_review():
    return {
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
    }


def _core_instrument():
    return {
        "market_country": "US",
        "symbol": "SPY",
        "layer": "core",
        "minimum": 1,
        "target": 1,
        "maximum": 1,
    }


def _allocation_review():
    return {
        "strategy": "us_kr_three_regime_v1",
        "cooldown_days": 30,
        "minimum_history_points": 200,
        "max_data_age_days": 7,
        "max_gap_days": 7,
        "drawdown_review": -0.15,
        "volatility_review": 0.30,
        "risk_on_trend": 0.50,
        "risk_off_trend": -0.50,
        "risk_on_max_risk_weight": 0.30,
        "risk_off_risk_weight": 0.50,
        "benchmarks": [
            {
                "key": "US/SPY",
                "source_kind": "stock",
                "market_country": "US",
                "symbol": "SPY",
                "weight": 0.30,
            },
            {
                "key": "US/QQQ",
                "source_kind": "stock",
                "market_country": "US",
                "symbol": "QQQ",
                "weight": 0.30,
            },
            {
                "key": "KR/KOSPI",
                "source_kind": "market_indicator",
                "market_country": "KR",
                "symbol": "KOSPI",
                "weight": 0.25,
            },
            {
                "key": "KR/KOSDAQ",
                "source_kind": "market_indicator",
                "market_country": "KR",
                "symbol": "KOSDAQ",
                "weight": 0.15,
            },
        ],
        "regimes": {
            "risk_on": {
                "cash_target": 0.03,
                "layers": {"core": 0.52, "satellite": 0.44, "experiment": 0.04},
            },
            "neutral": {
                "cash_target": 0.05,
                "layers": {"core": 0.60, "satellite": 0.38, "experiment": 0.02},
            },
            "risk_off": {
                "cash_target": 0.10,
                "layers": {"core": 0.70, "satellite": 0.29, "experiment": 0.01},
            },
        },
    }


def _dynamic_policy():
    return {
        "cash_reserve": {"minimum": 0.03, "target": 0.05, "maximum": 0.10},
        "performance": {
            "annual_return_target": 0.1,
            "measurement": "ytd_twr",
            "minimum_history_days": 365,
        },
        "risk_review": _risk_review(),
        "cadence": {"observation": "weekly", "inspection": "monthly"},
        "layers": {
            "core": {"minimum": 0.50, "target": 0.60, "maximum": 0.70},
            "satellite": {"minimum": 0.28, "target": 0.38, "maximum": 0.48},
            "experiment": {"minimum": 0.00, "target": 0.02, "maximum": 0.04},
        },
        "instruments": [
            {
                "market_country": "US",
                "symbol": "SPY",
                "layer": "core",
                "minimum": 0.50,
                "target": 0.60,
                "maximum": 0.70,
            },
            {
                "market_country": "US",
                "symbol": "QQQ",
                "layer": "satellite",
                "minimum": 0.28,
                "target": 0.38,
                "maximum": 0.48,
            },
            {
                "market_country": "US",
                "symbol": "GLD",
                "layer": "experiment",
                "minimum": 0.00,
                "target": 0.02,
                "maximum": 0.04,
            },
        ],
        "allocation_review": _allocation_review(),
    }


def test_policy_validation_preserves_allocation_review():
    from services.policy_validation import validate_policy

    policy = _dynamic_policy()

    normalized = validate_policy(
        policy,
        [("US", "SPY"), ("US", "QQQ"), ("US", "GLD")],
    )

    assert normalized["allocation_review"] == _allocation_review()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["benchmarks"].append(
                deepcopy(value["benchmarks"][0])
            ),
            "duplicate",
        ),
        (
            lambda value: value["benchmarks"][0].update(weight=0.10),
            "weights must sum to 1",
        ),
        (
            lambda value: value["regimes"].pop("risk_off"),
            "exactly risk_on, neutral, risk_off",
        ),
        (
            lambda value: value["regimes"]["neutral"]["layers"].update(core=0.61),
            "layer targets must sum to 1",
        ),
        (
            lambda value: value["regimes"]["risk_on"].update(cash_target=0.02),
            "cash target must be within cash_reserve range",
        ),
    ],
)
def test_policy_validation_rejects_invalid_allocation_review(mutator, message):
    from services.policy_validation import PolicyValidationError, validate_policy

    policy = _dynamic_policy()
    mutator(policy["allocation_review"])

    with pytest.raises(PolicyValidationError, match=message):
        validate_policy(
            policy,
            [("US", "SPY"), ("US", "QQQ"), ("US", "GLD")],
        )


def test_default_policy_is_seeded_once_and_is_replayable(monkeypatch, tmp_path):
    path = tmp_path / "policy.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))

    initialize_database()
    first = get_active_policy()
    initialize_database()
    second = get_active_policy()

    assert first is not None
    assert second == first
    assert first["version"] == 1
    assert first["policy"] == DEFAULT_POLICY
    assert first["policy_hash"] == policy_hash(DEFAULT_POLICY)
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM ips_policy_versions").fetchone()[0] == 1
        )


def test_policy_validation_rejects_unseen_identity_and_bad_layer_sum():
    from services.policy_validation import PolicyValidationError, validate_policy

    policy = {
        "cash_reserve": {"minimum": 0.1, "target": 0.15, "maximum": 0.2},
        "performance": {
            "annual_return_target": 0.1,
            "measurement": "trailing_12_month_twr",
            "minimum_history_days": 365,
        },
        "risk_review": _risk_review(),
        "cadence": {"observation": "weekly", "inspection": "monthly"},
        "layers": {
            "core": {"minimum": 0, "target": 0.7, "maximum": 0.9},
            "satellite": {"minimum": 0, "target": 0.2, "maximum": 0.3},
            "experiment": {"minimum": 0, "target": 0.2, "maximum": 0.2},
        },
        "instruments": [
            {
                "market_country": "US",
                "symbol": "NOPE",
                "layer": "core",
                "minimum": 0,
                "target": 0.7,
                "maximum": 0.7,
            }
        ],
    }
    with pytest.raises(PolicyValidationError) as error:
        validate_policy(policy, [("US", "SPY")])
    assert "layers target values must sum to 1" in str(error.value)
    assert "instrument not observed by Toss" in str(error.value)


def test_policy_validation_rejects_any_instrument_without_toss_observation():
    from services.policy_validation import PolicyValidationError, validate_policy

    policy = {
        "cash_reserve": {"minimum": 0.1, "target": 0.15, "maximum": 0.2},
        "performance": {
            "annual_return_target": 0.1,
            "measurement": "trailing_12_month_twr",
            "minimum_history_days": 365,
        },
        "risk_review": _risk_review(),
        "cadence": {"observation": "weekly", "inspection": "monthly"},
        "layers": {
            "core": {"minimum": 1, "target": 1, "maximum": 1},
            "satellite": {"minimum": 0, "target": 0, "maximum": 0},
            "experiment": {"minimum": 0, "target": 0, "maximum": 0},
        },
        "instruments": [
            {
                "market_country": "US",
                "symbol": "SPY",
                "layer": "core",
                "minimum": 1,
                "target": 1,
                "maximum": 1,
            }
        ],
    }
    with pytest.raises(PolicyValidationError, match="not observed by Toss"):
        validate_policy(policy, [])


def test_policy_validation_normalizes_risk_review():
    from services.policy_validation import validate_policy

    policy = {
        "cash_reserve": {"minimum": 0.1, "target": 0.15, "maximum": 0.2},
        "performance": {
            "annual_return_target": 0.1,
            "measurement": "ytd_twr",
            "minimum_history_days": 365,
        },
        "risk_review": _risk_review(),
        "cadence": {"observation": "weekly", "inspection": "monthly"},
        "layers": {
            "core": {"minimum": 1, "target": 1, "maximum": 1},
            "satellite": {"minimum": 0, "target": 0, "maximum": 0},
            "experiment": {"minimum": 0, "target": 0, "maximum": 0},
        },
        "instruments": [_core_instrument()],
    }

    normalized = validate_policy(policy, [("US", "SPY")])

    assert normalized["risk_review"] == _risk_review()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda risk: risk.update(lookback_sessions=0),
        lambda risk: risk.update(minimum_history_points=253),
        lambda risk: risk.update(max_data_age_days=True),
        lambda risk: risk.update(max_gap_days=-1),
        lambda risk: risk.update(account_drawdown_review=0),
        lambda risk: risk.update(account_drawdown_review=-1),
        lambda risk: risk.update(
            instrument_drawdown_review={"core": -0.25, "satellite": -0.2}
        ),
        lambda risk: risk.update(
            instrument_drawdown_review={
                "core": -0.25,
                "satellite": -0.2,
                "experiment": -0.15,
                "extra": -0.1,
            }
        ),
    ],
)
def test_policy_validation_rejects_invalid_risk_review(mutator):
    from services.policy_validation import PolicyValidationError, validate_policy

    risk = _risk_review()
    risk["instrument_drawdown_review"] = dict(risk["instrument_drawdown_review"])
    mutator(risk)
    policy = {
        "cash_reserve": {"minimum": 0.1, "target": 0.15, "maximum": 0.2},
        "performance": {
            "annual_return_target": 0.1,
            "measurement": "ytd_twr",
            "minimum_history_days": 365,
        },
        "risk_review": risk,
        "cadence": {"observation": "weekly", "inspection": "monthly"},
        "layers": {
            "core": {"minimum": 1, "target": 1, "maximum": 1},
            "satellite": {"minimum": 0, "target": 0, "maximum": 0},
            "experiment": {"minimum": 0, "target": 0, "maximum": 0},
        },
        "instruments": [_core_instrument()],
    }
    with pytest.raises(PolicyValidationError, match="risk_review"):
        validate_policy(policy, [("US", "SPY")])


def test_policy_validation_requires_risk_review():
    from services.policy_validation import PolicyValidationError, validate_policy

    policy = dict(DEFAULT_POLICY)
    policy.pop("risk_review", None)
    policy["instruments"] = [_core_instrument()]

    with pytest.raises(PolicyValidationError, match="risk_review"):
        validate_policy(policy, [("US", "SPY")])
