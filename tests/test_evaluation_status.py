from datetime import date

from core.evaluation import EvaluationOutput, EvaluationPeriod, EvaluationUnit
from services.evaluation_status import classify_evaluation_status


def _unit(**overrides):
    data = {
        "level": "asset",
        "name": "VOO",
        "parent_layer": "core",
        "target_weight": 0.5,
        "allowed_mdd": -0.3,
        "allowed_volatility": 0.4,
        "max_weight": 0.7,
        "min_efficiency": 0.0,
        "evaluation_period": EvaluationPeriod(
            label="3M",
            start_date=date(2026, 3, 23),
            end_date=date(2026, 6, 23),
        ),
    }
    data.update(overrides)
    return EvaluationUnit(**data)


def _output(**overrides):
    data = {
        "current_weight": 0.5,
        "weight_gap": 0.0,
        "period_return": 0.05,
        "cagr": 0.1,
        "mdd": -0.1,
        "volatility": 0.2,
        "risk_contribution": 0.1,
        "cagr_mdd_ratio": 1.0,
        "thesis_status": "valid",
        "burden": "low",
        "status": "OK",
    }
    data.update(overrides)
    return EvaluationOutput(**data)


def test_classifies_ok():
    status, reasons = classify_evaluation_status(_unit(), _output())

    assert status == "OK"
    assert reasons == []


def test_classifies_watch_for_soft_warning():
    status, reasons = classify_evaluation_status(_unit(), _output(thesis_status="watch"))

    assert status == "Watch"
    assert "thesis_watch" in reasons


def test_classifies_review_for_hard_breach():
    status, reasons = classify_evaluation_status(_unit(), _output(mdd=-0.5))

    assert status == "Review"
    assert "mdd_exceeded" in reasons


def test_classifies_action_for_broken_thesis_and_limit_breach_without_trade_signal():
    status, reasons = classify_evaluation_status(
        _unit(),
        _output(thesis_status="broken", current_weight=0.8),
    )

    assert status == "Action"
    assert "thesis_broken" in reasons
    assert all("buy" not in reason and "sell" not in reason for reason in reasons)
