import pandas as pd

from utils.ips import (
    classify_ips_action,
    compute_group_summary,
    compute_ips_allocation_status,
)


def _ips_config():
    return {
        "target_allocation": {
            "core": {"min": 0.70, "target": 0.80, "max": 0.90},
            "satellite": {"min": 0.10, "target": 0.20, "max": 0.30},
        },
        "action_priority": {
            "increase_dca": 1,
            "decrease_dca": 2,
            "review_thesis": 3,
            "exceptional_buy_review": 4,
            "consider_rebalance_sell": 5,
            "hold_observe": 6,
            "block_action": 7,
        },
        "rules": {"prefer_dca_over_sell": True},
    }


def _row(**overrides):
    base = {
        "risk_over": False,
        "efficiency_warning": False,
        "IPS적합도": 82.0,
        "IPS등급": "high",
        "갭%": 3.0,
        "실행": True,
        "수치후보": True,
        "group": "core",
        "dca_enabled": True,
        "thesis_status": "intact",
    }
    base.update(overrides)
    return base


def test_compute_group_summary_and_allocation_status():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "가중치": [0.8, 0.2],
            "위험기여도": [0.6, 0.4],
            "E": [0.7, 0.5],
            "group": ["core", "satellite"],
        }
    ).set_index("ticker")

    summary = compute_group_summary(metrics_df, _ips_config())
    status = compute_ips_allocation_status(summary, _ips_config())

    assert summary.loc[summary["group"] == "core", "weight"].sum() == 0.8
    assert summary.loc[summary["group"] == "satellite", "weight"].sum() == 0.2
    assert status["core_status"] == "in_range"
    assert status["satellite_status"] == "in_range"


def test_high_ips_fit_positive_core_gap_classifies_as_increase_dca():
    result = classify_ips_action(
        _row(),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "increase_dca"
    assert result["execution_type"] == "dca_adjustment"
    assert result["decision_summary"] == "IPS 적합 정기매수 증액 후보"
    assert "ips_fit_high" in result["reason_codes"]


def test_high_ips_fit_negative_gap_classifies_as_decrease_dca():
    result = classify_ips_action(
        _row(group="satellite", **{"갭%": -3.0, "risk_over": True}),
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "decrease_dca"
    assert result["execution_type"] == "dca_adjustment"
    assert "prefer_dca_over_sell" in result["reason_codes"]


def test_medium_core_positive_gap_allows_conditional_increase():
    result = classify_ips_action(
        _row(**{"IPS적합도": 62.0, "IPS등급": "medium"}),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "increase_dca"
    assert result["decision_summary"] == "IPS 조건부 코어 정기매수 증액 후보"


def test_medium_satellite_positive_gap_requires_thesis_review():
    result = classify_ips_action(
        _row(group="satellite", **{"IPS적합도": 62.0, "IPS등급": "medium"}),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
        decision_context="market_correction",
    )

    assert result["ips_action"] == "review_thesis"
    assert result["execution_type"] == "review_required"
    assert result["decision_summary"] == "하락장 위성 추가매수 전 점검"
    assert "satellite_correction_requires_review" in result["reason_codes"]


def test_low_ips_fit_defaults_to_thesis_review():
    result = classify_ips_action(
        _row(**{"IPS적합도": 38.0, "IPS등급": "low"}),
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "review_thesis"
    assert result["blocked_reason"] == "IPS 적합도가 낮아 실행보다 투자 논리 점검이 우선입니다."


def test_low_data_quality_blocks_otherwise_executable_action():
    result = classify_ips_action(
        _row(data_quality_low=True),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "block_action"
    assert result["execution_type"] == "blocked"


def test_broken_thesis_negative_gap_can_consider_sell():
    result = classify_ips_action(
        _row(
            group="satellite",
            dca_enabled=False,
            thesis_status="broken",
            risk_over=True,
            efficiency_warning=True,
            **{"갭%": -3.0, "IPS적합도": 35.0, "IPS등급": "low"},
        ),
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "consider_rebalance_sell"
    assert "sell_gate_passed" in result["reason_codes"]
    assert "thesis_broken" in result["reason_codes"]


def test_satellite_over_max_can_consider_sell_without_broken_thesis():
    result = classify_ips_action(
        _row(group="satellite", risk_over=True, **{"갭%": -3.0, "IPS적합도": 45.0, "IPS등급": "low"}),
        {"core_status": "in_range", "satellite_status": "over_max"},
        _ips_config(),
    )

    assert result["ips_action"] == "consider_rebalance_sell"
    assert "sell_gate_passed" in result["reason_codes"]


def test_market_correction_allows_core_reinforcement_even_with_efficiency_warning():
    result = classify_ips_action(
        _row(
            efficiency_warning=True,
            **{"IPS적합도": 62.0, "IPS등급": "medium"},
        ),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
        decision_context="market_correction",
    )

    assert result["ips_action"] == "increase_dca"
    assert result["decision_summary"] == "하락장 코어 정기매수 증액 후보"
    assert "efficiency_warning" in result["reason_codes"]
    assert "correction_core_reinforcement" in result["reason_codes"]


def test_sharp_drop_review_adds_buy_caution_without_immediate_buy_action():
    result = classify_ips_action(
        _row(),
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
        decision_context="sharp_drop_review",
    )

    assert result["ips_action"] == "increase_dca"
    assert result["execution_type"] == "dca_adjustment"
    assert result["ips_action"] != "exceptional_buy_review"
    assert "sharp_drop_buy_caution" in result["reason_codes"]
    assert any("FOMO" in note for note in result["risk_notes"])
