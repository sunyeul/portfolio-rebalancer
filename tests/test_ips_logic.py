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
    }


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


def test_risk_ok_efficiency_good_positive_gap_classifies_as_increase_dca():
    result = classify_ips_action(
        {
            "risk_over": False,
            "efficiency_good": True,
            "갭%": 3.0,
            "실행": True,
            "group": "core",
            "dca_enabled": True,
            "thesis_status": "intact",
        },
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "increase_dca"
    assert result["execution_type"] == "dca_adjustment"
    assert result["decision_context"] == "regular_review"
    assert result["decision_summary"] == "정기매수 배분 증액"


def test_execution_type_mapping_for_core_actions():
    cases = [
        (
            {
                "risk_over": False,
                "efficiency_good": True,
                "갭%": 0.0,
                "실행": False,
                "group": "core",
            },
            "hold_observe",
            "observe",
        ),
        (
            {
                "risk_over": False,
                "efficiency_good": False,
                "갭%": 3.0,
                "실행": True,
                "group": "core",
            },
            "review_thesis",
            "review_required",
        ),
        (
            {
                "risk_over": True,
                "efficiency_good": False,
                "갭%": -3.0,
                "실행": True,
                "group": "satellite",
                "dca_enabled": False,
                "thesis_status": "broken",
            },
            "consider_rebalance_sell",
            "exceptional_sell_review",
        ),
        (
            {
                "risk_over": False,
                "efficiency_good": True,
                "갭%": 3.0,
                "실행": True,
                "group": "core",
                "data_quality_low": True,
            },
            "block_action",
            "blocked",
        ),
    ]

    for row, expected_action, expected_execution_type in cases:
        result = classify_ips_action(
            row,
            {"core_status": "in_range", "satellite_status": "in_range"},
            _ips_config(),
        )
        assert result["ips_action"] == expected_action
        assert result["execution_type"] == expected_execution_type


def test_market_correction_keeps_core_increase_with_core_priority_summary():
    result = classify_ips_action(
        {
            "risk_over": False,
            "efficiency_good": True,
            "갭%": 3.0,
            "실행": True,
            "group": "core",
            "dca_enabled": True,
            "thesis_status": "intact",
        },
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
        decision_context="market_correction",
    )

    assert result["ips_action"] == "increase_dca"
    assert result["execution_type"] == "dca_adjustment"
    assert result["decision_summary"] == "코어 정기매수 증액 우선"
    assert "core_priority_context" in result["reason_codes"]
    assert any("코어" in reason for reason in result["decision_reasons"])


def test_market_correction_downgrades_satellite_increase_when_core_is_under_target():
    result = classify_ips_action(
        {
            "risk_over": False,
            "efficiency_good": True,
            "갭%": 3.0,
            "실행": True,
            "group": "satellite",
            "dca_enabled": True,
            "thesis_status": "intact",
        },
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
        decision_context="market_correction",
    )

    assert result["ips_action"] == "review_thesis"
    assert result["execution_type"] == "review_required"
    assert result["decision_summary"] == "하락장 위성 증액 전 점검"
    assert "satellite_downgraded_for_core_priority" in result["reason_codes"]


def test_sharp_drop_review_adds_buy_caution_without_immediate_buy_action():
    result = classify_ips_action(
        {
            "risk_over": False,
            "efficiency_good": True,
            "갭%": 3.0,
            "실행": True,
            "group": "core",
            "dca_enabled": True,
            "thesis_status": "intact",
        },
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
        decision_context="sharp_drop_review",
    )

    assert result["ips_action"] == "increase_dca"
    assert result["execution_type"] == "dca_adjustment"
    assert result["ips_action"] != "exceptional_buy_review"
    assert "sharp_drop_buy_caution" in result["reason_codes"]
    assert any("FOMO" in note for note in result["risk_notes"])


def test_risk_over_efficiency_low_broken_thesis_can_consider_sell():
    result = classify_ips_action(
        {
            "risk_over": True,
            "efficiency_good": False,
            "갭%": -3.0,
            "실행": True,
            "group": "satellite",
            "dca_enabled": False,
            "thesis_status": "broken",
        },
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "consider_rebalance_sell"


def test_risk_over_efficiency_low_intact_thesis_does_not_consider_sell():
    result = classify_ips_action(
        {
            "risk_over": True,
            "efficiency_good": False,
            "갭%": -3.0,
            "실행": True,
            "group": "satellite",
            "dca_enabled": True,
            "thesis_status": "intact",
        },
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] in {"decrease_dca", "review_thesis"}
    assert result["ips_action"] != "consider_rebalance_sell"


def test_prefer_dca_over_sell_blocks_satellite_over_max_sell_until_thesis_breaks():
    result = classify_ips_action(
        {
            "risk_over": True,
            "efficiency_good": False,
            "갭%": -3.0,
            "실행": True,
            "group": "satellite",
            "dca_enabled": True,
            "thesis_status": "intact",
        },
        {"core_status": "in_range", "satellite_status": "over_max"},
        _ips_config(),
    )

    assert result["ips_action"] == "decrease_dca"
    assert "prefer_dca_over_sell" in result["reason_codes"]


def test_low_data_quality_blocks_otherwise_executable_action():
    result = classify_ips_action(
        {
            "risk_over": False,
            "efficiency_good": True,
            "갭%": 3.0,
            "실행": True,
            "group": "core",
            "dca_enabled": True,
            "thesis_status": "intact",
            "data_quality_low": True,
        },
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "block_action"
