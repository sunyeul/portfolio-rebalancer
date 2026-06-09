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
            "block_action": 1,
            "rebalance_sell_review": 2,
            "risk_control_review": 3,
            "review_before_action": 4,
            "reduce_or_pause_dca": 5,
            "increase_dca": 6,
            "hold_observe": 7,
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
        "E": 0.8,
        "샤프": 2.0,
        "IR": 1.0,
        "변동성": 0.12,
        "최대낙폭": -0.05,
        "베타": 1.0,
        "위험기여도": 0.05,
        "현재%": 10.0,
        "RC_Target%": 10.0,
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
    assert result["decision_summary"] == "현재 비중이 목표보다 낮고, 주요 위험·효율·투자 논리 게이트를 통과해 정기매수 보강이 가능합니다."
    assert "ips_fit_high" in result["reason_codes"]


def test_high_ips_fit_negative_gap_classifies_as_reduce_or_pause_dca():
    result = classify_ips_action(
        _row(group="satellite", **{"갭%": -3.0}),
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "reduce_or_pause_dca"
    assert result["execution_type"] == "pause_or_reduce_dca"
    assert "prefer_dca_over_sell" in result["reason_codes"]


def test_medium_core_positive_gap_allows_conditional_increase():
    result = classify_ips_action(
        _row(**{"IPS적합도": 62.0, "IPS등급": "medium"}),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "increase_dca"
    assert result["decision_summary"] == "현재 비중이 목표보다 낮고, 주요 위험·효율·투자 논리 게이트를 통과해 정기매수 보강이 가능합니다."


def test_medium_satellite_positive_gap_requires_thesis_review():
    result = classify_ips_action(
        _row(group="satellite", **{"IPS적합도": 62.0, "IPS등급": "medium"}),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
        decision_context="market_correction",
    )

    assert result["ips_action"] == "review_before_action"
    assert result["execution_type"] == "review_required"
    assert result["decision_summary"] == "하락장 위성 자산은 증액 전 수익률과 변동성을 점검합니다."
    assert "satellite_correction_requires_review" in result["reason_codes"]


def test_low_ips_fit_defaults_to_thesis_review():
    result = classify_ips_action(
        _row(**{"IPS적합도": 38.0, "IPS등급": "low"}),
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "review_before_action"
    assert result["decision_summary"] == "정책 점수, 효율, 위험, 투자 논리 중 하나 이상이 약해 실행 전 점검이 필요합니다."


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

    assert result["ips_action"] == "rebalance_sell_review"
    assert "sell_gate_passed" in result["reason_codes"]
    assert "thesis_broken" in result["reason_codes"]


def test_satellite_over_max_can_consider_sell_without_broken_thesis():
    result = classify_ips_action(
        _row(
            group="satellite",
            risk_over=True,
            **{
                "갭%": -3.0,
                "IPS적합도": 45.0,
                "IPS등급": "low",
                "위험기여도": 0.2,
                "RC_Target%": 1.0,
            },
        ),
        {"core_status": "in_range", "satellite_status": "over_max"},
        _ips_config(),
    )

    assert result["ips_action"] == "rebalance_sell_review"
    assert "sell_gate_passed" in result["reason_codes"]


def test_satellite_over_max_with_small_gap_uses_risk_review_not_sell_review():
    result = classify_ips_action(
        _row(
            group="satellite",
            risk_over=True,
            **{
                "갭%": -1.5,
                "위험기여도": 0.2,
                "RC_Target%": 1.0,
                "변동성": 0.5,
            },
        ),
        {"core_status": "in_range", "satellite_status": "over_max"},
        _ips_config(),
    )

    assert result["ips_action"] == "risk_control_review"
    assert "sell_gate_passed" not in result["reason_codes"]


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
    assert result["decision_summary"] == "하락장에는 목표보다 낮은 코어 비중을 정기매수로 보강합니다."
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
    assert "sharp_drop_buy_caution" in result["reason_codes"]
    assert any("FOMO" in note for note in result["risk_notes"])


def test_satellite_watch_low_efficiency_blocks_increase_for_review():
    result = classify_ips_action(
        _row(
            group="satellite",
            thesis_status="watch",
            E=0.09,
            efficiency_warning=True,
            **{"샤프": -1.0},
        ),
        {"core_status": "over_max", "satellite_status": "under_target"},
        _ips_config(),
    )

    assert result["ips_action"] == "review_before_action"
    assert "efficiency_low" in result["reason_codes"]
    assert "sharpe_negative" in result["reason_codes"]
    assert "buy_gate_blocked" in result["reason_codes"]


def test_underweight_satellite_with_rc_cap_exceeded_goes_to_risk_review():
    result = classify_ips_action(
        _row(
            group="satellite",
            thesis_status="watch",
            E=0.94,
            **{
                "갭%": 6.0,
                "위험기여도": 0.42,
                "RC_Target%": 3.0,
                "변동성": 0.6,
            },
        ),
        {"core_status": "over_max", "satellite_status": "under_target"},
        _ips_config(),
    )

    assert result["ips_action"] == "risk_control_review"
    assert "rc_cap_exceeded" in result["reason_codes"]


def test_underweight_satellite_with_intact_thesis_and_clean_risk_can_increase():
    result = classify_ips_action(
        _row(
            group="satellite",
            thesis_status="intact",
            E=0.86,
            **{
                "갭%": 4.0,
                "샤프": 1.8,
                "IR": 0.7,
                "위험기여도": 0.03,
                "RC_Target%": 8.0,
                "return_total%": 12.0,
            },
        ),
        {"core_status": "over_max", "satellite_status": "under_target"},
        _ips_config(),
    )

    assert result["ips_action"] == "increase_dca"
    assert result["action_family"] == "buy_adjustment"
    assert "allocation_under_target" in result["reason_codes"]
    assert "risk_ok" in result["reason_codes"]


def test_underweight_satellite_watch_with_hot_momentum_requires_review():
    result = classify_ips_action(
        _row(
            group="satellite",
            thesis_status="watch",
            E=0.88,
            **{
                "갭%": 4.0,
                "샤프": 1.6,
                "IR": 0.5,
                "위험기여도": 0.03,
                "RC_Target%": 8.0,
                "return_total%": 85.0,
            },
        ),
        {"core_status": "over_max", "satellite_status": "under_target"},
        _ips_config(),
    )

    assert result["ips_action"] == "review_before_action"
    assert "momentum_too_hot" in result["reason_codes"]
    assert "buy_gate_blocked" in result["reason_codes"]


def test_broken_thesis_positive_gap_blocks_buy_instead_of_reviewing_increase():
    result = classify_ips_action(
        _row(
            group="core",
            thesis_status="broken",
            **{"갭%": 5.0},
        ),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "block_action"
    assert result["execution_type"] == "blocked"
    assert result["blocked_reason"] == "투자 논리가 훼손된 자산은 매수 또는 증액 판단을 차단합니다."


def test_unknown_core_thesis_positive_gap_requires_review_before_increase():
    result = classify_ips_action(
        _row(
            group="core",
            thesis_status="unknown",
            **{"갭%": 4.0},
        ),
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "review_before_action"
    assert "thesis_unknown" in result["reason_codes"]


def test_non_candidate_with_rc_cap_exceeded_still_surfaces_risk_review():
    result = classify_ips_action(
        _row(
            group="core",
            **{
                "갭%": -0.2,
                "실행": False,
                "수치후보": False,
                "히스테리시스제외": True,
                "위험기여도": 0.2,
                "RC_Target%": 5.0,
            },
        ),
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "risk_control_review"
    assert "hysteresis_blocked" in result["reason_codes"]
    assert "rc_cap_exceeded" in result["reason_codes"]


def test_satellite_over_max_without_strong_risk_prefers_dca_reduction_over_sell_review():
    result = classify_ips_action(
        _row(
            group="satellite",
            thesis_status="intact",
            **{
                "갭%": -4.0,
                "위험기여도": 0.04,
                "RC_Target%": 10.0,
                "변동성": 0.12,
                "최대낙폭": -0.05,
                "베타": 1.0,
            },
        ),
        {"core_status": "in_range", "satellite_status": "over_max"},
        _ips_config(),
    )

    assert result["ips_action"] == "reduce_or_pause_dca"
    assert "sell_gate_passed" not in result["reason_codes"]
    assert "prefer_dca_over_sell" in result["reason_codes"]
