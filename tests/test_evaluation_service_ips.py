import pandas as pd

from services.evaluation_service import _action_reason, run_evaluation


def test_run_evaluation_returns_ips_outputs_and_uses_ips_signals():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "가중치": [0.4, 0.6],
            "위험기여도": [0.3, 0.8],
            "E": [0.8, 0.2],
            "return_total": [0.1, -0.1],
            "group": ["core", "satellite_space"],
            "dca_enabled": [True, True],
            "thesis_status": ["intact", "intact"],
        }
    ).set_index("ticker")

    result = run_evaluation(
        metrics_df,
        {"VOO": 0.5, "UFO": 0.5},
        rc_over_thresh_pct=1.0,
        e_thresh=0.5,
    )

    assert not result.ips_action_df.empty
    assert not result.group_summary_df.empty
    assert "DCA강도점수" not in result.proposal_df.columns
    assert "E" in result.proposal_df.columns
    assert "E′" not in result.proposal_df.columns
    assert "RC_Gap%" in result.proposal_df.columns
    assert "제안조정%" in result.proposal_df.columns
    assert "판단사유" in result.proposal_df.columns
    legacy_column = "\uc0ac\ubd84\uba74"
    assert legacy_column not in result.proposal_df.columns
    ufo = result.proposal_df.loc[result.proposal_df["ticker"] == "UFO"].iloc[0]
    assert bool(ufo["risk_over"]) is True
    assert bool(ufo["efficiency_good"]) is False
    assert ufo["판단사유"] == "위험 초과 및 효율 미달"
    assert "ips_action" in result.ips_action_df.columns
    assert "risk_over" in result.ips_action_df.columns
    assert "efficiency_good" in result.ips_action_df.columns


def test_action_reason_labels_filter_and_execution_causes():
    assert (
        _action_reason({"실행": False, "히스테리시스제외": True, "최소거래미만": True})
        == "히스테리시스 범위 및 최소 거래 미만"
    )
    assert (
        _action_reason({"실행": False, "히스테리시스제외": True, "최소거래미만": False})
        == "히스테리시스 범위"
    )
    assert (
        _action_reason({"실행": False, "히스테리시스제외": False, "최소거래미만": True})
        == "최소 거래 미만"
    )
    assert (
        _action_reason({"실행": True, "risk_over": True, "efficiency_good": False, "갭%": 2})
        == "위험 초과 및 효율 미달"
    )
    assert (
        _action_reason({"실행": True, "risk_over": True, "efficiency_good": True, "갭%": 2})
        == "위험 초과"
    )
    assert (
        _action_reason({"실행": True, "risk_over": False, "efficiency_good": False, "갭%": 2})
        == "효율 미달"
    )
    assert (
        _action_reason({"실행": True, "risk_over": False, "efficiency_good": True, "갭%": 2})
        == "목표 대비 부족"
    )
    assert (
        _action_reason({"실행": True, "risk_over": False, "efficiency_good": True, "갭%": -2})
        == "목표 대비 초과"
    )
