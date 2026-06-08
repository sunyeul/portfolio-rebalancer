import pandas as pd

from services.evaluation_service import (
    _action_reason,
    build_ips_target_weights,
    run_evaluation,
)


def test_run_evaluation_returns_ips_outputs_and_uses_ips_signals():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "가중치": [0.4, 0.6],
            "위험기여도": [0.3, 0.8],
            "E": [0.8, 0.2],
            "return_total": [0.1, -0.1],
            "group": ["core", "satellite"],
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
    assert "execution_type" in result.ips_action_df.columns
    assert "decision_summary" in result.ips_action_df.columns
    assert "risk_notes" in result.ips_action_df.columns


def test_run_evaluation_defaults_to_regular_review_context():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO"],
            "가중치": [1.0],
            "위험기여도": [0.5],
            "E": [0.8],
            "return_total": [0.1],
            "group": ["core"],
            "dca_enabled": [True],
            "thesis_status": ["intact"],
        }
    ).set_index("ticker")

    result = run_evaluation(
        metrics_df,
        None,
        rc_over_thresh_pct=1.0,
        e_thresh=0.5,
    )

    action = result.ips_action_df.iloc[0]
    assert action["decision_context"] == "regular_review"
    assert result.ips_config_snapshot["decision_context"] == "regular_review"


def test_market_correction_context_downgrades_satellite_buy_and_blocks_final_trade():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO", "CASH"],
            "가중치": [0.7, 0.1, 0.2],
            "위험기여도": [0.3, 0.1, 0.0],
            "E": [0.8, 0.8, 0.0],
            "return_total": [0.1, 0.2, 0.0],
            "group": ["core", "satellite", "cash"],
            "dca_enabled": [True, True, False],
            "thesis_status": ["intact", "intact", "intact"],
        }
    ).set_index("ticker")

    result = run_evaluation(
        metrics_df,
        None,
        rc_over_thresh_pct=100.0,
        e_thresh=0.5,
        decision_context="market_correction",
    )

    ufo_action = result.ips_action_df.loc[result.ips_action_df["ticker"] == "UFO"].iloc[0]
    ufo_proposal = result.proposal_df.loc[result.proposal_df["ticker"] == "UFO"].iloc[0]
    assert ufo_action["ips_action"] == "review_thesis"
    assert ufo_action["execution_type"] == "review_required"
    assert ufo_action["decision_summary"] == "하락장 위성 증액 전 점검"
    assert bool(ufo_proposal["수치후보"]) is True
    assert bool(ufo_proposal["실행"]) is False
    assert ufo_proposal["제안조정%"] == 0.0
    assert ufo_proposal["판단사유"] == "투자 논리 점검"


def test_market_correction_context_increases_underweight_core_even_with_low_efficiency():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "가중치": [0.6, 0.4],
            "위험기여도": [0.2, 0.8],
            "E": [0.2, 0.9],
            "return_total": [-0.1, 0.2],
            "group": ["core", "satellite"],
            "dca_enabled": [True, True],
            "thesis_status": ["intact", "intact"],
            "missing_ratio": [0.0, 0.0],
            "observation_count": [120, 120],
        }
    ).set_index("ticker")

    result = run_evaluation(
        metrics_df,
        None,
        rc_over_thresh_pct=100.0,
        e_thresh=0.5,
        decision_context="market_correction",
    )

    voo_action = result.ips_action_df.loc[result.ips_action_df["ticker"] == "VOO"].iloc[0]
    voo_proposal = result.proposal_df.loc[result.proposal_df["ticker"] == "VOO"].iloc[0]
    assert voo_action["ips_action"] == "increase_dca"
    assert voo_action["decision_summary"] == "하락장 코어 정기매수 증액 후보"
    assert "correction_core_reinforcement" in voo_action["reason_codes"]
    assert bool(voo_proposal["수치후보"]) is True
    assert bool(voo_proposal["실행"]) is True
    assert voo_proposal["제안조정%"] > 0
    assert voo_proposal["판단사유"] == "하락장 코어 정기매수 증액 후보"


def test_build_ips_target_weights_keeps_cash_and_allocates_remaining_by_ips_targets():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ", "UFO", "CASH"],
            "가중치": [0.3, 0.3, 0.3, 0.1],
            "group": ["core", "core", "satellite", "cash"],
        }
    ).set_index("ticker")
    ips_config = {
        "target_allocation": {
            "core": {"target": 0.8},
            "satellite": {"target": 0.2},
        }
    }

    target = build_ips_target_weights(metrics_df, ips_config)

    assert round(float(target[["VOO", "QQQ"]].sum()), 4) == 0.72
    assert round(float(target["UFO"]), 4) == 0.18
    assert round(float(target["CASH"]), 4) == 0.1
    assert round(float(target.sum()), 4) == 1.0


def test_run_evaluation_uses_auto_targets_when_explicit_targets_are_absent():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "가중치": [0.6, 0.4],
            "위험기여도": [0.5, 0.5],
            "E": [0.8, 0.2],
            "return_total": [0.1, -0.1],
            "group": ["core", "satellite"],
            "dca_enabled": [True, True],
            "thesis_status": ["intact", "intact"],
        }
    ).set_index("ticker")

    result = run_evaluation(
        metrics_df,
        None,
        rc_over_thresh_pct=1.0,
        e_thresh=0.5,
    )

    voo = result.proposal_df.loc[result.proposal_df["ticker"] == "VOO"].iloc[0]
    ufo = result.proposal_df.loc[result.proposal_df["ticker"] == "UFO"].iloc[0]
    assert voo["목표%"] == 80.0
    assert ufo["목표%"] == 20.0
    assert voo["갭%"] == 20.0
    assert ufo["갭%"] == -20.0
    assert bool(voo["실행"]) is True
    assert bool(ufo["실행"]) is True
    assert voo["제안조정%"] == 20.0
    assert ufo["제안조정%"] == -20.0


def test_run_evaluation_zeros_final_trade_when_ips_requires_thesis_review():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "가중치": [0.75, 0.25],
            "위험기여도": [0.4, 0.6],
            "E": [0.8, 0.2],
            "return_total": [0.1, -0.1],
            "group": ["core", "satellite"],
            "dca_enabled": [True, False],
            "thesis_status": ["intact", "intact"],
        }
    ).set_index("ticker")

    result = run_evaluation(
        metrics_df,
        None,
        rc_over_thresh_pct=1.0,
        e_thresh=0.5,
    )

    ufo = result.proposal_df.loc[result.proposal_df["ticker"] == "UFO"].iloc[0]
    ufo_action = result.ips_action_df.loc[result.ips_action_df["ticker"] == "UFO"].iloc[0]
    assert ufo_action["ips_action"] == "review_thesis"
    assert bool(ufo["수치후보"]) is True
    assert ufo["참고조정%"] == -5.0
    assert bool(ufo["실행"]) is False
    assert ufo["제안조정%"] == 0.0
    assert ufo["판단사유"] == "투자 논리 점검"


def test_run_evaluation_blocks_low_data_quality_final_trade():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ"],
            "가중치": [0.4, 0.6],
            "위험기여도": [0.2, 0.8],
            "E": [0.8, 0.8],
            "return_total": [0.1, 0.1],
            "group": ["core", "core"],
            "dca_enabled": [True, True],
            "thesis_status": ["intact", "intact"],
            "missing_ratio": [0.3, 0.0],
            "observation_count": [30, 120],
        }
    ).set_index("ticker")

    result = run_evaluation(
        metrics_df,
        {"VOO": 0.6, "QQQ": 0.4},
        rc_over_thresh_pct=1.0,
        e_thresh=0.5,
    )

    voo = result.proposal_df.loc[result.proposal_df["ticker"] == "VOO"].iloc[0]
    voo_action = result.ips_action_df.loc[result.ips_action_df["ticker"] == "VOO"].iloc[0]
    assert voo_action["ips_action"] == "block_action"
    assert bool(voo["수치후보"]) is True
    assert bool(voo["실행"]) is False
    assert voo["제안조정%"] == 0.0
    assert voo["판단사유"] == "행동 보류"


def test_run_evaluation_does_not_drop_trade_flags_on_ticker_index_alignment():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["GLD", "QQQ", "MU"],
            "가중치": [0.2, 0.2, 0.6],
            "위험기여도": [0.1, 0.1, 0.8],
            "E": [0.8, 0.8, 0.8],
            "return_total": [0.1, 0.1, 0.1],
            "group": ["core", "core", "satellite"],
            "dca_enabled": [True, True, True],
            "thesis_status": ["intact", "intact", "intact"],
        }
    ).set_index("ticker")

    result = run_evaluation(
        metrics_df,
        None,
        rc_over_thresh_pct=1.0,
        e_thresh=0.5,
    )

    assert result.proposal_df["실행"].notna().all()
    assert result.proposal_df["히스테리시스제외"].notna().all()
    assert result.proposal_df["최소거래미만"].notna().all()
    assert result.proposal_df["제안조정%"].abs().sum() > 0


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
