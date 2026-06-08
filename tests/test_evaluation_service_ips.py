import pandas as pd

from services.evaluation_service import (
    _action_reason,
    build_ips_target_weights,
    compute_ips_fit_breakdown,
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
    assert "IPS적합도" in result.proposal_df.columns
    assert "IPS등급" in result.proposal_df.columns
    assert "IPS점수_역할" in result.proposal_df.columns
    assert "IPS점수_비중" in result.proposal_df.columns
    assert "IPS점수_논리" in result.proposal_df.columns
    assert "IPS점수_위험" in result.proposal_df.columns
    assert "IPS점수_실행" in result.proposal_df.columns
    assert "IPS점수_E" in result.proposal_df.columns
    assert "IPS점수_데이터" in result.proposal_df.columns
    ufo = result.proposal_df.loc[result.proposal_df["ticker"] == "UFO"].iloc[0]
    assert bool(ufo["risk_over"]) is True
    assert bool(ufo["efficiency_warning"]) is True
    assert ufo["IPS등급"] == "high"
    assert "효율 경고" in ufo["판단사유"]
    assert "ips_action" in result.ips_action_df.columns
    assert "risk_over" in result.ips_action_df.columns
    assert "efficiency_warning" in result.ips_action_df.columns
    assert "IPS적합도" in result.ips_action_df.columns
    assert "execution_type" in result.ips_action_df.columns
    assert "decision_summary" in result.ips_action_df.columns
    assert "risk_notes" in result.ips_action_df.columns


def test_compute_ips_fit_breakdown_scores_range_and_components():
    proposal = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "갭%": [5.0, 5.0],
            "E": [0.8, 0.2],
            "RC_Over%": [0.0, 4.0],
            "group": ["core", "satellite"],
            "dca_enabled": [True, True],
            "thesis_status": ["intact", "watch"],
            "missing_ratio": [0.0, 0.0],
            "observation_count": [120, 120],
            "data_quality_low": [False, False],
            "risk_over": [False, True],
            "수치후보": [True, True],
        }
    )

    scores = compute_ips_fit_breakdown(proposal, {}, e_thresh=0.5)

    assert scores["IPS적합도"].between(0, 100).all()
    assert scores.loc[0, "IPS적합도"] > scores.loc[1, "IPS적합도"]
    assert scores.loc[0, "IPS점수_역할"] > scores.loc[1, "IPS점수_역할"]
    assert scores.loc[0, "IPS점수_논리"] > scores.loc[1, "IPS점수_논리"]
    assert scores.loc[0, "IPS점수_위험"] > scores.loc[1, "IPS점수_위험"]
    assert bool(scores.loc[1, "efficiency_warning"]) is True


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
    assert voo_action["decision_summary"] == "코어 정기매수 증액 우선"
    assert "core_priority_context" in voo_action["reason_codes"]
    assert "efficiency_warning" in voo_action["reason_codes"]
    assert bool(voo_proposal["수치후보"]) is True
    assert bool(voo_proposal["실행"]) is True
    assert voo_proposal["제안조정%"] > 0
    assert voo_proposal["판단사유"] == "코어 정기매수 증액 우선"


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


def test_build_ips_target_weights_keeps_group_internal_current_share():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ", "UFO"],
            "가중치": [0.3, 0.3, 0.4],
            "위험기여도": [0.2, 0.6, 0.5],
            "E": [0.9, 0.2, 0.8],
            "group": ["core", "core", "satellite"],
            "thesis_status": ["intact", "watch", "intact"],
            "missing_ratio": [0.0, 0.0, 0.0],
            "observation_count": [120, 120, 120],
        }
    ).set_index("ticker")
    ips_config = {
        "target_allocation": {
            "core": {"target": 0.8},
            "satellite": {"target": 0.2},
        }
    }

    target = build_ips_target_weights(metrics_df, ips_config)

    assert round(float(target[["VOO", "QQQ"]].sum()), 4) == 0.8
    assert round(float(target["UFO"]), 4) == 0.2
    assert round(float(target["VOO"]), 4) == 0.4
    assert round(float(target["QQQ"]), 4) == 0.4
    assert round(float(target.sum()), 4) == 1.0


def test_build_ips_target_weights_preserves_existing_group_share_without_score_tilt():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ"],
            "가중치": [0.2, 0.6],
            "위험기여도": [0.1, 0.8],
            "E": [0.9, 0.1],
            "group": ["core", "core"],
            "thesis_status": ["intact", "watch"],
        }
    ).set_index("ticker")
    ips_config = {
        "target_allocation": {"core": {"target": 1.0}},
    }

    target = build_ips_target_weights(metrics_df, ips_config)

    assert round(float(target["VOO"]), 4) == 0.25
    assert round(float(target["QQQ"]), 4) == 0.75


def test_build_ips_target_weights_keeps_current_internal_share_when_scores_are_tied():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ"],
            "가중치": [0.2, 0.6],
            "위험기여도": [0.5, 0.5],
            "E": [0.8, 0.8],
            "group": ["core", "core"],
            "thesis_status": ["intact", "intact"],
            "missing_ratio": [0.0, 0.0],
            "observation_count": [120, 120],
        }
    ).set_index("ticker")
    ips_config = {
        "target_allocation": {"core": {"target": 1.0}},
        "target_weighting": {"blend": 1.0},
    }

    target = build_ips_target_weights(metrics_df, ips_config)

    assert round(float(target["VOO"]), 4) == 0.25
    assert round(float(target["QQQ"]), 4) == 0.75


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
    assert ufo_action["ips_action"] in {"review_thesis", "decrease_dca"}
    assert bool(ufo["수치후보"]) is True
    assert ufo["참고조정%"] == -5.0
    if ufo_action["ips_action"] == "review_thesis":
        assert bool(ufo["실행"]) is False
        assert ufo["제안조정%"] == 0.0
        assert ufo["판단사유"] == "투자 논리 점검"
    else:
        assert bool(ufo["실행"]) is True
        assert ufo["제안조정%"] < 0


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
        _action_reason({"실행": True, "IPS등급": "high", "risk_over": True, "efficiency_warning": True, "갭%": 2})
        == "IPS 적합 · 위험 초과 · 효율 경고 · 목표 대비 부족"
    )
    assert (
        _action_reason({"실행": True, "IPS등급": "medium", "risk_over": True, "efficiency_warning": False, "갭%": 2})
        == "IPS 조건부 점검 · 위험 초과 · 목표 대비 부족"
    )
    assert (
        _action_reason({"실행": True, "IPS등급": "low", "risk_over": False, "efficiency_warning": True, "갭%": 2})
        == "IPS 부적합 · 효율 경고 · 목표 대비 부족"
    )
    assert (
        _action_reason({"실행": True, "IPS등급": "high", "risk_over": False, "efficiency_warning": False, "갭%": 2})
        == "IPS 적합 · 목표 대비 부족"
    )
    assert (
        _action_reason({"실행": True, "IPS등급": "high", "risk_over": False, "efficiency_warning": False, "갭%": -2})
        == "IPS 적합 · 목표 대비 초과"
    )
