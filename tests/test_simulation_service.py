import pandas as pd

from services.simulation_service import (
    _prepare_metrics,
    _scenario_target_weights,
    run_counterfactual_simulation,
    run_ips_backtest,
)
from utils.ips_config import load_ips_config
from utils.metrics import annualize_cov


def _metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ", "UFO"],
            "가중치": [0.55, 0.15, 0.30],
            "위험기여도": [0.35, 0.15, 0.50],
            "E": [0.8, 0.7, 0.65],
            "return_total": [0.1, 0.08, 0.2],
            "group": ["core", "core", "satellite"],
            "dca_enabled": [True, True, True],
            "thesis_status": ["intact", "intact", "watch"],
            "missing_ratio": [0.0, 0.0, 0.35],
            "observation_count": [120, 120, 40],
        }
    ).set_index("ticker")


def _returns_df() -> pd.DataFrame:
    index = pd.date_range("2025-01-31", periods=8, freq="ME")
    return pd.DataFrame(
        {
            "VOO": [0.02, -0.01, 0.01, 0.015, -0.005, 0.01, 0.012, 0.008],
            "QQQ": [0.025, -0.015, 0.012, 0.02, -0.006, 0.015, 0.01, 0.011],
            "UFO": [0.08, -0.05, 0.04, 0.03, -0.02, 0.05, 0.02, 0.03],
        },
        index=index,
    )


def _correction_sensitive_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
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


def _correction_sensitive_returns_df() -> pd.DataFrame:
    index = pd.date_range("2025-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "VOO": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "UFO": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        },
        index=index,
    )


def test_counterfactual_preserves_baseline_and_computes_deltas():
    result = run_counterfactual_simulation(
        _metrics_df(),
        "current_proposal",
        rc_over_thresh_pct=2.0,
        e_thresh=0.5,
    )

    assert "baseline" in result
    assert "scenario" in result
    assert result["baseline"]["weights"] == result["scenario"]["weights"]
    assert result["deltas"]["assets"]
    assert all(row["delta_weight_pct"] == 0 for row in result["deltas"]["assets"])
    assert result["interpretation"][0].startswith("이 결과는 정책 적용 결과 비교")


def test_pause_satellite_new_buys_blocks_satellite_increase_signal():
    result = run_counterfactual_simulation(
        _metrics_df(),
        "pause_satellite_new_buys",
        rc_over_thresh_pct=100.0,
        e_thresh=0.5,
    )

    assert result["scenario"]["actions"]["UFO"] != "increase_dca"
    assert result["deltas"]["groups"]["core"]["delta_pct"] >= -0.01
    assert result["deltas"]["groups"]["satellite"]["delta_pct"] <= 0.01
    assert all(
        change["scenario_action"] != "increase_dca" or change["ticker"] != "UFO"
        for change in result["action_changes"]
    )


def test_pause_satellite_new_buys_keeps_satellite_reduction_available_when_overweight():
    result = run_counterfactual_simulation(
        _correction_sensitive_metrics_df(),
        "pause_satellite_new_buys",
        rc_over_thresh_pct=1.0,
        e_thresh=0.5,
    )

    assert result["scenario"]["actions"]["UFO"] in {
        "risk_control_review",
        "rebalance_sell_review",
        "reduce_or_pause_dca",
    }
    assert result["deltas"]["groups"]["satellite"]["delta_pct"] == 0


def test_core_reinforcement_reports_core_gap_direction_and_warnings():
    result = run_counterfactual_simulation(
        _metrics_df(),
        "core_reinforcement",
        rc_over_thresh_pct=100.0,
        e_thresh=0.5,
    )

    assert "core" in result["deltas"]["groups"]
    assert result["deltas"]["groups"]["core"]["delta_pct"] >= -0.01
    assert any("데이터 품질 경고" in warning for warning in result["warnings"])
    assert any("투자 논리 점검" in warning for warning in result["warnings"])


def test_core_reinforcement_does_not_reduce_already_high_core_weight():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ", "UFO"],
            "가중치": [0.70, 0.23, 0.07],
            "위험기여도": [0.45, 0.25, 0.30],
            "E": [0.8, 0.7, 0.65],
            "return_total": [0.1, 0.08, 0.2],
            "group": ["core", "core", "satellite"],
            "dca_enabled": [True, True, True],
            "thesis_status": ["intact", "intact", "watch"],
            "missing_ratio": [0.0, 0.0, 0.0],
            "observation_count": [120, 120, 120],
        }
    ).set_index("ticker")

    result = run_counterfactual_simulation(
        metrics_df,
        "core_reinforcement",
        rc_over_thresh_pct=100.0,
        e_thresh=0.5,
    )

    assert result["deltas"]["groups"]["core"]["delta_pct"] >= -0.01
    assert result["scenario"]["group_weights"]["core"] >= result["baseline"]["group_weights"]["core"] - 0.0001


def test_core_reinforcement_target_does_not_lower_current_core_weight():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "QQQ", "UFO"],
            "가중치": [0.70, 0.23, 0.07],
            "위험기여도": [0.45, 0.25, 0.30],
            "E": [0.8, 0.7, 0.65],
            "return_total": [0.1, 0.08, 0.2],
            "group": ["core", "core", "satellite"],
            "dca_enabled": [True, True, True],
            "thesis_status": ["intact", "intact", "watch"],
            "missing_ratio": [0.0, 0.0, 0.0],
            "observation_count": [120, 120, 120],
        }
    ).set_index("ticker")

    prepared = _prepare_metrics(metrics_df)
    target = _scenario_target_weights(prepared, "core_reinforcement", load_ips_config())
    current_core = float(prepared.loc[prepared["group"] == "core", "가중치"].sum())
    target_core = float(target.loc[prepared["group"] == "core"].sum())

    assert target_core >= current_core


def test_ips_backtest_returns_ips_and_performance_metrics_without_nan():
    returns = _returns_df()
    cov_matrix = annualize_cov(returns)

    result = run_ips_backtest(
        returns,
        _metrics_df(),
        ["current_ips", "core_first_dca", "pause_overweight_satellite"],
        cov_matrix=cov_matrix,
        frequency="monthly",
    )

    assert len(result["strategy_summaries"]) == 3
    assert result["ips_fit_summary"]
    assert result["performance_summary"]
    assert result["timeline"]
    for summary in result["strategy_summaries"]:
        assert "strategy_label" in summary
        assert summary["ips_violation_count"] >= 0
        assert summary["adjustment_count"] >= 0
        assert summary["cagr"] is not None
        assert summary["volatility"] is not None


def test_ips_backtest_deduplicates_repeated_strategy_inputs():
    result = run_ips_backtest(
        _returns_df(),
        _metrics_df(),
        ["current_ips", "current_ips", "core_first_dca"],
        frequency="monthly",
    )

    assert [row["strategy"] for row in result["strategy_summaries"]].count("current_ips") == 1
    assert {row["strategy_label"] for row in result["strategy_summaries"]} >= {
        "현재 IPS 유지",
        "코어 부족분 우선",
    }


def test_ips_backtest_cagr_uses_monthly_horizon_not_trading_day_count():
    result = run_ips_backtest(
        _returns_df(),
        _metrics_df(),
        ["return_chasing_reference"],
        frequency="monthly",
    )

    summary = result["strategy_summaries"][0]
    assert summary["strategy"] == "return_chasing_reference"
    assert summary["cagr"] is not None
    assert -1.0 < summary["cagr"] < 2.0


def test_ips_backtest_applies_selected_decision_context_to_monthly_policy_gate():
    regular = run_ips_backtest(
        _correction_sensitive_returns_df(),
        _correction_sensitive_metrics_df(),
        ["current_ips"],
        frequency="monthly",
        decision_context="regular_review",
    )
    correction = run_ips_backtest(
        _correction_sensitive_returns_df(),
        _correction_sensitive_metrics_df(),
        ["current_ips"],
        frequency="monthly",
        decision_context="market_correction",
    )

    regular_core = regular["timeline"][0]["core_weight"]
    correction_core = correction["timeline"][0]["core_weight"]
    assert regular_core > 0.6
    assert correction_core >= regular_core
    assert regular["timeline"][0]["decision_context"] == "regular_review"
    assert correction["timeline"][0]["decision_context"] == "market_correction"
