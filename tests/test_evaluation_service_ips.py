import pandas as pd

from services.evaluation_service import run_evaluation


def test_run_evaluation_returns_ips_outputs_and_uses_ips_signals():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "가중치": [0.4, 0.6],
            "위험기여도": [0.3, 0.8],
            "E": [0.8, 0.2],
            "E′": [0.6, 0.9],
            "return_total": [0.1, -0.1],
            "group": ["core", "satellite_space"],
            "role": ["broad_etf", "theme_etf"],
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
    assert "DCA강도점수" in result.proposal_df.columns
    assert "E" in result.proposal_df.columns
    assert "E′" in result.proposal_df.columns
    legacy_column = "\uc0ac\ubd84\uba74"
    assert legacy_column not in result.proposal_df.columns
    ufo = result.proposal_df.loc[result.proposal_df["ticker"] == "UFO"].iloc[0]
    assert bool(ufo["risk_over"]) is True
    assert bool(ufo["efficiency_good"]) is False
    assert "ips_action" in result.ips_action_df.columns
    assert "risk_over" in result.ips_action_df.columns
    assert "efficiency_good" in result.ips_action_df.columns
