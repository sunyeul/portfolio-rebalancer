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
        "groups": {
            "core": {"type": "core"},
            "satellite_space": {"type": "satellite"},
        },
        "action_priority": {
            "increase_dca": 1,
            "decrease_dca": 2,
            "review_thesis": 3,
            "consider_rebalance_sell": 4,
            "hold_observe": 5,
            "block_action": 6,
        },
    }


def test_compute_group_summary_and_allocation_status():
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO", "UFO"],
            "가중치": [0.8, 0.2],
            "위험기여도": [0.6, 0.4],
            "E": [0.7, 0.5],
            "DCA강도점수": [0.8, 0.6],
            "group": ["core", "satellite_space"],
        }
    ).set_index("ticker")

    summary = compute_group_summary(metrics_df, _ips_config())
    status = compute_ips_allocation_status(summary, _ips_config())

    assert summary.loc[summary["group_type"] == "core", "weight"].sum() == 0.8
    assert summary.loc[summary["group_type"] == "satellite", "weight"].sum() == 0.2
    assert status["core_status"] == "in_range"
    assert status["satellite_status"] == "in_range"


def test_risk_ok_efficiency_good_positive_gap_classifies_as_increase_dca():
    result = classify_ips_action(
        {
            "risk_over": False,
            "efficiency_good": True,
            "갭%": 3.0,
            "실행": True,
            "group_type": "core",
            "dca_enabled": True,
            "thesis_status": "intact",
        },
        {"core_status": "under_target", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] == "increase_dca"


def test_risk_over_efficiency_low_broken_thesis_can_consider_sell():
    result = classify_ips_action(
        {
            "risk_over": True,
            "efficiency_good": False,
            "갭%": -3.0,
            "실행": True,
            "group_type": "satellite",
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
            "group_type": "satellite",
            "dca_enabled": True,
            "thesis_status": "intact",
        },
        {"core_status": "in_range", "satellite_status": "in_range"},
        _ips_config(),
    )

    assert result["ips_action"] in {"decrease_dca", "review_thesis"}
    assert result["ips_action"] != "consider_rebalance_sell"
