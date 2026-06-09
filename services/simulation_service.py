"""IPS-safe counterfactual simulation and limited backtesting services."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from services.evaluation_service import build_ips_target_weights, run_evaluation
from utils.ips import compute_group_summary, compute_ips_allocation_status, fixed_group
from utils.ips_config import load_ips_config
from utils.metrics import (
    daily_to_annual_vol,
    max_drawdown,
    risk_contributions,
    sharpe_ratio,
)


CounterfactualScenario = Literal[
    "current_proposal",
    "core_reinforcement",
    "pause_satellite_new_buys",
    "dca_shift_to_core",
]

BacktestStrategy = Literal[
    "current_ips",
    "core_first_dca",
    "pause_overweight_satellite",
    "return_chasing_reference",
]

BACKTEST_STRATEGY_LABELS = {
    "current_ips": "현재 IPS 유지",
    "core_first_dca": "코어 부족분 우선",
    "pause_overweight_satellite": "위성 초과 신규매수 중단",
    "return_chasing_reference": "수익률 중심 참고",
}


class SimulationError(Exception):
    """Raised when a simulation cannot be computed from the current analysis."""


def _prepare_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    if "ticker" in df.columns:
        df = df.set_index("ticker")
    df["group"] = df.get("group", pd.Series("unclassified", index=df.index)).fillna(
        "unclassified"
    ).map(fixed_group)
    df["dca_enabled"] = (
        df.get("dca_enabled", pd.Series(True, index=df.index)).fillna(True).astype(bool)
    )
    df["thesis_status"] = df.get(
        "thesis_status", pd.Series("unknown", index=df.index)
    ).fillna("unknown")
    df["missing_ratio"] = df.get("missing_ratio", pd.Series(0.0, index=df.index)).fillna(
        0.0
    )
    df["observation_count"] = df.get(
        "observation_count", pd.Series(9999, index=df.index)
    ).fillna(9999)
    return df


def _common_cov(cov_matrix: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame | None:
    if cov_matrix is None:
        return None
    common = cov_matrix.index.intersection(index)
    if len(common) == 0:
        return None
    return cov_matrix.loc[common, common]


def _expected_weights(proposal_df: pd.DataFrame) -> pd.Series:
    current = proposal_df.set_index("ticker")["현재%"].astype(float) / 100.0
    orders = proposal_df.set_index("ticker")["제안조정%"].astype(float) / 100.0
    expected = (current + orders).clip(lower=0.0)
    total = float(expected.sum())
    return expected / total if total > 0 else expected


def _expected_rc(
    weights: pd.Series,
    metrics_df: pd.DataFrame,
    cov_matrix: pd.DataFrame | None,
) -> pd.Series:
    cov = _common_cov(cov_matrix, weights.index)
    if cov is None:
        return metrics_df["위험기여도"].reindex(weights.index).fillna(0.0)
    rc = risk_contributions(weights.reindex(cov.index).fillna(0.0), cov)
    return rc.reindex(weights.index).fillna(0.0)


def _group_weights(weights: pd.Series, metrics_df: pd.DataFrame) -> dict[str, float]:
    groups = metrics_df["group"].reindex(weights.index).fillna("unclassified").map(fixed_group)
    return {
        group: float(weights[groups == group].sum())
        for group in ["core", "satellite", "cash", "unclassified"]
    }


def _scenario_metrics(metrics_df: pd.DataFrame, scenario: CounterfactualScenario) -> pd.DataFrame:
    scenario_df = metrics_df.copy()
    if scenario == "dca_shift_to_core":
        scenario_df.loc[scenario_df["group"] == "core", "dca_enabled"] = True
    return scenario_df


def _redirect_satellite_buy_gap_to_core(
    target: pd.Series,
    current: pd.Series,
    group: pd.Series,
) -> pd.Series:
    adjusted = target.copy()
    satellite_buy_gap = (
        adjusted[(group == "satellite") & (adjusted > current)]
        - current[(group == "satellite") & (adjusted > current)]
    ).clip(lower=0.0)
    shift_amount = float(satellite_buy_gap.sum())
    if shift_amount <= 0:
        return adjusted

    adjusted.loc[satellite_buy_gap.index] = current.reindex(satellite_buy_gap.index)
    core_index = adjusted[group == "core"].index
    if len(core_index) == 0:
        return adjusted

    core_target = adjusted.reindex(core_index).fillna(0.0)
    core_share = (
        core_target / core_target.sum()
        if core_target.sum() > 0
        else pd.Series(1.0 / len(core_index), index=core_index)
    )
    adjusted.loc[core_index] = core_target + core_share * shift_amount
    return adjusted


def _scenario_target_weights(
    metrics_df: pd.DataFrame,
    scenario: CounterfactualScenario,
    ips_config: dict,
) -> pd.Series | None:
    if scenario == "current_proposal":
        return None

    target = build_ips_target_weights(metrics_df, ips_config)
    current = metrics_df["가중치"].astype(float)
    group = metrics_df["group"].map(fixed_group)
    if scenario == "pause_satellite_new_buys":
        target = _redirect_satellite_buy_gap_to_core(target, current, group)
    if scenario == "core_reinforcement":
        target_cfg = ips_config.get("target_allocation", {})
        core_cfg = target_cfg.get("core", {})
        core_target = float(core_cfg.get("max", core_cfg.get("target", 0.8)))
        cash_weight = float(current[group == "cash"].sum())
        adjustable = max(0.0, 1.0 - cash_weight)
        current_core_total = float(current[group == "core"].sum())
        desired_core = min(adjustable, max(current_core_total, core_target * adjustable))
        core_mask = group == "core"
        satellite_mask = group == "satellite"
        if core_mask.any():
            core_current = current[core_mask]
            core_share = core_current / core_current.sum() if core_current.sum() > 0 else pd.Series(
                1.0 / int(core_mask.sum()), index=core_current.index
            )
            target.loc[core_mask] = core_share * max(float(target[core_mask].sum()), desired_core)
        if satellite_mask.any():
            satellite_total = max(0.0, adjustable - float(target[core_mask].sum()))
            satellite_current = current[satellite_mask]
            satellite_share = (
                satellite_current / satellite_current.sum()
                if satellite_current.sum() > 0
                else pd.Series(1.0 / int(satellite_mask.sum()), index=satellite_current.index)
            )
            target.loc[satellite_mask] = satellite_share * satellite_total
    elif scenario == "dca_shift_to_core":
        target = _redirect_satellite_buy_gap_to_core(target, current, group)

    total = float(target.sum())
    return target / total if total > 0 else target


def _evaluation_summary(result, metrics_df: pd.DataFrame, cov_matrix: pd.DataFrame | None) -> dict:
    weights = _expected_weights(result.proposal_df)
    rc = _expected_rc(weights, metrics_df, cov_matrix)
    proposal = result.proposal_df.set_index("ticker")
    actions = result.ips_action_df.set_index("ticker")
    return {
        "weights": {ticker: float(value) for ticker, value in weights.items()},
        "risk_contributions": {ticker: float(value) for ticker, value in rc.items()},
        "target_gaps_pct": {
            ticker: float(value)
            for ticker, value in proposal["갭%"].astype(float).items()
        },
        "group_weights": _group_weights(weights, metrics_df),
        "actions": {
            ticker: str(row.get("ips_action", "hold_observe"))
            for ticker, row in actions.iterrows()
        },
        "action_labels": {
            ticker: str(row.get("action_label", "유지·관찰"))
            for ticker, row in actions.iterrows()
        },
    }


def _counterfactual_deltas(baseline: dict, scenario: dict) -> dict:
    tickers = sorted(set(baseline["weights"]) | set(scenario["weights"]))
    asset_deltas = []
    for ticker in tickers:
        base_weight = float(baseline["weights"].get(ticker, 0.0))
        scenario_weight = float(scenario["weights"].get(ticker, 0.0))
        base_rc = float(baseline["risk_contributions"].get(ticker, 0.0))
        scenario_rc = float(scenario["risk_contributions"].get(ticker, 0.0))
        asset_deltas.append(
            {
                "ticker": ticker,
                "baseline_weight": base_weight,
                "scenario_weight": scenario_weight,
                "delta_weight_pct": (scenario_weight - base_weight) * 100.0,
                "baseline_risk_contribution": base_rc,
                "scenario_risk_contribution": scenario_rc,
                "delta_risk_contribution_pct": (scenario_rc - base_rc) * 100.0,
                "baseline_gap_pct": float(baseline["target_gaps_pct"].get(ticker, 0.0)),
                "scenario_gap_pct": float(scenario["target_gaps_pct"].get(ticker, 0.0)),
            }
        )

    group_deltas = {
        group: {
            "baseline": float(baseline["group_weights"].get(group, 0.0)),
            "scenario": float(scenario["group_weights"].get(group, 0.0)),
            "delta_pct": (
                float(scenario["group_weights"].get(group, 0.0))
                - float(baseline["group_weights"].get(group, 0.0))
            )
            * 100.0,
        }
        for group in ["core", "satellite", "cash", "unclassified"]
    }
    return {"assets": asset_deltas, "groups": group_deltas}


def _action_changes(baseline: dict, scenario: dict) -> list[dict]:
    changes = []
    for ticker in sorted(set(baseline["actions"]) | set(scenario["actions"])):
        baseline_action = baseline["actions"].get(ticker, "hold_observe")
        scenario_action = scenario["actions"].get(ticker, "hold_observe")
        if baseline_action != scenario_action:
            changes.append(
                {
                    "ticker": ticker,
                    "baseline_action": baseline_action,
                    "scenario_action": scenario_action,
                    "baseline_label": baseline["action_labels"].get(ticker, baseline_action),
                    "scenario_label": scenario["action_labels"].get(ticker, scenario_action),
                }
            )
    return changes


def _warnings(metrics_df: pd.DataFrame, scenario_result) -> list[str]:
    warnings: list[str] = []
    low_quality = metrics_df[
        (metrics_df["missing_ratio"].astype(float) > 0.2)
        | (metrics_df["observation_count"].astype(float) < 60)
    ].index.tolist()
    if low_quality:
        warnings.append(
            "데이터 품질 경고: "
            + ", ".join(low_quality)
            + "는 관측 수 또는 결측률 때문에 정책 적용 결과를 보수적으로 해석해야 합니다."
        )
    thesis_watch = metrics_df[
        metrics_df["thesis_status"].astype(str).isin(["broken", "watch", "unknown"])
    ]
    if not thesis_watch.empty:
        warnings.append(
            "투자 논리 점검: "
            + ", ".join(thesis_watch.index.tolist())
            + "는 수치 변화보다 보유 논리 확인이 먼저입니다."
        )
    for _, row in scenario_result.ips_action_df.iterrows():
        for note in row.get("risk_notes", []) or []:
            if note not in warnings:
                warnings.append(str(note))
    return warnings


def _interpretation(scenario: CounterfactualScenario, deltas: dict, warnings: list[str]) -> list[str]:
    group_deltas = deltas["groups"]
    interpretation = [
        "이 결과는 정책 적용 결과 비교이며 즉시 매수/매도 추천이 아닙니다.",
    ]
    if scenario in {"core_reinforcement", "dca_shift_to_core"}:
        core_delta = group_deltas["core"]["delta_pct"]
        interpretation.append(
            f"코어 비중 변화는 {core_delta:.2f}%p입니다. 정기매수 조정으로 목표 비중 회복 가능성을 확인하세요."
        )
    if scenario == "pause_satellite_new_buys":
        sat_delta = group_deltas["satellite"]["delta_pct"]
        interpretation.append(
            f"위성 비중 변화는 {sat_delta:.2f}%p입니다. 위성 신규매수 중단은 리스크 통제 관점에서 검토합니다."
        )
    if warnings:
        interpretation.append("경고가 있는 자산은 수익률보다 데이터 품질과 투자 논리 확인을 우선합니다.")
    return interpretation


def run_counterfactual_simulation(
    metrics_df: pd.DataFrame,
    scenario: CounterfactualScenario,
    rc_over_thresh_pct: float,
    e_thresh: float,
    cov_matrix: pd.DataFrame | None = None,
    decision_context: str = "regular_review",
) -> dict:
    """Compare the current IPS evaluation with a preset counterfactual policy."""
    if scenario not in {
        "current_proposal",
        "core_reinforcement",
        "pause_satellite_new_buys",
        "dca_shift_to_core",
    }:
        raise SimulationError("지원하지 않는 counterfactual preset입니다.")

    base_metrics = _prepare_metrics(metrics_df)
    ips_config = load_ips_config()
    baseline_result = run_evaluation(
        base_metrics,
        None,
        rc_over_thresh_pct,
        e_thresh,
        cov_matrix=cov_matrix,
        decision_context=decision_context,
    )
    scenario_metrics = _scenario_metrics(base_metrics, scenario)
    scenario_target = _scenario_target_weights(scenario_metrics, scenario, ips_config)
    scenario_result = run_evaluation(
        scenario_metrics,
        scenario_target,
        rc_over_thresh_pct,
        e_thresh,
        cov_matrix=cov_matrix,
        decision_context=decision_context,
    )

    baseline_summary = _evaluation_summary(baseline_result, base_metrics, cov_matrix)
    scenario_summary = _evaluation_summary(scenario_result, scenario_metrics, cov_matrix)
    deltas = _counterfactual_deltas(baseline_summary, scenario_summary)
    warnings = _warnings(scenario_metrics, scenario_result)

    return {
        "baseline": baseline_summary,
        "scenario": scenario_summary,
        "deltas": deltas,
        "action_changes": _action_changes(baseline_summary, scenario_summary),
        "warnings": warnings,
        "interpretation": _interpretation(scenario, deltas, warnings),
    }


def _month_end_returns(returns_smooth: pd.DataFrame) -> pd.DataFrame:
    returns = returns_smooth.copy()
    returns.index = pd.to_datetime(returns.index)
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    return monthly.dropna(how="all")


def _period_cagr(nav: pd.Series) -> float:
    clean = nav.dropna()
    if clean.empty or float(clean.iloc[0]) <= 0:
        return float("nan")
    start = pd.Timestamp(clean.index[0])
    end = pd.Timestamp(clean.index[-1])
    elapsed_days = max(1, (end - start).days)
    years = elapsed_days / 365.25
    return float((float(clean.iloc[-1]) / float(clean.iloc[0])) ** (1.0 / years) - 1.0)


def _strategy_target(
    strategy: BacktestStrategy,
    weights: pd.Series,
    metrics_template: pd.DataFrame,
    ips_config: dict,
    trailing_returns: pd.Series,
) -> pd.Series:
    metrics = metrics_template.copy()
    metrics["가중치"] = weights.reindex(metrics.index).fillna(0.0)
    if strategy == "return_chasing_reference":
        positive = trailing_returns.reindex(metrics.index).fillna(0.0).clip(lower=0.0)
        if float(positive.sum()) > 0:
            target = positive / positive.sum()
        else:
            target = weights.copy()
        return target.reindex(metrics.index).fillna(0.0)

    target = build_ips_target_weights(metrics, ips_config)
    group = metrics["group"].map(fixed_group)
    status = compute_ips_allocation_status(
        compute_group_summary(metrics, ips_config), ips_config
    )
    if strategy == "core_first_dca" and status["core_status"] in {"under_min", "under_target"}:
        core_index = target[group == "core"].index
        if len(core_index) > 0:
            shortfall = max(0.0, float(target[group == "core"].sum() - weights[group == "core"].sum()))
            core_share = target.reindex(core_index)
            core_share = core_share / core_share.sum() if core_share.sum() > 0 else pd.Series(
                1.0 / len(core_index), index=core_index
            )
            target.loc[core_index] = target.loc[core_index] + core_share * shortfall
    elif (
        strategy == "pause_overweight_satellite"
        and status["satellite_status"] == "over_max"
    ):
        satellite_index = target[group == "satellite"].index
        if len(satellite_index) > 0:
            target.loc[satellite_index] = np.minimum(
                target.loc[satellite_index], weights.reindex(satellite_index)
            )

    total = float(target.sum())
    return target / total if total > 0 else weights


def _ips_counts(weights: pd.Series, metrics_template: pd.DataFrame, ips_config: dict, cov_matrix: pd.DataFrame | None) -> dict:
    metrics = metrics_template.copy()
    metrics["가중치"] = weights.reindex(metrics.index).fillna(0.0)
    group_summary = compute_group_summary(metrics, ips_config)
    status = compute_ips_allocation_status(group_summary, ips_config)
    rc = _expected_rc(weights, metrics, cov_matrix)
    target = build_ips_target_weights(metrics, ips_config)
    rc_target = _expected_rc(target, metrics, cov_matrix)
    rc_cap = rc_target.apply(lambda value: min(0.12, float(value) * 1.5))
    return {
        "ips_violation": int(
            status["core_status"] in {"under_min", "over_max"}
            or status["satellite_status"] in {"under_min", "over_max"}
        ),
        "satellite_over": int(status["satellite_status"] == "over_max"),
        "risk_contribution_over": int((rc > rc_cap.reindex(rc.index).fillna(np.inf)).sum()),
        "core_gap": float(
            max(
                0.0,
                float(ips_config.get("target_allocation", {}).get("core", {}).get("target", 0.8))
                - float(weights[metrics["group"] == "core"].sum()),
            )
        ),
    }


def _metrics_for_weights(
    metrics_template: pd.DataFrame,
    weights: pd.Series,
    cov_matrix: pd.DataFrame | None,
) -> pd.DataFrame:
    metrics = metrics_template.copy()
    metrics["가중치"] = weights.reindex(metrics.index).fillna(0.0)
    cov = _common_cov(cov_matrix, metrics.index)
    if cov is not None:
        rc = risk_contributions(
            metrics["가중치"].reindex(cov.index).fillna(0.0),
            cov,
        )
        metrics["위험기여도"] = rc.reindex(metrics.index).fillna(
            metrics.get("위험기여도", pd.Series(0.0, index=metrics.index))
        )
    return metrics


def _evaluation_gated_weights(
    weights: pd.Series,
    target: pd.Series,
    metrics_template: pd.DataFrame,
    cov_matrix: pd.DataFrame | None,
    decision_context: str,
) -> pd.Series:
    metrics = _metrics_for_weights(metrics_template, weights, cov_matrix)
    result = run_evaluation(
        metrics,
        target.reindex(metrics.index).fillna(0.0),
        rc_over_thresh_pct=1.5,
        e_thresh=0.5,
        cov_matrix=cov_matrix,
        decision_context=decision_context,
    )
    gated = _expected_weights(result.proposal_df).reindex(weights.index).fillna(weights)
    total = float(gated.sum())
    return gated / total if total > 0 else weights


def run_ips_backtest(
    returns_smooth: pd.DataFrame,
    metrics_df: pd.DataFrame,
    strategies: list[BacktestStrategy],
    cov_matrix: pd.DataFrame | None = None,
    frequency: Literal["monthly"] = "monthly",
    decision_context: str = "regular_review",
    rf: float = 0.0,
) -> dict:
    """Run a limited monthly IPS policy comparison over existing return data."""
    if frequency != "monthly":
        raise SimulationError("v1 백테스트는 monthly 점검만 지원합니다.")
    if not strategies:
        raise SimulationError("비교할 전략을 1개 이상 선택해주세요.")

    metrics = _prepare_metrics(metrics_df)
    returns = returns_smooth.copy()
    if "Date" in returns.columns or "date" in returns.columns or "index" in returns.columns:
        date_col = "Date" if "Date" in returns.columns else "date" if "date" in returns.columns else "index"
        returns = returns.set_index(date_col)
    common = returns.columns.intersection(metrics.index)
    if len(common) == 0:
        raise SimulationError("백테스트에 사용할 공통 자산 수익률이 없습니다.")
    returns = returns[common].astype(float)
    metrics = metrics.reindex(common)
    monthly_returns = _month_end_returns(returns)
    if monthly_returns.empty:
        raise SimulationError("월간 백테스트를 계산할 수 있는 수익률 데이터가 부족합니다.")

    ips_config = load_ips_config()
    allowed = {
        "current_ips",
        "core_first_dca",
        "pause_overweight_satellite",
        "return_chasing_reference",
    }
    timeline: list[dict] = []
    summaries = []
    ips_fit = []
    performance = []

    ordered_strategies = list(dict.fromkeys(strategies))

    for strategy in ordered_strategies:
        if strategy not in allowed:
            raise SimulationError(f"지원하지 않는 백테스트 전략입니다: {strategy}")
        strategy_label = BACKTEST_STRATEGY_LABELS[strategy]
        weights = metrics["가중치"].astype(float).fillna(0.0)
        weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(
            1.0 / len(metrics.index), index=metrics.index
        )
        nav_values = [1.0]
        nav_index = [monthly_returns.index[0] - pd.offsets.MonthEnd(1)]
        adjustments = 0
        ips_violations = 0
        satellite_over_periods = 0
        rc_over_count = 0
        core_gaps: list[float] = []

        for date, row in monthly_returns.iterrows():
            counts = _ips_counts(weights, metrics, ips_config, cov_matrix)
            ips_violations += counts["ips_violation"]
            satellite_over_periods += counts["satellite_over"]
            rc_over_count += counts["risk_contribution_over"]
            core_gaps.append(counts["core_gap"])

            trailing = monthly_returns.loc[:date].tail(3).mean()
            target = _strategy_target(strategy, weights, metrics, ips_config, trailing)
            gated_target = _evaluation_gated_weights(
                weights,
                target,
                metrics,
                cov_matrix,
                decision_context,
            )
            if float((gated_target - weights).abs().sum()) >= 0.01:
                adjustments += 1
            weights = gated_target
            period_return = float((row.reindex(weights.index).fillna(0.0) * weights).sum())
            nav_values.append(nav_values[-1] * (1.0 + period_return))
            weights = (weights * (1.0 + row.reindex(weights.index).fillna(0.0))).clip(lower=0.0)
            weights = weights / weights.sum() if weights.sum() > 0 else gated_target

            timeline.append(
                {
                    "strategy": strategy,
                    "strategy_label": strategy_label,
                    "decision_context": decision_context,
                    "date": pd.Timestamp(date).date().isoformat(),
                    "nav": float(nav_values[-1]),
                    "core_weight": float(weights[metrics["group"] == "core"].sum()),
                    "satellite_weight": float(weights[metrics["group"] == "satellite"].sum()),
                    "ips_violation": bool(counts["ips_violation"]),
                    "satellite_over": bool(counts["satellite_over"]),
                    "risk_contribution_over_count": counts["risk_contribution_over"],
                }
            )

        nav = pd.Series(nav_values, index=pd.to_datetime(nav_index + list(monthly_returns.index)))
        monthly_nav_returns = nav.pct_change().dropna()
        cagr = _period_cagr(nav)
        vol = daily_to_annual_vol(monthly_nav_returns) / np.sqrt(252 / 12)
        sharpe = sharpe_ratio(cagr, vol, rf)
        mdd = max_drawdown(nav)
        avg_core_gap = float(np.mean(core_gaps)) if core_gaps else 0.0
        months_to_recover = next(
            (idx for idx, gap in enumerate(core_gaps, start=1) if gap <= 0.005),
            None,
        )
        strategy_summary = {
            "strategy": strategy,
            "strategy_label": strategy_label,
            "decision_context": decision_context,
            "cagr": float(cagr) if pd.notna(cagr) else None,
            "volatility": float(vol) if pd.notna(vol) else None,
            "max_drawdown": float(mdd) if pd.notna(mdd) else None,
            "sharpe": float(sharpe) if pd.notna(sharpe) else None,
            "ips_violation_count": int(ips_violations),
            "satellite_over_periods": int(satellite_over_periods),
            "risk_contribution_over_count": int(rc_over_count),
            "adjustment_count": int(adjustments),
            "avg_core_gap": avg_core_gap,
            "months_to_core_target_recovery": months_to_recover,
        }
        summaries.append(strategy_summary)
        ips_fit.append(
            {
                "strategy": strategy,
                "strategy_label": strategy_label,
                "ips_violation_count": int(ips_violations),
                "satellite_over_periods": int(satellite_over_periods),
                "risk_contribution_over_count": int(rc_over_count),
                "adjustment_count": int(adjustments),
                "avg_core_gap": avg_core_gap,
            }
        )
        performance.append(
            {
                "strategy": strategy,
                "strategy_label": strategy_label,
                "cagr": strategy_summary["cagr"],
                "volatility": strategy_summary["volatility"],
                "max_drawdown": strategy_summary["max_drawdown"],
                "sharpe": strategy_summary["sharpe"],
            }
        )

    summaries = sorted(
        summaries,
        key=lambda row: (
            row["ips_violation_count"],
            row["satellite_over_periods"],
            row["risk_contribution_over_count"],
            row["adjustment_count"],
        ),
    )
    return {
        "strategy_summaries": summaries,
        "ips_fit_summary": ips_fit,
        "performance_summary": performance,
        "timeline": timeline,
    }
