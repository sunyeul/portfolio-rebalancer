"""IPS 기반 그룹 요약 및 액션 분류."""

from __future__ import annotations

import pandas as pd


ACTION_LABELS = {
    "increase_dca": "정기매수 증액 후보",
    "decrease_dca": "정기매수 감액/중단 후보",
    "hold_observe": "유지·관찰",
    "review_thesis": "투자 논리 점검",
    "consider_rebalance_sell": "예외적 리밸런싱 매도 검토",
    "block_action": "행동 보류",
}

NEXT_STEPS = {
    "increase_dca": "다음 정기매수 배분에서 해당 자산 또는 그룹의 비중을 늘립니다.",
    "decrease_dca": "신규 매수 중단 또는 정기매수 배분 축소를 우선 검토합니다.",
    "hold_observe": "매매하지 않고 다음 점검까지 관찰합니다.",
    "review_thesis": "투자 논리, 중복성, ETF 대체 가능성을 점검합니다.",
    "consider_rebalance_sell": "정기매수 조정으로 해결하기 어려운 경우에만 매도 여부를 검토합니다.",
    "block_action": "FOMO, 단기 급락, 평단 방어성 행동을 보류합니다.",
}


def get_group_type(group: str, ips_config: dict) -> str:
    """IPS 설정에서 그룹 타입을 조회합니다."""
    return ips_config.get("groups", {}).get(group, {}).get("type", "unknown")


def classify_range(value: float, cfg: dict) -> str:
    """비중이 IPS 범위 대비 어느 상태인지 분류합니다."""
    if value < cfg["min"]:
        return "under_min"
    if value < cfg["target"]:
        return "under_target"
    if value <= cfg["max"]:
        return "in_range"
    return "over_max"


def compute_group_summary(metrics_df: pd.DataFrame, ips_config: dict) -> pd.DataFrame:
    """자산별 metrics_df에서 IPS 그룹 요약을 계산합니다."""
    df = metrics_df.copy()
    df["group"] = df.get("group", "ungrouped")
    df["group"] = df["group"].fillna("ungrouped")
    df["group_type"] = df["group"].map(lambda g: get_group_type(str(g), ips_config))
    if "DCA강도점수" not in df.columns:
        df["DCA강도점수"] = df["E′"] if "E′" in df.columns else df["E"]

    return (
        df.groupby(["group_type", "group"], as_index=False)
        .agg(
            weight=("가중치", "sum"),
            risk_contribution=("위험기여도", "sum"),
            avg_efficiency=("E", "mean"),
            avg_dca_score=("DCA강도점수", "mean"),
        )
        .sort_values(["group_type", "group"])
    )


def compute_ips_allocation_status(
    group_summary: pd.DataFrame, ips_config: dict
) -> dict:
    """코어/위성 비중이 IPS 범위 안에 있는지 계산합니다."""
    target_cfg = ips_config.get("target_allocation", {})
    core_cfg = target_cfg.get("core", {"min": 0.70, "target": 0.80, "max": 0.90})
    sat_cfg = target_cfg.get("satellite", {"min": 0.10, "target": 0.20, "max": 0.30})

    core_weight = group_summary.loc[
        group_summary["group_type"] == "core", "weight"
    ].sum()
    satellite_weight = group_summary.loc[
        group_summary["group_type"] == "satellite", "weight"
    ].sum()

    return {
        "core_weight": float(core_weight),
        "satellite_weight": float(satellite_weight),
        "core_status": classify_range(float(core_weight), core_cfg),
        "satellite_status": classify_range(float(satellite_weight), sat_cfg),
    }


def _action_result(
    ips_action: str,
    reason_codes: list[str],
    ips_config: dict,
    blocked_reason: str | None = None,
    next_step: str | None = None,
) -> dict:
    priorities = ips_config.get("action_priority", {})
    return {
        "ips_action": ips_action,
        "action_label": ACTION_LABELS[ips_action],
        "action_priority": priorities.get(ips_action, 99),
        "reason_codes": reason_codes,
        "next_step": next_step or NEXT_STEPS[ips_action],
        "blocked_reason": blocked_reason,
    }


def _sell_gate_allows(row: pd.Series | dict, allocation_status: dict) -> bool:
    thesis_status = row.get("thesis_status", "unknown")
    role = row.get("role", "unknown")
    group_type = row.get("group_type", "unknown")
    gap = float(row.get("갭%", 0) or 0)

    return (
        thesis_status == "broken"
        or role in {"duplicate", "small_position"}
        or (
            group_type == "satellite"
            and allocation_status.get("satellite_status") == "over_max"
        )
        or (gap < 0 and abs(gap) >= 1.0 and thesis_status == "broken")
    )


def classify_ips_action(
    row: pd.Series | dict,
    allocation_status: dict,
    ips_config: dict | None = None,
) -> dict:
    """IPS 원천 신호와 메타데이터로 최종 액션을 분류합니다."""
    ips_config = ips_config or {}
    gap = float(row.get("갭%", 0) or 0)
    risk_over = bool(row.get("risk_over", False))
    efficiency_good = bool(row.get("efficiency_good", False))
    group_type = row.get("group_type", "unknown")
    dca_enabled = bool(row.get("dca_enabled", True))
    thesis_status = row.get("thesis_status", "unknown")
    should_execute = bool(row.get("실행", False))

    if not should_execute:
        return _action_result(
            "hold_observe",
            ["within_hysteresis_or_below_min_trade"],
            ips_config,
        )

    if group_type == "unknown":
        next_step = NEXT_STEPS["review_thesis"]
        if allocation_status.get("core_status") in {"under_min", "under_target"}:
            next_step += " 판단이 어려운 자산보다 코어 정기매수 증액을 우선합니다."
        return _action_result(
            "review_thesis", ["unknown_group_type"], ips_config, next_step=next_step
        )

    if not risk_over and efficiency_good:
        if gap > 0 and dca_enabled:
            return _action_result(
                "increase_dca",
                ["risk_ok", "efficiency_good", "positive_gap"],
                ips_config,
            )
        reason_codes = ["risk_ok", "efficiency_good"]
        if gap > 0:
            reason_codes.extend(["positive_gap", "dca_disabled"])
        return _action_result("hold_observe", reason_codes, ips_config)

    if risk_over and efficiency_good:
        if gap < 0 and dca_enabled:
            return _action_result(
                "decrease_dca",
                ["risk_over", "efficiency_good", "negative_gap"],
                ips_config,
            )
        reason_codes = ["risk_over", "efficiency_good", "avoid_immediate_increase"]
        if gap < 0:
            reason_codes.extend(["negative_gap", "dca_disabled"])
        return _action_result(
            "hold_observe",
            reason_codes,
            ips_config,
        )

    if not risk_over and not efficiency_good:
        return _action_result(
            "review_thesis", ["risk_ok", "efficiency_low"], ips_config
        )

    if risk_over and not efficiency_good:
        if _sell_gate_allows(row, allocation_status):
            reason_codes = ["risk_over", "efficiency_low", "sell_gate_passed"]
            if thesis_status == "broken":
                reason_codes.append("thesis_broken")
            else:
                reason_codes.append("thesis_not_broken")
            return _action_result(
                "consider_rebalance_sell",
                reason_codes,
                ips_config,
            )
        if gap < 0 and dca_enabled:
            return _action_result(
                "decrease_dca",
                [
                    "risk_over",
                    "efficiency_low",
                    "negative_gap",
                    "thesis_not_broken",
                    "sell_gate_blocked",
                ],
                ips_config,
            )
        return _action_result(
            "review_thesis",
            ["risk_over", "efficiency_low", "thesis_not_broken", "sell_gate_blocked"],
            ips_config,
            blocked_reason="매도 게이트 조건을 충족하지 못했습니다.",
        )

    return _action_result("review_thesis", ["unclassified"], ips_config)


def classify_ips_actions(
    proposal_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    allocation_status: dict,
    ips_config: dict,
) -> pd.DataFrame:
    """proposal_df와 metrics_df를 합쳐 전체 IPS 액션 테이블을 생성합니다."""
    metrics_cols = ["group", "role", "dca_enabled", "thesis_status"]
    meta = metrics_df[metrics_cols].copy()
    meta["group_type"] = meta["group"].map(lambda g: get_group_type(str(g), ips_config))

    df = proposal_df.copy()
    meta_df = meta.reset_index().rename(columns={"index": "ticker"})
    for col in ["ticker", *metrics_cols, "group_type"]:
        if col not in df.columns and col in meta_df.columns:
            df = df.merge(meta_df[["ticker", col]], on="ticker", how="left")
    df["group"] = df["group"].fillna("ungrouped")
    df["role"] = df["role"].fillna("unknown")
    df["dca_enabled"] = df["dca_enabled"].fillna(True).astype(bool)
    df["thesis_status"] = df["thesis_status"].fillna("unknown")
    df["group_type"] = df["group_type"].fillna("unknown")

    action_rows = [
        classify_ips_action(row, allocation_status, ips_config)
        for _, row in df.iterrows()
    ]
    actions = pd.DataFrame(action_rows)
    result = pd.concat([df.reset_index(drop=True), actions], axis=1)
    result["reason_codes_text"] = result["reason_codes"].map(
        lambda codes: ", ".join(codes)
    )
    return result.sort_values(["action_priority", "ticker"]).reset_index(drop=True)
