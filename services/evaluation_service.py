"""평가 및 제안 서비스 계층.

# AIDEV-NOTE: service-layer-separation; Streamlit 의존성 제거, 순수 Python 로직으로 재구성
"""

from typing import NamedTuple

import pandas as pd

from utils.metrics import compute_rc_target, risk_contributions
from utils.optimization import calculate_orders_with_constraints
from utils.ips_config import load_ips_config
from utils.ips import (
    DEFAULT_GROUP,
    classify_ips_actions,
    compute_group_summary,
    compute_ips_allocation_status,
    fixed_group,
)


FINAL_EXECUTABLE_ACTIONS = {
    "increase_dca",
    "decrease_dca",
    "consider_rebalance_sell",
}


class EvaluationResult(NamedTuple):
    """평가 결과 데이터 구조."""

    proposal_df: pd.DataFrame
    ips_action_df: pd.DataFrame
    group_summary_df: pd.DataFrame
    sell_list: pd.DataFrame
    buy_list: pd.DataFrame
    fine_tune_list: pd.DataFrame
    rc_violations: pd.DataFrame
    ips_config_snapshot: dict | None = None


class EvaluationError(Exception):
    """평가 처리 중 발생하는 오류."""

    pass


def build_ips_target_weights(metrics_df: pd.DataFrame, ips_config: dict) -> pd.Series:
    """고정 그룹 분류와 IPS 목표로 자산별 목표 비중을 자동 생성합니다."""
    current = metrics_df["가중치"].fillna(0).astype(float)
    group_series = metrics_df.get("group", pd.Series(DEFAULT_GROUP, index=metrics_df.index))
    group_series = group_series.fillna(DEFAULT_GROUP).map(fixed_group)
    target = current.copy()

    locked_mask = group_series.isin(["cash", "unclassified"])
    locked_weight = float(current[locked_mask].sum())
    remaining_weight = max(0.0, 1.0 - locked_weight)
    adjustable_group_values = [
        group for group in ("core", "satellite") if (group_series == group).any()
    ]

    if not adjustable_group_values:
        return target / target.sum() if target.sum() > 0 else target

    target_cfg = ips_config.get("target_allocation", {})
    desired = {
        group: float(target_cfg.get(group, {}).get("target", 0.0))
        for group in adjustable_group_values
    }
    desired_total = sum(desired.values())
    if desired_total <= 0:
        desired = {group: 1.0 for group in adjustable_group_values}
        desired_total = float(len(adjustable_group_values))

    for group in adjustable_group_values:
        group_mask = group_series == group
        group_target = remaining_weight * desired[group] / desired_total
        current_in_group = current[group_mask]
        current_group_total = float(current_in_group.sum())
        if current_group_total > 0:
            target[group_mask] = current_in_group / current_group_total * group_target
        else:
            target[group_mask] = group_target / int(group_mask.sum())

    if target.sum() > 0:
        target = target / target.sum()
    return target


def _action_reason(row: pd.Series | dict) -> str:
    """제안 행의 실행/보류 이유를 사람이 읽는 짧은 문구로 변환합니다."""
    within_hysteresis = bool(row.get("히스테리시스제외", False))
    below_min_trade = bool(row.get("최소거래미만", False))
    should_execute = bool(row.get("실행", False))

    if not should_execute:
        if bool(row.get("data_quality_low", False)):
            return "데이터 신뢰도 낮음"
        if within_hysteresis and below_min_trade:
            return "히스테리시스 범위 및 최소 거래 미만"
        if within_hysteresis:
            return "히스테리시스 범위"
        if below_min_trade:
            return "최소 거래 미만"
        return "보류"

    risk_over = bool(row.get("risk_over", False))
    efficiency_good = bool(row.get("efficiency_good", False))
    gap_pct = float(row.get("갭%", 0) or 0)

    if risk_over and not efficiency_good:
        reason = "위험 초과 및 효율 미달"
    elif risk_over:
        reason = "위험 초과"
    elif not efficiency_good:
        reason = "효율 미달"
    elif gap_pct > 0:
        reason = "목표 대비 부족"
    elif gap_pct < 0:
        reason = "목표 대비 초과"
    else:
        reason = "실행 후보"
    if bool(row.get("data_quality_low", False)):
        reason += " · 데이터 신뢰도 낮음"
    return reason


def _apply_ips_execution_gate(
    proposal: pd.DataFrame,
    ips_action_df: pd.DataFrame,
) -> pd.DataFrame:
    """IPS 액션을 최종 실행 게이트로 적용합니다."""
    gated = proposal.copy()
    action_meta = ips_action_df.set_index("ticker")

    for idx, row in gated.iterrows():
        ticker = row["ticker"]
        action = action_meta.loc[ticker] if ticker in action_meta.index else None
        ips_action = action.get("ips_action") if action is not None else "hold_observe"
        gap_pct = float(row.get("갭%", 0) or 0)
        numeric_candidate = bool(row.get("수치후보", row.get("실행", False)))

        sign_allowed = (
            (ips_action == "increase_dca" and gap_pct > 0)
            or (ips_action in {"decrease_dca", "consider_rebalance_sell"} and gap_pct < 0)
        )
        final_execute = numeric_candidate and ips_action in FINAL_EXECUTABLE_ACTIONS and sign_allowed

        if final_execute:
            gated.at[idx, "실행"] = True
            gated.at[idx, "제안조정%"] = row["참고조정%"]
            gated.at[idx, "판단사유"] = _action_reason(gated.loc[idx])
        else:
            gated.at[idx, "실행"] = False
            gated.at[idx, "제안조정%"] = 0.0
            if numeric_candidate and action is not None:
                gated.at[idx, "판단사유"] = action.get("action_label", "보류")
            else:
                gated.at[idx, "판단사유"] = _action_reason(gated.loc[idx])

    return gated


def _post_trade_rc(
    mdf: pd.DataFrame,
    proposal: pd.DataFrame,
    cov_matrix: pd.DataFrame | None,
) -> pd.Series:
    """최종 제안조정 반영 후 예상 RC를 계산합니다."""
    if cov_matrix is None:
        return mdf["위험기여도"].copy()

    current = mdf["가중치"].astype(float).copy()
    orders = proposal.set_index("ticker")["제안조정%"].reindex(current.index).fillna(0) / 100.0
    expected = (current + orders).clip(lower=0)
    if expected.sum() > 0:
        expected = expected / expected.sum()

    common = cov_matrix.index.intersection(expected.index)
    if len(common) == 0:
        return mdf["위험기여도"].copy()
    rc = risk_contributions(expected.reindex(common), cov_matrix.loc[common, common])
    return rc.reindex(mdf.index).fillna(mdf["위험기여도"])


def run_evaluation(
    metrics_df: pd.DataFrame,
    target_weights: pd.Series | dict[str, float] | None,
    rc_over_thresh_pct: float,
    e_thresh: float,
    cov_matrix: pd.DataFrame | None = None,
) -> EvaluationResult:
    """평가 & 실행 계획 제안을 실행합니다.

    Args:
        metrics_df: 분석 결과 메트릭 데이터프레임
        target_weights: 목표 가중치 (Series 또는 dict, None이면 현재 가중치 사용)
        rc_over_thresh_pct: RC_Over 임계값 (%)
        e_thresh: 효율 점수 E 임계값
        cov_matrix: 연율화된 공분산 행렬 (None이면 단순 비중 기반 RC_Target 사용)

    Returns:
        EvaluationResult: 평가 결과

    Raises:
        EvaluationError: 평가 실패 시
    """
    mdf = metrics_df.copy()
    ips_config_snapshot = load_ips_config()
    mdf["group"] = (
        mdf.get("group", pd.Series(DEFAULT_GROUP, index=mdf.index))
        .fillna(DEFAULT_GROUP)
        .map(fixed_group)
    )
    mdf["dca_enabled"] = (
        mdf.get("dca_enabled", pd.Series(True, index=mdf.index))
        .fillna(True)
        .astype(bool)
    )
    mdf["thesis_status"] = mdf.get("thesis_status", pd.Series(index=mdf.index)).fillna(
        "unknown"
    )
    mdf["return_total"] = mdf.get("return_total", pd.Series(index=mdf.index))
    mdf["missing_ratio"] = mdf.get("missing_ratio", pd.Series(0, index=mdf.index)).fillna(0)
    mdf["observation_count"] = (
        mdf.get("observation_count", pd.Series(9999, index=mdf.index))
        .fillna(9999)
    )
    mdf["data_quality_low"] = (
        mdf["missing_ratio"].astype(float) > 0.2
    ) | (mdf["observation_count"].astype(float) < 60)

    # 목표 가중치 시리즈 구축
    if target_weights is None:
        tgt = build_ips_target_weights(mdf, ips_config_snapshot)
    elif isinstance(target_weights, dict):
        tgt = pd.Series(target_weights, index=mdf.index).fillna(0)
    else:
        tgt = target_weights.reindex(mdf.index).fillna(0)

    # RC 타깃 계산: 공분산 행렬 기반 기하학적 계산 또는 단순 비중 기반
    # AIDEV-NOTE: geometric-rc-target; 공분산 행렬을 고려한 기하학적 RC_Target 계산
    if cov_matrix is not None:
        # 공분산 행렬이 제공되면 기하학적 계산
        try:
            rc_target = compute_rc_target(tgt.fillna(0), cov_matrix)
            rc_target = rc_target.reindex(mdf.index).fillna(0)
        except Exception:
            # 계산 실패 시 폴백: 단순 비중 기반
            rc_target = tgt.fillna(0)
    else:
        # 공분산 행렬이 없으면 단순 비중 기반 (하위 호환성)
        rc_target = tgt.fillna(0)
    mdf["RC_Target"] = rc_target
    rc_gap = mdf["위험기여도"] - mdf["RC_Target"]
    mdf["RC_Over"] = rc_gap.clip(lower=0)

    # E를 효율 판단과 정기매수 우선순위에 모두 사용
    mdf["효율E"] = mdf["E"]

    rc_over_pct = mdf["RC_Over"] * 100  # 백분율로 표시
    mdf["risk_over"] = rc_over_pct > rc_over_thresh_pct
    mdf["efficiency_good"] = mdf["효율E"] >= e_thresh

    # 갭 분석
    current_w = mdf["가중치"]
    gap = tgt - current_w

    proposal = pd.DataFrame(
        {
            "ticker": mdf.index,
            "현재%": (current_w * 100).round(2).values,
            "목표%": (tgt * 100).round(2).values,
            "갭%": (gap * 100).round(2).values,
            "E": mdf["E"].round(2).values,
            "RC_Gap%": (rc_gap * 100).round(2).values,
            "RC_Over%": rc_over_pct.round(2).values,
            "RC_Target%": (rc_target * 100).round(2).values,
            "return_total%": (mdf["return_total"] * 100).round(2).values,
            "group": mdf["group"].values,
            "dca_enabled": mdf["dca_enabled"].values,
            "thesis_status": mdf["thesis_status"].values,
            "missing_ratio": mdf["missing_ratio"].values,
            "observation_count": mdf["observation_count"].values,
            "data_quality_low": mdf["data_quality_low"].values,
            "risk_over": mdf["risk_over"].values,
            "efficiency_good": mdf["efficiency_good"].values,
        }
    )

    # AIDEV-NOTE: trade-filtering-rules; 히스테리시스(affinity-based), 최소거래(1.0%p) 적용하여 과잉거래 방지

    # 히스테리시스 밴드: Affinity 기반 (상수항 + 비례항)
    # AIDEV-NOTE: affinity-hysteresis-band; 작은 자산도 최소한의 허용 범위 확보
    hysteresis_constant = 0.005  # 0.5%p 상수항
    hysteresis_factor = 0.15  # 15% 비례항
    max_gap_pct = hysteresis_constant + (tgt * hysteresis_factor).clip(
        lower=hysteresis_constant
    )
    within_band = gap.abs() <= max_gap_pct

    # 최소 거래 단위: 1.0%p 이상만 처리
    min_trade_pct = 1.0 / 100.0
    above_min_trade = gap.abs() >= min_trade_pct

    # 거래 대상 필터링
    should_trade = above_min_trade & (~within_band)

    proposal["히스테리시스제외"] = within_band.values
    proposal["최소거래미만"] = (~above_min_trade).values
    proposal["수치후보"] = should_trade.values
    proposal["실행"] = should_trade.values
    proposal["제안조정%"] = 0.0
    proposal["참고조정%"] = 0.0
    proposal["판단사유"] = proposal.apply(_action_reason, axis=1)

    # 실행 규칙: 우선순위 정의
    sell_list = proposal[(proposal["갭%"] < 0) & proposal["실행"]].copy()
    sell_list = sell_list.sort_values(["현재%", "RC_Over%"], ascending=[False, False])

    buy_list = proposal[(proposal["갭%"] > 0) & proposal["실행"]].copy()
    buy_list = buy_list.sort_values(["갭%", "E"], ascending=[False, False])

    # AIDEV-NOTE: constrained-scaling; 현금 중립성과 RC 상한을 동시에 만족하는 반복적 스케일링
    # 제약 조건을 고려한 주문 조정
    if cov_matrix is not None and len(buy_list) > 0 and len(sell_list) > 0:
        # 현재 가중치와 목표 가중치 준비
        current_w = mdf["가중치"]

        # 초기 주문 (갭 기반, 실행 대상 자산만)
        initial_orders = pd.Series(0.0, index=mdf.index)
        for _, row in buy_list.iterrows():
            ticker = row["ticker"]
            initial_orders[ticker] = row["갭%"] / 100.0  # 양수 (매수)
        for _, row in sell_list.iterrows():
            ticker = row["ticker"]
            initial_orders[ticker] = row["갭%"] / 100.0  # 음수 (매도)

        # RC 상한선 계산
        rc_cap_single = 0.12  # 단일 자산 최대 12%
        rc_cap_target_ratio = 1.5  # RC_Target의 1.5배
        rc_cap = pd.Series(index=mdf.index, dtype=float)
        for ticker in mdf.index:
            rc_cap[ticker] = min(rc_cap_single, rc_target[ticker] * rc_cap_target_ratio)

        # 현재 RC
        current_rc = mdf["위험기여도"]

        # 제약 조건을 만족하도록 주문 조정
        adjusted_orders, conv_info = calculate_orders_with_constraints(
            current_w,
            current_w + initial_orders,  # 목표 가중치 = 현재 + 주문
            current_rc,
            rc_cap,
            cov_matrix,
            max_iterations=10,
            tolerance=0.001,
        )

        # 조정된 주문을 buy_list와 sell_list에 반영
        buy_list = buy_list.copy()
        sell_list = sell_list.copy()
        buy_list["제안조정%"] = buy_list["갭%"].round(2)
        sell_list["제안조정%"] = sell_list["갭%"].round(2)

        for idx in buy_list.index:
            ticker = buy_list.loc[idx, "ticker"]
            if ticker in adjusted_orders.index:
                adjusted_gap = adjusted_orders[ticker] * 100.0
                buy_list.at[idx, "제안조정%"] = round(adjusted_gap, 2)

        for idx in sell_list.index:
            ticker = sell_list.loc[idx, "ticker"]
            if ticker in adjusted_orders.index:
                adjusted_gap = adjusted_orders[ticker] * 100.0
                sell_list.at[idx, "제안조정%"] = round(adjusted_gap, 2)
    else:
        # 공분산 행렬이 없거나 실행 대상이 없으면 단순 스케일링 (하위 호환성)
        total_sell = sell_list["갭%"].abs().sum() if len(sell_list) > 0 else 0
        total_buy_before_scale = buy_list["갭%"].sum() if len(buy_list) > 0 else 0

        if total_buy_before_scale > 0 and total_sell > 0:
            scale_factor = (
                min(1.0, total_sell / total_buy_before_scale)
                if total_buy_before_scale > 0
                else 1.0
            )
            buy_list["제안조정%"] = (buy_list["갭%"] * scale_factor).round(2)
        else:
            buy_list["제안조정%"] = buy_list["갭%"].round(2)

        if len(sell_list) > 0:
            sell_list["제안조정%"] = sell_list["갭%"].round(2)

    for trade_list in (buy_list, sell_list):
        for idx, row in trade_list.iterrows():
            proposal.at[idx, "참고조정%"] = row["제안조정%"]
            proposal.at[idx, "제안조정%"] = row["제안조정%"]

    group_summary_df = compute_group_summary(mdf, ips_config_snapshot)
    allocation_status = compute_ips_allocation_status(group_summary_df, ips_config_snapshot)
    ips_action_df = classify_ips_actions(
        proposal_df=proposal,
        metrics_df=mdf,
        group_summary_df=group_summary_df,
        allocation_status=allocation_status,
        ips_config=ips_config_snapshot,
    )
    proposal = _apply_ips_execution_gate(proposal, ips_action_df)
    ips_action_df = classify_ips_actions(
        proposal_df=proposal,
        metrics_df=mdf,
        group_summary_df=group_summary_df,
        allocation_status=allocation_status,
        ips_config=ips_config_snapshot,
    )

    sell_list = proposal[(proposal["갭%"] < 0) & proposal["실행"]].copy()
    sell_list = sell_list.sort_values(["현재%", "RC_Over%"], ascending=[False, False])
    buy_list = proposal[(proposal["갭%"] > 0) & proposal["실행"]].copy()
    buy_list = buy_list.sort_values(["갭%", "E"], ascending=[False, False])
    fine_tune = proposal[
        (proposal["실행"]) & (proposal["갭%"].abs() <= 1.0)
    ].copy()

    post_trade_rc = _post_trade_rc(mdf, proposal, cov_matrix)
    rc_cap_single = 0.12  # 단일 자산 최대 12%
    rc_cap_target_ratio = 1.5  # RC_Target의 1.5배
    violations = []
    for ticker in mdf.index:
        rc_cap = min(rc_cap_single, rc_target[ticker] * rc_cap_target_ratio)
        if post_trade_rc[ticker] > rc_cap:
            violations.append(
                {
                    "ticker": ticker,
                    "현재RC%": (post_trade_rc[ticker] * 100),
                    "RC상한%": (rc_cap * 100),
                    "상태": "⚠️ 경고: RC 상한 초과 위험",
                }
            )
    rc_violations_df = pd.DataFrame(violations) if violations else pd.DataFrame()

    return EvaluationResult(
        proposal_df=proposal,
        ips_action_df=ips_action_df,
        group_summary_df=group_summary_df,
        sell_list=sell_list,
        buy_list=buy_list,
        fine_tune_list=fine_tune,
        rc_violations=rc_violations_df,
        ips_config_snapshot=ips_config_snapshot,
    )
