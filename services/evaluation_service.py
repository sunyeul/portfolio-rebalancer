"""평가 및 제안 서비스 계층.

# AIDEV-NOTE: service-layer-separation; Streamlit 의존성 제거, 순수 Python 로직으로 재구성
"""

import json
from typing import NamedTuple

import numpy as np
import pandas as pd


class EvaluationResult(NamedTuple):
    """평가 결과 데이터 구조."""

    proposal_df: pd.DataFrame
    sell_list: pd.DataFrame
    buy_list: pd.DataFrame
    fine_tune_list: pd.DataFrame
    rc_violations: pd.DataFrame
    quadrant_chart_json: str  # Chart.js JSON


class EvaluationError(Exception):
    """평가 처리 중 발생하는 오류."""

    pass


def run_evaluation(
    metrics_df: pd.DataFrame,
    target_weights: pd.Series | dict[str, float] | None,
    rc_over_thresh_pct: float,
    e_thresh: float,
) -> EvaluationResult:
    """평가 & 실행 계획 제안을 실행합니다.

    # AIDEV-NOTE: rc-over-e-quadrant; X축 RC_Over%, Y축 효율점수 E′(보정)로 2×2 사분면 분류

    Args:
        metrics_df: 분석 결과 메트릭 데이터프레임
        target_weights: 목표 가중치 (Series 또는 dict, None이면 현재 가중치 사용)
        rc_over_thresh_pct: RC_Over 임계값 (%)
        e_thresh: 효율 점수 E′ 임계값

    Returns:
        EvaluationResult: 평가 결과

    Raises:
        EvaluationError: 평가 실패 시
    """
    mdf = metrics_df.copy()

    # 목표 가중치 시리즈 구축
    if target_weights is None:
        tgt = mdf["가중치"]
    elif isinstance(target_weights, dict):
        tgt = pd.Series(target_weights, index=mdf.index).fillna(0)
    else:
        tgt = target_weights.reindex(mdf.index).fillna(0)

    # RC 타깃 계산: 목표 가중치에 비례
    # AIDEV-NOTE: rc-target-proportion; RC_Target = 목표 가중치 (직관적, KISS 원칙)
    rc_target = tgt.fillna(0)
    mdf["RC_Target"] = rc_target
    mdf["RC_Over"] = (mdf["위험기여도"] - mdf["RC_Target"]).clip(lower=0)

    # E′ 사용 (보정된 효율 점수); E′ 없으면 E 폴백
    mdf["효율E"] = mdf["E′"] if "E′" in mdf.columns else mdf["E"]

    # 2×2 사분면 분류: RC_Over (X축) vs E′ (Y축)
    rc_over_pct = mdf["RC_Over"] * 100  # 백분율로 표시
    over_thresh = rc_over_pct > rc_over_thresh_pct
    good_eff = mdf["효율E"] >= e_thresh

    mdf["사분면"] = np.select(
        [
            (~over_thresh) & good_eff,  # Q1: 낮은 RC_Over & 높은 E
            over_thresh & good_eff,  # Q2: 높은 RC_Over & 높은 E
            (~over_thresh) & (~good_eff),  # Q3: 낮은 RC_Over & 낮은 E
            over_thresh & (~good_eff),  # Q4: 높은 RC_Over & 낮은 E
        ],
        ["Q1 핵심", "Q2 성장", "Q3 개선", "Q4 위험관리"],
        default="분류 안됨",
    )

    # Chart.js 호환 데이터 구조 생성
    # AIDEV-NOTE: chartjs-data-structure; Chart.js scatter chart를 위한 데이터 구조 생성
    chart_data_points = [
        {
            "x": float(rc_over_pct[ticker]),
            "y": float(mdf.loc[ticker, "효율E"]),
            "ticker": ticker,
        }
        for ticker in mdf.index
    ]

    chart_config = {
        "datasets": [
            {
                "label": "자산",
                "data": chart_data_points,
                "backgroundColor": "rgba(0, 100, 200, 0.6)",
                "borderColor": "rgba(0, 100, 200, 1)",
                "pointRadius": 8,
                "pointHoverRadius": 10,
            }
        ],
        "thresholds": {
            "rc_over": float(rc_over_thresh_pct),
            "efficiency": float(e_thresh),
        },
        "labels": {
            "x": "RC_Over (%)",
            "y": "효율 점수 E",
            "title": "사분면 분류 (RC_Over vs 효율E)",
        },
    }

    # Chart.js 데이터를 JSON으로 직렬화
    quadrant_chart_json = json.dumps(chart_config, ensure_ascii=False)

    # 갭 분석
    current_w = mdf["가중치"]
    gap = tgt - current_w

    proposal = pd.DataFrame(
        {
            "ticker": mdf.index,
            "사분면": mdf["사분면"].values,
            "현재%": (current_w * 100).round(2).values,
            "목표%": (tgt * 100).round(2).values,
            "갭%": (gap * 100).round(2).values,
            "E′": mdf["효율E"].round(2).values,
            "RC_Over%": rc_over_pct.round(2).values,
            "RC_Target%": (rc_target * 100).round(2).values,
            "return_total%": (mdf["return_total"] * 100).round(2).values,
        }
    )

    # AIDEV-NOTE: trade-filtering-rules; 히스테리시스(±20%), 최소거래(1.0%p) 적용하여 과잉거래 방지

    # 히스테리시스 밴드: 목표 ±20% 내 거래 제외
    # AIDEV-FIXME: hysteresis-definition; 갭의 절대값이 목표의 ±20% 이내면 거래 제외
    max_gap_pct = (tgt * 0.20).clip(lower=0.01)  # 목표의 20% 또는 최소 1%
    within_band = gap.abs() <= max_gap_pct

    # 최소 거래 단위: 1.0%p 이상만 처리
    min_trade_pct = 1.0 / 100.0
    above_min_trade = gap.abs() >= min_trade_pct

    # 거래 대상 필터링
    should_trade = above_min_trade & (~within_band)

    proposal["히스테리시스제외"] = within_band
    proposal["최소거래미만"] = ~above_min_trade
    proposal["실행"] = should_trade

    # 실행 규칙: 우선순위 정의
    sell_list = proposal[
        (proposal["사분면"] == "Q4 위험관리") & proposal["실행"]
    ].copy()
    sell_list = sell_list.sort_values(["현재%", "RC_Over%"], ascending=[False, False])

    buy_list = proposal[(proposal["갭%"] > 0) & proposal["실행"]].copy()
    buy_list = buy_list.sort_values(["갭%", "E′"], ascending=[False, False])

    fine_tune = proposal[
        (proposal["사분면"].isin(["Q1 핵심", "Q2 성장"]))
        & (proposal["실행"])
        & (proposal["갭%"].abs() <= 1.0)
    ].copy()

    # AIDEV-NOTE: cash-neutral-scaling; 매도합계 = 매수합계가 되도록 비례 스케일링
    # AIDEV-FIXME: cash-neutral-edge-case; 단순 스케일링으로 RC 상한 위반 가능, 임의 플래그 추가

    total_sell = sell_list["갭%"].abs().sum() if len(sell_list) > 0 else 0
    total_buy_before_scale = buy_list["갭%"].sum() if len(buy_list) > 0 else 0

    if total_buy_before_scale > 0 and total_sell > 0:
        # 매도 규모에 맞춰 매수 스케일 조정
        scale_factor = (
            min(1.0, total_sell / total_buy_before_scale)
            if total_buy_before_scale > 0
            else 1.0
        )
        buy_list["조정갭%"] = (buy_list["갭%"] * scale_factor).round(2)
    else:
        buy_list["조정갭%"] = buy_list["갭%"].round(2)

    if len(sell_list) > 0:
        sell_list["조정갭%"] = sell_list["갭%"].round(2)

    # RC 상한선 체크 (경고)
    rc_cap_single = 0.12  # 단일 자산 최대 12%
    rc_cap_target_ratio = 1.5  # RC_Target의 1.5배

    post_trade_rc = mdf["위험기여도"].copy()
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
        sell_list=sell_list,
        buy_list=buy_list,
        fine_tune_list=fine_tune,
        rc_violations=rc_violations_df,
        quadrant_chart_json=quadrant_chart_json,
    )
