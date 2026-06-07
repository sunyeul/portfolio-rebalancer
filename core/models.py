"""Pydantic 데이터 모델: 포트폴리오 분석 결과 구조.

# AIDEV-NOTE: pydantic-models;
# - AssetMetrics: 각 자산의 분석 결과 (CAGR, Sharpe, RC, etc.)
# - PortfolioMetrics: 포트폴리오 수준 결과
# - ProposalRow: 리밸런싱 제안 행
# AIDEV-TODO: json-export; Pydantic 모델의 model_dump_json() 활용하여 결과 저장 기능 추가
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import numpy as np


class AssetMetrics(BaseModel):
    """자산별 분석 지표.

    # AIDEV-NOTE: metrics-precision; NaN 값은 None으로 변환하여 JSON 직렬화 호환성 보장
    """

    ticker: str = Field(..., description="자산 티커")
    cagr: float | None = Field(None, description="연복합성장률 (CAGR)")
    volatility: float | None = Field(None, description="연율화된 변동성 (%)")
    sharpe: float | None = Field(None, description="샤프 지수")
    max_drawdown: float | None = Field(None, description="최대 낙폭 (%)")
    information_ratio: float | None = Field(None, description="정보 비율 (IR)")
    beta: float | None = Field(None, description="베타")
    alpha: float | None = Field(None, description="알파")
    risk_contribution: float | None = Field(None, description="위험기여도")
    return_contribution: float | None = Field(None, description="수익기여도")
    weight: float = Field(..., ge=0, le=1, description="현재 가중치")
    efficiency_score: float | None = Field(None, ge=0, le=1, description="효율 점수 E")
    return_total: float | None = Field(
        None, description="YTD 수익률 (소수, 예: 0.1234 = 12.34%)"
    )
    efficiency_score_prime: float | None = Field(
        None, ge=0, le=1, description="보정된 효율 점수 E′"
    )
    dca_intensity_score: float | None = Field(
        None, ge=0, le=1, description="정기매수 강도 참고 점수"
    )
    group: str = Field("ungrouped", description="IPS 관리 그룹")
    role: str = Field("unknown", description="자산 역할")
    dca_enabled: bool = Field(True, description="정기매수 조정 대상 여부")
    thesis_status: str = Field("unknown", description="투자 논리 상태")

    @field_validator("*", mode="before")
    @classmethod
    def convert_nan_to_none(cls, v):
        """NaN 값을 None으로 변환하여 JSON 호환성 보장."""
        if isinstance(v, float) and np.isnan(v):
            return None
        return v


class PortfolioMetrics(BaseModel):
    """포트폴리오 수준 분석 지표."""

    cagr: float | None = Field(None, description="포트폴리오 CAGR")
    volatility: float | None = Field(None, description="포트폴리오 변동성")
    sharpe: float | None = Field(None, description="포트폴리오 샤프 지수")
    max_drawdown: float | None = Field(None, description="포트폴리오 최대 낙폭")

    @field_validator("*", mode="before")
    @classmethod
    def convert_nan_to_none(cls, v):
        """NaN 값을 None으로 변환."""
        if isinstance(v, float) and np.isnan(v):
            return None
        return v


class BenchmarkMetrics(BaseModel):
    """벤치마크 수준 분석 지표."""

    cagr: float | None = Field(None, description="벤치마크 CAGR")
    volatility: float | None = Field(None, description="벤치마크 변동성")
    sharpe: float | None = Field(None, description="벤치마크 샤프 지수")

    @field_validator("*", mode="before")
    @classmethod
    def convert_nan_to_none(cls, v):
        """NaN 값을 None으로 변환."""
        if isinstance(v, float) and np.isnan(v):
            return None
        return v


class ProposalRow(BaseModel):
    """리밸런싱 제안 행."""

    ticker: str = Field(..., description="자산 티커")
    current_weight_pct: float = Field(..., description="현재 가중치 (%)")
    target_weight_pct: float = Field(..., description="목표 가중치 (%)")
    gap_pct: float = Field(..., description="갭 (%)")
    efficiency_score: float | None = Field(None, description="효율 점수")
    rc_over_pct: float = Field(..., description="RC_Over (%)")
    rc_target_pct: float = Field(..., description="RC_Target (%)")
    return_total: float | None = Field(None, description="누적 수익률")
    efficiency_score_prime: float | None = Field(
        None, description="보정된 효율 점수 E′"
    )
    dca_intensity_score: float | None = Field(
        None, description="정기매수 강도 참고 점수"
    )
    group: str = Field("ungrouped", description="IPS 관리 그룹")
    role: str = Field("unknown", description="자산 역할")
    dca_enabled: bool = Field(True, description="정기매수 조정 대상 여부")
    thesis_status: str = Field("unknown", description="투자 논리 상태")
    risk_over: bool = Field(..., description="위험기여도 초과 여부")
    efficiency_good: bool = Field(..., description="효율 점수 양호 여부")
    within_hysteresis: bool = Field(..., description="히스테리시스 대역 내")
    below_min_trade: bool = Field(..., description="최소거래 미만")
    should_execute: bool = Field(..., description="실행 여부")
    adjusted_gap_pct: float | None = Field(None, description="조정 후 갭 (%)")

    @field_validator("efficiency_score", mode="before")
    @classmethod
    def convert_nan_to_none(cls, v):
        """NaN 값을 None으로 변환."""
        if isinstance(v, float) and np.isnan(v):
            return None
        return v


class RCViolation(BaseModel):
    """RC 상한선 위반 항목."""

    ticker: str = Field(..., description="자산 티커")
    current_rc_pct: float = Field(..., description="현재 RC (%)")
    rc_cap_pct: float = Field(..., description="RC 상한 (%)")
    status: str = Field(..., description="상태 메시지")


class IPSActionRow(BaseModel):
    """IPS 기반 운영 액션 행."""

    ticker: str = Field(..., description="자산 티커")
    risk_over: bool = Field(..., description="위험기여도 초과 여부")
    efficiency_good: bool = Field(..., description="효율 점수 양호 여부")
    ips_action: str = Field(..., description="IPS 액션 코드")
    action_label: str = Field(..., description="표시용 액션명")
    action_priority: int = Field(..., description="표시 우선순위")
    reason_codes: List[str] = Field(default_factory=list, description="액션 이유 코드")
    next_step: str = Field(..., description="다음 행동")
    blocked_reason: str | None = Field(None, description="보류 또는 차단 이유")


class IPSGroupSummaryRow(BaseModel):
    """IPS 그룹 요약 행."""

    group_type: str = Field(..., description="그룹 유형")
    group: str = Field(..., description="그룹명")
    weight: float = Field(..., description="현재 비중")
    risk_contribution: float = Field(..., description="위험기여도")
    avg_efficiency: float | None = Field(None, description="평균 E")
    avg_dca_score: float | None = Field(None, description="평균 DCA강도점수")


class RebalancingResult(BaseModel):
    """전체 리밸런싱 분석 결과.

    # AIDEV-NOTE: result-schema; 평가 단계의 모든 출력을 단일 스키마로 캡슐화
    """

    asset_metrics: List[AssetMetrics] = Field(..., description="자산별 지표")
    portfolio_metrics: PortfolioMetrics = Field(..., description="포트폴리오 지표")
    benchmark_metrics: Optional[BenchmarkMetrics] = Field(
        None, description="벤치마크 지표"
    )
    proposal_rows: List[ProposalRow] = Field(..., description="리밸런싱 제안")
    ips_action_rows: List[IPSActionRow] = Field(
        default_factory=list, description="IPS 기반 운영 액션"
    )
    group_summary_rows: List[IPSGroupSummaryRow] = Field(
        default_factory=list, description="IPS 그룹 요약"
    )
    rc_violations: List[RCViolation] = Field(..., description="RC 상한선 위반")
    sell_list: List[ProposalRow] = Field(default_factory=list, description="축소 대상")
    buy_list: List[ProposalRow] = Field(default_factory=list, description="증가 대상")
    fine_tune_list: List[ProposalRow] = Field(
        default_factory=list, description="세부 조정 대상"
    )
