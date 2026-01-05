"""포트폴리오 리밸런싱 최적화 유틸리티.

# AIDEV-NOTE: constraint-optimization; 현금 중립성과 RC 상한을 동시에 만족하는 반복적 스케일링 알고리즘
"""

import numpy as np
import pandas as pd

from utils.metrics import risk_contributions


def calculate_orders_with_constraints(
    current_weights: pd.Series,
    target_weights: pd.Series,
    current_rc: pd.Series,
    rc_cap: pd.Series,
    cov_matrix: pd.DataFrame,
    max_iterations: int = 10,
    tolerance: float = 0.001,
) -> tuple[pd.Series, dict]:
    """제약 조건을 만족하도록 주문을 조정합니다.

    반복적으로 다음 제약 조건을 만족하도록 조정:
    1. Cash neutrality: sum(orders) ≈ 0
    2. RC cap: RC_i ≤ RC_cap_i for all i

    Args:
        current_weights: 현재 포트폴리오 가중치
        target_weights: 목표 포트폴리오 가중치
        current_rc: 현재 위험 기여도
        rc_cap: 각 자산의 RC 상한선
        cov_matrix: 연율화된 공분산 행렬
        max_iterations: 최대 반복 횟수
        tolerance: 수렴 허용 오차 (현금 중립성)

    Returns:
        (adjusted_orders, convergence_info) 튜플
        - adjusted_orders: 조정된 주문 (가중치 변화량)
        - convergence_info: 수렴 정보 딕셔너리
    """
    # 1. 초기 주문 계산
    orders = target_weights - current_weights

    # 인덱스 정렬 (공분산 행렬과 일치)
    common_index = cov_matrix.index.intersection(orders.index)
    if len(common_index) == 0:
        return orders, {"converged": False, "reason": "no_common_assets"}

    orders = orders.reindex(common_index).fillna(0)
    current_weights = current_weights.reindex(common_index).fillna(0)
    rc_cap = rc_cap.reindex(common_index).fillna(np.inf)
    cov_matrix = cov_matrix.loc[common_index, common_index]

    convergence_info = {
        "converged": False,
        "iterations": 0,
        "cash_neutrality_error": float("inf"),
        "rc_violations": [],
    }

    # 2. 반복적 조정
    for iteration in range(max_iterations):
        # A. Cash Neutrality Scaling
        buys = orders[orders > 0]
        sells = orders[orders < 0]

        if len(buys) == 0 or len(sells) == 0:
            # 매수 또는 매도가 없으면 현금 중립성 자동 만족
            convergence_info["converged"] = True
            convergence_info["iterations"] = iteration + 1
            convergence_info["cash_neutrality_error"] = 0.0
            break

        total_buy = buys.sum()
        total_sell = abs(sells.sum())

        if total_buy == 0 or total_sell == 0:
            # 매수 또는 매도 총액이 0이면 종료
            convergence_info["converged"] = True
            convergence_info["iterations"] = iteration + 1
            convergence_info["cash_neutrality_error"] = abs(total_buy - total_sell)
            break

        # 매도 총액에 맞춰 매수 스케일링 (보수적 접근: 매도를 늘리지 않음)
        scale_factor = min(1.0, total_sell / total_buy)
        orders[orders > 0] *= scale_factor

        # B. 예상 포트폴리오 가중치 계산
        expected_weights = current_weights + orders
        expected_weights = expected_weights.clip(lower=0, upper=1)  # 가중치 범위 제한

        # C. 예상 RC 계산
        expected_rc = risk_contributions(expected_weights, cov_matrix)

        # D. RC 상한선 위반 체크
        violations = expected_rc > rc_cap
        convergence_info["rc_violations"] = violations.sum()

        if not violations.any():
            # 모든 제약 조건 만족
            convergence_info["converged"] = True
            convergence_info["iterations"] = iteration + 1
            convergence_info["cash_neutrality_error"] = abs(orders.sum())
            break

        # E. 위반된 자산의 주문 축소
        # 위반 비율에 따라 주문량을 감소시킴
        violation_mask = violations
        reduction_factor = 0.5  # 감쇠 계수 (보수적 접근)
        orders[violation_mask] *= reduction_factor

        # F. 수렴 체크 (현금 중립성)
        cash_error = abs(orders.sum())
        convergence_info["cash_neutrality_error"] = cash_error
        convergence_info["iterations"] = iteration + 1

        if cash_error < tolerance and convergence_info["rc_violations"] == 0:
            convergence_info["converged"] = True
            break

    return orders, convergence_info
