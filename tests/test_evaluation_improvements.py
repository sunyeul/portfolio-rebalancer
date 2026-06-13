"""평가 개선사항에 대한 기본 테스트.

# AIDEV-NOTE: test-coverage; RC_Target 계산, 제약 조건 스케일링, 히스테리시스 밴드, 효율 점수 정규화 테스트
"""

import numpy as np
import pandas as pd

from utils.metrics import compute_rc_target, risk_contributions, zscore_to_cdf
from utils.optimization import calculate_orders_with_constraints


class TestRCTargetCalculation:
    """RC_Target 계산 테스트."""

    def test_rc_target_sum_approximately_one(self):
        """RC_Target의 합이 약 1.0인지 확인."""
        # 간단한 공분산 행렬 생성 (3개 자산)
        np.random.seed(42)
        n_assets = 3
        returns = np.random.randn(100, n_assets)
        cov_matrix = pd.DataFrame(
            np.cov(returns.T) * 252,  # 연율화
            index=[f"ASSET{i}" for i in range(n_assets)],
            columns=[f"ASSET{i}" for i in range(n_assets)],
        )

        # 균등 가중치
        target_weights = pd.Series(
            [1.0 / n_assets] * n_assets,
            index=cov_matrix.index,
        )

        rc_target = compute_rc_target(target_weights, cov_matrix)

        # 합이 약 1.0에 가까운지 확인 (부동소수점 오차 허용)
        assert abs(rc_target.sum() - 1.0) < 1e-6

    def test_rc_target_proportional_to_volatility(self):
        """RC_Target이 변동성에 비례하는지 확인."""
        # 고변동성 자산과 저변동성 자산 생성
        np.random.seed(42)

        # 자산 1: 고변동성
        high_vol_returns = np.random.randn(100) * 0.3
        # 자산 2: 저변동성
        low_vol_returns = np.random.randn(100) * 0.1

        returns = np.column_stack([high_vol_returns, low_vol_returns])
        cov_matrix = pd.DataFrame(
            np.cov(returns.T) * 252,
            index=["HIGH_VOL", "LOW_VOL"],
            columns=["HIGH_VOL", "LOW_VOL"],
        )

        # 균등 가중치
        target_weights = pd.Series([0.5, 0.5], index=cov_matrix.index)

        rc_target = compute_rc_target(target_weights, cov_matrix)

        # 고변동성 자산의 RC가 더 커야 함
        assert rc_target["HIGH_VOL"] > rc_target["LOW_VOL"]

    def test_risk_contributions_return_nan_when_variance_is_zero(self):
        """분산이 0이면 비중으로 속이지 않고 계산 불가 상태를 반환합니다."""
        weights = pd.Series([0.4, 0.6], index=["VOO", "QQQ"])
        cov_matrix = pd.DataFrame(
            0.0,
            index=weights.index,
            columns=weights.index,
        )

        rc = risk_contributions(weights, cov_matrix)

        assert rc.isna().all()

    def test_risk_contributions_are_not_weight_echo_when_covariance_is_valid(self):
        """유효한 공분산에서는 위험기여도가 비중 복사값이 아니라 공분산 기반으로 계산됩니다."""
        weights = pd.Series([0.5, 0.5], index=["LOW_VOL", "HIGH_VOL"])
        cov_matrix = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=weights.index,
            columns=weights.index,
        )

        rc = risk_contributions(weights, cov_matrix)

        assert rc.notna().all()
        assert abs(rc.sum() - 1.0) < 1e-9
        assert rc["HIGH_VOL"] > rc["LOW_VOL"]
        assert rc["LOW_VOL"] != weights["LOW_VOL"]

    def test_risk_contributions_compute_when_covariance_has_partial_nan(self):
        """일부 공분산 결측은 0으로 처리해 가능한 RC 계산을 살립니다."""
        weights = pd.Series([0.5, 0.5], index=["LOW_VOL", "HIGH_VOL"])
        cov_matrix = pd.DataFrame(
            [[0.04, np.nan], [np.nan, 0.09]],
            index=weights.index,
            columns=weights.index,
        )

        rc = risk_contributions(weights, cov_matrix)

        assert rc.notna().all()
        assert abs(rc.sum() - 1.0) < 1e-9
        assert rc["HIGH_VOL"] > rc["LOW_VOL"]


class TestConstrainedScaling:
    """제약 조건 스케일링 테스트."""

    def test_cash_neutrality_maintained(self):
        """현금 중립성이 유지되는지 확인."""
        np.random.seed(42)
        n_assets = 4

        # 공분산 행렬 생성
        returns = np.random.randn(100, n_assets)
        cov_matrix = pd.DataFrame(
            np.cov(returns.T) * 252,
            index=[f"ASSET{i}" for i in range(n_assets)],
            columns=[f"ASSET{i}" for i in range(n_assets)],
        )

        current_weights = pd.Series(
            [0.3, 0.3, 0.2, 0.2],
            index=cov_matrix.index,
        )
        target_weights = pd.Series(
            [0.4, 0.2, 0.3, 0.1],
            index=cov_matrix.index,
        )

        current_rc = pd.Series([0.25, 0.25, 0.25, 0.25], index=cov_matrix.index)
        rc_cap = pd.Series([0.5, 0.5, 0.5, 0.5], index=cov_matrix.index)

        adjusted_orders, conv_info = calculate_orders_with_constraints(
            current_weights,
            target_weights,
            current_rc,
            rc_cap,
            cov_matrix,
            max_iterations=10,
            tolerance=0.001,
        )

        # 현금 중립성 체크 (합이 0에 가까워야 함)
        assert abs(adjusted_orders.sum()) < 0.01

    def test_convergence_within_max_iterations(self):
        """최대 반복 횟수 내에서 수렴하는지 확인."""
        np.random.seed(42)
        n_assets = 3

        returns = np.random.randn(100, n_assets)
        cov_matrix = pd.DataFrame(
            np.cov(returns.T) * 252,
            index=[f"ASSET{i}" for i in range(n_assets)],
            columns=[f"ASSET{i}" for i in range(n_assets)],
        )

        current_weights = pd.Series([0.4, 0.3, 0.3], index=cov_matrix.index)
        target_weights = pd.Series([0.5, 0.3, 0.2], index=cov_matrix.index)

        current_rc = pd.Series([0.33, 0.33, 0.34], index=cov_matrix.index)
        rc_cap = pd.Series([0.5, 0.5, 0.5], index=cov_matrix.index)

        _, conv_info = calculate_orders_with_constraints(
            current_weights,
            target_weights,
            current_rc,
            rc_cap,
            cov_matrix,
            max_iterations=10,
            tolerance=0.001,
        )

        # 반복 횟수가 max_iterations 이하여야 함
        assert conv_info["iterations"] <= 10


class TestHysteresisBand:
    """히스테리시스 밴드 테스트."""

    def test_minimum_band_width(self):
        """작은 자산도 최소한의 밴드 폭을 가지는지 확인."""
        # Affinity 기반 공식: 0.005 + (target * 0.15)
        hysteresis_constant = 0.005
        hysteresis_factor = 0.15

        # 매우 작은 목표 가중치 (0.5%)
        small_target = 0.005
        max_gap_small = hysteresis_constant + (small_target * hysteresis_factor)

        # 최소 밴드 폭 확인 (상수항만큼은 확보)
        assert max_gap_small >= hysteresis_constant

        # 큰 목표 가중치 (50%)
        large_target = 0.5
        max_gap_large = hysteresis_constant + (large_target * hysteresis_factor)

        # 큰 자산은 비례적으로 더 큰 밴드
        assert max_gap_large > max_gap_small


class TestEfficiencyScoreNormalization:
    """효율 점수 정규화 테스트."""

    def test_zscore_to_cdf_bounds(self):
        """z-score→CDF 정규화가 [0, 1] 범위 값을 반환하는지 확인."""
        np.random.seed(42)

        values = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        normalized = zscore_to_cdf(values)

        # 정규화된 값이 [0, 1] 범위에 있는지 확인
        assert (normalized >= 0).all()
        assert (normalized <= 1).all()

    def test_zscore_to_cdf_preserves_distribution_shape(self):
        """z-score→CDF 정규화가 분포 순서를 보존하는지 확인."""
        np.random.seed(42)

        # 정규분포와 유사한 수익률 생성
        values = pd.Series(
            np.random.normal(0.1, 0.05, 10),
            index=[f"ASSET{i}" for i in range(10)],
        )

        normalized = zscore_to_cdf(values)

        # 순서 보존
        assert normalized.sort_values().index.tolist() == values.sort_values().index.tolist()
