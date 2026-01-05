import numpy as np
import pandas as pd

# 연간 거래일 수 (표준: 252일)
거래일_연간 = 252


def normalize_weights(weights: pd.Series) -> pd.Series:
    """가중치를 정규화하여 합이 1이 되도록 합니다.

    Args:
        weights: 가중치 시리즈

    Returns:
        정규화된 가중치 시리즈
    """
    s = weights.copy().astype(float)
    total = s.sum()
    if total == 0:
        return s
    return s / total


def cagr_from_series(price: pd.Series) -> float:
    """가격 시리즈에서 CAGR (연복합성장률)을 계산합니다.

    Args:
        price: 정규화된 가격 시리즈 (NAV)

    Returns:
        CAGR (소수점 형식, 예: 0.1234 = 12.34%)
    """
    if price.empty or price.iloc[0] <= 0:
        return np.nan
    n = len(price)
    years = n / 거래일_연간
    return (price.iloc[-1] / price.iloc[0]) ** (1 / years) - 1


def daily_to_annual_vol(dret: pd.Series) -> float:
    """일일 수익률에서 연율화된 변동성을 계산합니다.

    Args:
        dret: 일일 수익률 시리즈

    Returns:
        연율화된 변동성 (표준편차)
    """
    return float(dret.std() * np.sqrt(거래일_연간))


def sharpe_ratio(cagr: float, vol: float, rf: float = 0.0) -> float:
    """샤프 지수를 계산합니다.

    Args:
        cagr: 연복합성장률
        vol: 연율화된 변동성
        rf: 무위험 수익률

    Returns:
        샤프 지수
    """
    if vol is None or vol == 0 or np.isnan(vol):
        return np.nan
    return (cagr - rf) / vol


def max_drawdown(price: pd.Series) -> float:
    """최대 낙폭을 계산합니다.

    Args:
        price: 정규화된 가격 시리즈 (NAV)

    Returns:
        최대 낙폭 (음수, 예: -0.25 = -25%)
    """
    if price.empty:
        return np.nan
    roll_max = price.cummax()
    dd = price / roll_max - 1
    return float(dd.min())


def annualize_cov(dret: pd.DataFrame) -> pd.DataFrame:
    """일일 수익률의 공분산을 연율화합니다.

    Args:
        dret: 일일 수익률 데이터프레임

    Returns:
        연율화된 공분산 행렬
    """
    return dret.cov() * 거래일_연간


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """각 자산의 위험 기여도를 계산합니다.

    Args:
        weights: 자산 가중치 시리즈
        cov: 연율화된 공분산 행렬

    Returns:
        각 자산의 위험 기여도 (전체 포트폴리오 위험의 비율)
    """
    w = weights.values.reshape(-1, 1)
    Sigma = cov.values
    port_var = float(w.T @ Sigma @ w)
    if port_var <= 0:
        return pd.Series(np.nan, index=weights.index)
    mrc = Sigma @ w  # 한계 위험 기여도
    rc = (w * mrc) / port_var
    rc = rc.flatten()
    return pd.Series(rc, index=weights.index)


def compute_rc_target(
    target_weights: pd.Series, cov_matrix: pd.DataFrame
) -> pd.Series:
    """목표 포트폴리오의 이론적 위험 기여도를 계산합니다.

    # AIDEV-NOTE: geometric-rc-target; 공분산 행렬을 고려한 기하학적 RC_Target 계산
    # RC_Target_i = w_target_i * (Σ @ w_target)_i / sqrt(w_target^T @ Σ @ w_target)

    Args:
        target_weights: 목표 포트폴리오 가중치 시리즈
        cov_matrix: 연율화된 공분산 행렬

    Returns:
        목표 포트폴리오의 위험 기여도 시리즈
    """
    return risk_contributions(target_weights, cov_matrix)


def price_to_nav(price: pd.Series) -> pd.Series:
    """가격을 순자산가치(NAV)로 정규화합니다.

    Args:
        price: 가격 시리즈

    Returns:
        정규화된 NAV (시작값 = 1.0)
    """
    return price / price.iloc[0]


def compute_portfolio_nav(dret: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """포트폴리오의 순자산가치(NAV)를 계산합니다.

    Args:
        dret: 일일 수익률 데이터프레임
        weights: 각 자산의 가중치

    Returns:
        포트폴리오 NAV 시리즈
    """
    w = weights.reindex(dret.columns).fillna(0)
    port_dret = (dret * w).sum(axis=1)
    nav = (1 + port_dret).cumprod()
    return nav


def winsorize_returns(
    returns: pd.DataFrame, lower: float = 0.025, upper: float = 0.975
) -> pd.DataFrame:
    """극단값을 제거하여 수익률을 윈저라이즈합니다.

    # AIDEV-NOTE: return-stabilization; 극단값을 클립하여 이상치의 영향을 제거 (공분산 안정화)

    Args:
        returns: 일일 수익률 데이터프레임
        lower: 하단 분위수 (기본값: 2.5%)
        upper: 상단 분위수 (기본값: 97.5%)

    Returns:
        윈저라이즈된 수익률 데이터프레임
    """
    result = returns.copy()
    for col in result.columns:
        q_lower = result[col].quantile(lower)
        q_upper = result[col].quantile(upper)
        result[col] = result[col].clip(lower=q_lower, upper=q_upper)
    return result


def moving_average(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """데이터프레임에 이동평균을 적용합니다.

    # AIDEV-NOTE: smoothing-strategy; 3윈도우 이동평균으로 일일 변동성 감소 및 메트릭 안정화

    Args:
        df: 입력 데이터프레임 (예: 수익률)
        window: 이동평균 윈도우 크기 (기본값: 3)

    Returns:
        이동평균이 적용된 데이터프레임
    """
    return df.rolling(window=window, center=False).mean()


def tracking_error(active_daily_returns: pd.Series) -> float:
    """추적 오차를 계산합니다 (연율화).

    Args:
        active_daily_returns: 자산 수익률 - 벤치마크 수익률

    Returns:
        연율화된 추적 오차 (추적 오차가 없으면 NaN)
    """
    clean_returns = active_daily_returns.dropna()
    if len(clean_returns) == 0:
        return float("nan")
    daily_te = float(clean_returns.std())
    return daily_te * np.sqrt(거래일_연간)


def information_ratio(asset_cagr: float, bench_cagr: float, te: float) -> float:
    """정보 비율(IR)을 계산합니다.

    Args:
        asset_cagr: 자산의 CAGR
        bench_cagr: 벤치마크의 CAGR
        te: 추적 오차 (연율화)

    Returns:
        정보 비율 (TE가 0이거나 NaN이면 NaN)
    """
    if te is None or np.isnan(te) or te == 0:
        return float("nan")
    return (asset_cagr - bench_cagr) / te


def zscore_to_cdf(x: pd.Series) -> pd.Series:
    """z-score를 표준정규분포의 누적분포함수(CDF) 값으로 변환합니다.

    # AIDEV-NOTE: normalization-via-cdf; z-score→CDF로 [0,1] 범위로 정규화하여 해석 가능성 향상

    Args:
        x: 입력 시리즈

    Returns:
        [0, 1] 범위로 정규화된 시리즈 (CDF 값)
    """
    from scipy.special import erf

    m = x.mean()
    s = x.std(ddof=0)

    # 표준편차가 0이거나 NaN인 경우: 모든 값을 0.5로 설정 (중앙값)
    if s == 0 or np.isnan(s):
        return pd.Series(np.full(len(x), 0.5), index=x.index)

    z = (x - m) / s
    cdf_vals = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    return pd.Series(cdf_vals, index=x.index)


def compute_efficiency_score(sharpe: pd.Series, ir: pd.Series) -> pd.Series:
    """효율 점수(E)를 계산합니다: E = 0.6·Sharpe_norm + 0.4·IR_norm.

    # AIDEV-NOTE: efficiency-score-formula; Sharpe와 IR을 z-score→CDF로 정규화한 후 가중 결합

    Args:
        sharpe: 샤프 지수 시리즈
        ir: 정보 비율 시리즈 (NaN 가능)

    Returns:
        효율 점수 E 시리즈 (0~1 범위)
    """
    # IR의 NaN을 중앙값으로 대체 (벤치마크 없는 자산)
    ir_filled = ir.fillna(ir.median())

    s_norm = zscore_to_cdf(sharpe)
    ir_norm = zscore_to_cdf(ir_filled)

    e = 0.6 * s_norm + 0.4 * ir_norm
    return e


def beta(asset_returns: pd.Series, bench_returns: pd.Series) -> float:
    """베타를 계산합니다 (자산의 벤치마크 대비 체계적 위험).

    # AIDEV-NOTE: beta-calculation; 자산과 벤치마크의 공분산을 각각의 분산으로 나누어 계산

    Args:
        asset_returns: 자산의 일일 수익률 시리즈
        bench_returns: 벤치마크의 일일 수익률 시리즈

    Returns:
        베타 (벤치마크 대비 변동성 계수)
    """
    # 길이 맞춰서 공통 인덱스로 정렬
    aligned_asset = asset_returns.reindex(bench_returns.index).dropna()
    aligned_bench = bench_returns.reindex(aligned_asset.index).dropna()

    if len(aligned_bench) < 2:
        return float("nan")

    covariance = float(aligned_asset.cov(aligned_bench))
    bench_variance = float(aligned_bench.var())

    if bench_variance == 0 or np.isnan(bench_variance):
        return float("nan")

    return covariance / bench_variance


def alpha(asset_cagr: float, asset_beta: float, bench_cagr: float, rf: float) -> float:
    """알파를 계산합니다 (벤치마크를 초과한 수익률).

    # AIDEV-NOTE: alpha-formula; CAPM 기반: Alpha = CAGR - (RF + Beta × (BenchCAGR - RF))

    Args:
        asset_cagr: 자산의 CAGR
        asset_beta: 자산의 베타
        bench_cagr: 벤치마크의 CAGR
        rf: 무위험 수익률

    Returns:
        알파 (초과 수익률, 소수점 형식)
    """
    if np.isnan(asset_beta) or np.isnan(asset_cagr) or np.isnan(bench_cagr):
        return float("nan")

    expected_return = rf + asset_beta * (bench_cagr - rf)
    return asset_cagr - expected_return


def compute_ytd_return(prices: pd.Series) -> float:
    """연초누적 수익률(YTD)를 계산합니다.

    # AIDEV-NOTE: ytd-proxy-computation; 가격 시리즈의 첫 거래일(연초) 대비 마지막 거래일의 수익률

    Args:
        prices: 가격 시리즈

    Returns:
        YTD 수익률 (소수점 형식, 예: 0.1234 = 12.34%)
    """
    if prices.empty or len(prices) < 2:
        return np.nan
    first_price = prices.iloc[0]
    last_price = prices.iloc[-1]
    if first_price <= 0:
        return np.nan
    return (last_price - first_price) / first_price


def apply_momentum_adjustment(
    efficiency_scores: pd.Series,
    return_ytd_series: pd.Series,
    momentum_weight: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """효율 점수에 수익률 모멘텀 보정을 적용합니다.

    # AIDEV-NOTE: momentum-adjustment-formula; E′ = E + momentum_weight × normalized(return_ytd)
    # - z-score→CDF 변환으로 정규화 (Sharpe/IR과 일관성 유지)
    # - return_ytd 없는 자산은 중앙값으로 대체

    Args:
        efficiency_scores: 원본 효율 점수 E 시리즈
        return_ytd_series: YTD 수익률 시리즈 (NaN 포함 가능)
        momentum_weight: 모멘텀 가중치 (기본값: 0.2)

    Returns:
        (보정된 효율 점수 E′, 정규화된 수익률 값)
    """
    # 정규화: z-score → CDF (Sharpe/IR과 일관성 유지)
    ytd_filled = return_ytd_series.fillna(return_ytd_series.median())
    ytd_normalized = zscore_to_cdf(ytd_filled)

    # E′ = E + momentum_weight × normalized
    efficiency_prime = efficiency_scores + momentum_weight * ytd_normalized
    efficiency_prime = efficiency_prime.clip(lower=0, upper=1.0)  # [0, 1] 범위 제한

    return efficiency_prime, ytd_normalized


def preprocess_return_total(
    returns: pd.Series,
    lower: float = 0.025,
    upper: float = 0.975,
    use_log: bool = False,
) -> pd.Series:
    """누적 수익률(return_total)을 전처리합니다.

    # AIDEV-NOTE: return-total-preprocessing; winsorize + 선택적 log1p로 극단값 완화

    Args:
        returns: 누적 수익률 시리즈 (소수점 형식, 예: 0.1234 = 12.34%)
        lower: 하단 분위수 (기본값: 2.5%)
        upper: 상단 분위수 (기본값: 97.5%)
        use_log: log1p 변환 여부 (극단값 완화, 기본값: False)

    Returns:
        전처리된 수익률 시리즈
    """
    result = returns.copy()

    # Step 1: Winsorize
    q_lower = result.quantile(lower)
    q_upper = result.quantile(upper)
    result = result.clip(lower=q_lower, upper=q_upper)

    # Step 2: Optional log transformation
    if use_log:
        # log1p(r): log(1 + r)로 극단값 압축
        result = np.log1p(result)

    return result
