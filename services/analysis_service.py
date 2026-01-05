"""데이터 분석 서비스 계층.

# AIDEV-NOTE: service-layer-separation; Streamlit 의존성 제거, 순수 Python 로직으로 재구성
"""

from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

from utils.data_fetcher import (
    ensure_tickers_exist,
    fetch_prices,
    compute_ytd_returns,
)
from utils.metrics import (
    annualize_cov,
    cagr_from_series,
    compute_efficiency_score,
    compute_portfolio_nav,
    daily_to_annual_vol,
    information_ratio,
    max_drawdown,
    moving_average,
    normalize_weights,
    price_to_nav,
    risk_contributions,
    sharpe_ratio,
    tracking_error,
    winsorize_returns,
    alpha,
    beta,
    apply_momentum_adjustment,
    preprocess_return_total,
)


class AnalysisResult(NamedTuple):
    """분석 결과 데이터 구조."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    returns_smooth: pd.DataFrame
    weights_no_bench: pd.Series
    metrics_df: pd.DataFrame
    port_nav: pd.Series
    bench_nav: pd.Series | None
    portfolio_metrics: dict[str, float]
    benchmark_metrics: dict[str, float] | None
    missing_tickers: list[str]


class AnalysisError(Exception):
    """분석 처리 중 발생하는 오류."""

    pass


def run_analysis(
    asset_df: pd.DataFrame,
    period: int | str,
    rf: float,
    bench: str,
    momentum_weight: float = 0.2,
) -> AnalysisResult:
    """데이터 조회 & 보강을 실행합니다.

    Args:
        asset_df: 정규화된 자산 데이터프레임 (ticker, allocation, weight 컬럼 포함)
        period: 평가 기간 (정수: 개월 수 또는 문자열: 'YTD', 'Max')
        rf: 무위험 수익률 (연간, 소수)
        bench: 벤치마크 티커
        momentum_weight: 모멘텀 가중치 (E′ 계산용, 기본값: 0.2)

    Returns:
        AnalysisResult: 분석 결과

    Raises:
        AnalysisError: 분석 실패 시
    """
    # 날짜 범위 설정
    end = datetime.today()
    # AIDEV-NOTE: flexible-period-handling; 개월 수를 timedelta로 변환 (정수 입력 지원)
    if isinstance(period, int):
        # 개월 수 기반 계산: 대략 30일/월 사용 (더 정확한 달력 계산도 가능)
        start = end - timedelta(days=period * 30)
    elif period == "YTD":
        start = datetime(end.year, 1, 1)
    else:  # Max
        start = end - timedelta(days=365 * 15)

    all_tickers = asset_df["ticker"].tolist()
    if bench and bench not in all_tickers:
        all_tickers += [bench]

    # 가격 데이터 조회
    try:
        # 튜플로 변환하여 캐싱 가능하게 함
        prices = fetch_prices(
            tuple(all_tickers), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )
    except Exception as e:
        raise AnalysisError(f"가격 데이터 조회 실패: {e}") from e

    present, missing = ensure_tickers_exist(prices, all_tickers)
    prices = prices[present].dropna()

    if prices.empty or len(prices.columns) == 0:
        raise AnalysisError("선택한 티커/시간 범위에 대한 가격 데이터가 없습니다.")

    # 일일 수익률 계산
    returns = prices.pct_change().dropna()

    # AIDEV-NOTE: return-smoothing; 수익률을 윈저라이즈 + 3-window MA로 스무딩하여 이상치 완화 및 공분산 안정화
    returns_smooth = winsorize_returns(returns)
    returns_smooth = moving_average(returns_smooth).dropna()

    # 현재 가중치를 사용한 포트폴리오 NAV (벤치마크 가중치 무시)
    weights = asset_df.set_index("ticker")["weight"].reindex(prices.columns).fillna(0)
    if bench in weights.index:
        weights_no_bench = weights.copy()
        weights_no_bench.loc[bench] = 0
        weights_no_bench = normalize_weights(weights_no_bench)
    else:
        weights_no_bench = weights

    # AIDEV-NOTE: benchmark-only-guard; 벤치마크만 남아 있으면 포트폴리오 계산 불가능하므로 조기 종료
    if weights_no_bench.sum() == 0:
        raise AnalysisError(
            f"포트폴리오에 실제 자산이 없습니다. "
            f"벤치마크('{bench}')만 입력되었으므로 포트폴리오 지표를 계산할 수 없습니다. "
            "최소 1개 이상의 자산을 포트폴리오에 추가해주세요."
        )

    port_nav = compute_portfolio_nav(
        returns_smooth[weights_no_bench.index], weights_no_bench
    )
    bench_nav = price_to_nav(prices[bench]) if bench in prices.columns else None

    # 자산별 메트릭 계산 (스무딩된 수익률 사용)
    asset_metrics = []
    for t in prices.columns:
        px = prices[t].dropna()
        nav = price_to_nav(px)
        dr = (
            returns_smooth[t].dropna()
            if t in returns_smooth.columns
            else pd.Series(dtype=float)
        )
        cagr = cagr_from_series(nav)
        vol = daily_to_annual_vol(dr) if len(dr) > 0 else np.nan
        sr = sharpe_ratio(cagr, vol, rf)
        mdd = max_drawdown(nav)

        # IR 계산: 벤치마크가 있는 경우만
        ir = np.nan
        asset_beta = np.nan
        asset_alpha = np.nan
        if bench in returns_smooth.columns and t != bench:
            active_returns = returns_smooth[t] - returns_smooth[bench]
            te = tracking_error(active_returns)
            bench_cagr = (
                cagr_from_series(price_to_nav(prices[bench]))
                if bench in prices.columns
                else np.nan
            )
            ir = information_ratio(cagr, bench_cagr, te)

            # AIDEV-NOTE: beta-alpha-computation; 벤치마크 대비 베타 및 알파 계산
            asset_beta = beta(returns_smooth[t], returns_smooth[bench])
            asset_alpha = alpha(cagr, asset_beta, bench_cagr, rf)

        asset_metrics.append(
            {
                "ticker": t,
                "CAGR": cagr,
                "변동성": vol,
                "샤프": sr,
                "최대낙폭": mdd,
                "IR": ir,
                "베타": asset_beta,
                "알파": asset_alpha,
            }
        )
    metrics_df = pd.DataFrame(asset_metrics).set_index("ticker")

    # 포트폴리오 수준 메트릭 (스무딩된 수익률 사용)
    port_dr = (returns_smooth[weights_no_bench.index] * weights_no_bench).sum(axis=1)
    port_nav_series = (1 + port_dr).cumprod()
    port_cagr = cagr_from_series(port_nav_series)
    port_vol = daily_to_annual_vol(port_dr)
    port_sharpe = sharpe_ratio(port_cagr, port_vol, rf)
    port_mdd = max_drawdown(port_nav_series)

    portfolio_metrics = {
        "cagr": port_cagr,
        "volatility": port_vol,
        "sharpe": port_sharpe,
        "max_drawdown": port_mdd,
    }

    # 벤치마크 메트릭
    benchmark_metrics = None
    if bench in prices.columns:
        bnav = price_to_nav(prices[bench])
        bdr = (
            returns_smooth[bench] if bench in returns_smooth.columns else returns[bench]
        )
        bench_cagr = cagr_from_series(bnav)
        bench_vol = daily_to_annual_vol(bdr)
        bench_sharpe = sharpe_ratio(bench_cagr, bench_vol, rf)
        benchmark_metrics = {
            "cagr": bench_cagr,
            "volatility": bench_vol,
            "sharpe": bench_sharpe,
        }

    # 위험 기여도 계산 (스무딩된 수익률의 공분산 사용)
    cov_ann = annualize_cov(returns_smooth[weights_no_bench.index])
    rc = risk_contributions(weights_no_bench, cov_ann)
    metrics_df["위험기여도"] = rc

    # 수익률 기여도 (CAGR * 가중치로 근사)
    metrics_df["수익기여도"] = metrics_df["CAGR"].fillna(0) * weights_no_bench.reindex(
        metrics_df.index
    ).fillna(0)

    # 가중치 병합
    metrics_df["가중치"] = weights_no_bench.reindex(metrics_df.index).fillna(0)

    # AIDEV-NOTE: efficiency-score-computation; E = 0.6·Sharpe_norm + 0.4·IR_norm (z-score→CDF 정규화)
    metrics_df["E"] = compute_efficiency_score(metrics_df["샤프"], metrics_df["IR"])

    # return_total 수집 및 보강
    # AIDEV-NOTE: return-total-enrichment; 사용자 입력 return_total 우선, 미제공 시 가격 데이터에서 계산
    asset_return_map = dict(zip(asset_df["ticker"], asset_df["return_total"]))
    computed_returns = compute_ytd_returns(prices[weights_no_bench.index])

    metrics_df["return_total"] = metrics_df.index.map(
        lambda t: asset_return_map.get(t)
        if pd.notna(asset_return_map.get(t))
        else computed_returns.get(t, np.nan)
    )

    # return_total 전처리: winsorize + 선택적 log1p
    # AIDEV-NOTE: return-total-preprocess; 극단값 완화 (winsorize 2.5/97.5 + log1p 선택)
    return_total_preprocessed = preprocess_return_total(
        metrics_df["return_total"], lower=0.025, upper=0.975, use_log=False
    )

    # E′ 보정: 모멘텀 조정 적용 (전처리된 return_total 사용)
    e_prime, return_quantile = apply_momentum_adjustment(
        metrics_df["E"],
        return_total_preprocessed,
        momentum_weight=momentum_weight,
    )
    metrics_df["E′"] = e_prime
    metrics_df["수익률분위"] = return_quantile

    return AnalysisResult(
        prices=prices,
        returns=returns,
        returns_smooth=returns_smooth,
        weights_no_bench=weights_no_bench,
        metrics_df=metrics_df,
        port_nav=port_nav,
        bench_nav=bench_nav,
        portfolio_metrics=portfolio_metrics,
        benchmark_metrics=benchmark_metrics,
        missing_tickers=missing,
    )
