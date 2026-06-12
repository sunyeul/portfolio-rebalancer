"""데이터 분석 서비스 계층.

# AIDEV-NOTE: service-layer-separation; Streamlit 의존성 제거, 순수 Python 로직으로 재구성
"""

from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

from utils.data_fetcher import (
    ensure_tickers_exist,
    format_no_price_data_message,
    fetch_prices,
    compute_ytd_returns,
)
from core.asset import DEFAULT_GROUP
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
)

DEFAULT_RF = 0.025
DEFAULT_BENCH = "SPY:80,QQQ:20"


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


class BenchmarkSpec(NamedTuple):
    """분석에 사용할 벤치마크 정의."""

    label: str
    components: pd.Series


def parse_benchmark(bench: str) -> BenchmarkSpec | None:
    """단일 티커 또는 TICKER:WEIGHT 목록을 벤치마크 정의로 파싱합니다."""
    normalized = bench.strip().upper()
    if not normalized:
        return None

    if ":" not in normalized:
        return BenchmarkSpec(normalized, pd.Series({normalized: 1.0}, dtype=float))

    components: dict[str, float] = {}
    for part in normalized.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise AnalysisError(
                "복합 벤치마크는 'SPY:80,QQQ:20' 형식으로 입력해주세요."
            )
        ticker, raw_weight = token.split(":", 1)
        ticker = ticker.strip().upper()
        try:
            weight = float(raw_weight.strip())
        except ValueError as exc:
            raise AnalysisError(
                "복합 벤치마크 비중은 숫자로 입력해주세요. 예: SPY:80,QQQ:20"
            ) from exc
        if not ticker or weight <= 0:
            raise AnalysisError("복합 벤치마크 티커와 비중은 양수여야 합니다.")
        components[ticker] = components.get(ticker, 0.0) + weight

    weights = pd.Series(components, dtype=float)
    if weights.empty or weights.sum() <= 0:
        raise AnalysisError("벤치마크를 1개 이상 입력해주세요.")
    return BenchmarkSpec(normalized, weights / weights.sum())


def _add_composite_benchmark(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark: BenchmarkSpec | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """복합 벤치마크를 가상 NAV/수익률 컬럼으로 추가합니다."""
    if benchmark is None or len(benchmark.components) == 1:
        return prices, returns
    if benchmark.label in prices.columns:
        return prices, returns

    component_tickers = list(benchmark.components.index)
    available = [ticker for ticker in component_tickers if ticker in returns.columns]
    if not available:
        return prices, returns

    weights = benchmark.components.reindex(available)
    weights = weights / weights.sum()
    bench_returns = (returns[available] * weights).sum(axis=1)
    bench_nav = pd.concat(
        [pd.Series([1.0], index=prices.index[:1]), (1 + bench_returns).cumprod()]
    )
    if bench_nav.empty:
        return prices, returns

    prices_with_benchmark = prices.copy()
    returns_with_benchmark = returns.copy()
    prices_with_benchmark[benchmark.label] = bench_nav.reindex(prices.index).ffill()
    returns_with_benchmark[benchmark.label] = bench_returns
    return prices_with_benchmark, returns_with_benchmark


def _price_data_quality(prices_raw: pd.DataFrame, ticker: str) -> dict[str, object]:
    """원본 가격 데이터 기준의 자산별 신뢰도 지표를 계산합니다."""
    if ticker not in prices_raw.columns or len(prices_raw.index) == 0:
        return {
            "data_start": None,
            "data_end": None,
            "observation_count": 0,
            "missing_ratio": np.nan,
        }

    series = prices_raw[ticker]
    valid = series.dropna()
    if valid.empty:
        data_start = None
        data_end = None
    else:
        data_start = pd.Timestamp(valid.index.min()).date().isoformat()
        data_end = pd.Timestamp(valid.index.max()).date().isoformat()

    return {
        "data_start": data_start,
        "data_end": data_end,
        "observation_count": int(valid.shape[0]),
        "missing_ratio": float(series.isna().sum() / len(series)),
    }


def run_analysis(
    asset_df: pd.DataFrame,
    period: int | str,
    rf: float,
    bench: str,
) -> AnalysisResult:
    """데이터 조회 & 보강을 실행합니다.

    Args:
        asset_df: 정규화된 자산 데이터프레임 (ticker, allocation, weight 컬럼 포함)
        period: 평가 기간 (정수: 개월 수 또는 문자열: 'YTD', 'Max')
        rf: 무위험 수익률 (연간, 소수)
        bench: 벤치마크 티커 또는 'SPY:80,QQQ:20' 형식의 복합 벤치마크

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

    benchmark = parse_benchmark(bench)
    bench_label = benchmark.label if benchmark else ""
    benchmark_tickers = list(benchmark.components.index) if benchmark else []

    all_tickers = asset_df["ticker"].tolist()
    for ticker in benchmark_tickers:
        if ticker not in all_tickers:
            all_tickers.append(ticker)

    # 가격 데이터 조회
    try:
        # 튜플로 변환하여 캐싱 가능하게 함
        start_text = start.strftime("%Y-%m-%d")
        end_text = end.strftime("%Y-%m-%d")
        prices = fetch_prices(tuple(all_tickers), start_text, end_text)
    except Exception as e:
        raise AnalysisError(f"가격 데이터 조회 실패: {e}") from e

    present, missing = ensure_tickers_exist(prices, all_tickers)
    prices_raw = prices[present].copy()
    prices = prices_raw.ffill().dropna(how="all")

    if prices.empty or len(prices.columns) == 0:
        raise AnalysisError(
            format_no_price_data_message(all_tickers, start_text, end_text, missing)
        )

    # 일일 수익률 계산
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    prices, returns = _add_composite_benchmark(prices, returns, benchmark)

    # AIDEV-NOTE: return-smoothing; 수익률을 윈저라이즈 + 3-window MA로 스무딩하여 이상치 완화 및 공분산 안정화
    returns_smooth = winsorize_returns(returns)
    returns_smooth = moving_average(returns_smooth).dropna(how="all")

    # 현재 가중치를 사용한 포트폴리오 NAV
    priced_assets = [ticker for ticker in asset_df["ticker"] if ticker in prices.columns]
    weights = asset_df.set_index("ticker")["weight"].reindex(priced_assets).fillna(0)
    weights_no_bench = weights

    # AIDEV-NOTE: priced-asset-guard; 가격 데이터가 확보된 실제 보유 자산이 없으면 포트폴리오 계산 불가능
    if weights_no_bench.sum() == 0:
        raise AnalysisError(
            f"포트폴리오에 실제 자산이 없습니다. "
            f"입력 자산의 가격 데이터가 없거나 벤치마크('{bench_label}')만 조회되었습니다. "
            "최소 1개 이상의 자산을 포트폴리오에 추가해주세요."
        )

    port_nav = compute_portfolio_nav(
        returns_smooth[weights_no_bench.index], weights_no_bench
    )
    bench_nav = price_to_nav(prices[bench_label]) if bench_label in prices.columns else None
    bench_cagr = cagr_from_series(bench_nav) if bench_nav is not None else np.nan
    benchmark_returns = (
        returns_smooth[bench_label]
        if bench_label in returns_smooth.columns
        else pd.Series(dtype=float)
    )

    # 자산별 메트릭 계산 (스무딩된 수익률 사용)
    asset_metrics = []
    for t in priced_assets:
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
        if not benchmark_returns.empty and t != bench_label:
            active_returns = returns_smooth[t] - benchmark_returns
            te = tracking_error(active_returns)
            ir = information_ratio(cagr, bench_cagr, te)

            # AIDEV-NOTE: beta-alpha-computation; 벤치마크 대비 베타 및 알파 계산
            asset_beta = beta(returns_smooth[t], benchmark_returns)
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
                **_price_data_quality(prices_raw, t),
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
    if bench_nav is not None:
        bdr = (
            benchmark_returns
            if not benchmark_returns.empty
            else returns[bench_label]
        )
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

    # IPS 메타데이터 병합
    meta_cols = ["group", "dca_enabled", "thesis_status"]
    asset_meta = asset_df.set_index("ticker")
    for col in meta_cols:
        if col in asset_meta.columns:
            metrics_df[col] = asset_meta[col].reindex(metrics_df.index)

    metrics_df["group"] = metrics_df.get(
        "group", pd.Series(index=metrics_df.index)
    ).fillna(DEFAULT_GROUP)
    metrics_df["dca_enabled"] = (
        metrics_df.get("dca_enabled", pd.Series(True, index=metrics_df.index))
        .fillna(True)
        .astype(bool)
    )
    metrics_df["thesis_status"] = metrics_df.get(
        "thesis_status", pd.Series(index=metrics_df.index)
    ).fillna("unknown")

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
