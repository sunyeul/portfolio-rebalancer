from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_fetcher import ensure_tickers_exist, fetch_prices
from utils.helpers import show_metric_help_expander
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


def show_data_analysis(period: int | str, rf: float, bench: str):
    """데이터 조회 & 보강 단계를 표시하고 처리합니다.

    Args:
        period: 평가 기간 (정수: 개월 수 또는 문자열: 'YTD', 'Max')
        rf: 무위험 수익률 (연간, 소수)
        bench: 벤치마크 티커
    """
    st.subheader("2️⃣ 데이터 조회 & 보강")

    if "asset_df" not in st.session_state or st.session_state.asset_df is None:
        st.info("먼저 단계 1을 완료해주세요.")
        return False

    asset_df = st.session_state.asset_df

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
    with st.spinner("가격 데이터 조회 중..."):
        prices = fetch_prices(
            all_tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        )

    present = ensure_tickers_exist(prices, all_tickers)
    prices = prices[present].dropna()

    if prices.empty or len(prices.columns) == 0:
        st.error("선택한 티커/시간 범위에 대한 가격 데이터가 없습니다.")
        return False

    st.write(f"가격 데이터 형태: {prices.shape}")

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
        st.error(
            "❌ 오류: 포트폴리오에 실제 자산이 없습니다. "
            f"벤치마크('{bench}')만 입력되었으므로 포트폴리오 지표를 계산할 수 없습니다. "
            "최소 1개 이상의 자산을 포트폴리오에 추가해주세요."
        )
        return False

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

    # 벤치마크 메트릭
    bench_cagr = bench_vol = bench_sharpe = np.nan
    if bench in prices.columns:
        bnav = price_to_nav(prices[bench])
        bdr = (
            returns_smooth[bench] if bench in returns_smooth.columns else returns[bench]
        )
        bench_cagr = cagr_from_series(bnav)
        bench_vol = daily_to_annual_vol(bdr)
        bench_sharpe = sharpe_ratio(bench_cagr, bench_vol, rf)

    st.markdown("### 포트폴리오 & 벤치마크 메트릭")
    cols = st.columns(4)
    cols[0].metric("포트폴리오 CAGR", f"{port_cagr:.2%}")
    cols[1].metric("포트폴리오 변동성", f"{port_vol:.2%}")
    cols[2].metric("포트폴리오 샤프", f"{port_sharpe:.2f}")
    cols[3].metric("포트폴리오 최대낙폭", f"{port_mdd:.2%}")

    st.info(
        "💡 **지표 설명**: "
        "**CAGR** = 연간 평균 성장률 | "
        "**변동성** = 연율화 변동성 | "
        "**샤프** = 위험 조정 수익률 | "
        "**최대낙폭** = 피크에서 바닥까지의 낙폭"
    )

    if bench in prices.columns:
        cols = st.columns(3)
        cols[0].metric("벤치마크 CAGR", f"{bench_cagr:.2%}")
        cols[1].metric("벤치마크 변동성", f"{bench_vol:.2%}")
        cols[2].metric("벤치마크 샤프", f"{bench_sharpe:.2f}")

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

    st.markdown("### 보강된 데이터 (자산별)")
    st.dataframe(
        metrics_df.style.format(
            {
                "CAGR": "{:.2%}",
                "변동성": "{:.2%}",
                "샤프": "{:.2f}",
                "최대낙폭": "{:.2%}",
                "IR": "{:.2f}",
                "E": "{:.2f}",
                "베타": "{:.2f}",
                "알파": "{:.2%}",
                "위험기여도": "{:.2%}",
                "수익기여도": "{:.2%}",
                "가중치": "{:.2%}",
            }
        ),
        width="stretch",
    )

    show_metric_help_expander()

    # 세션 상태에 저장
    st.session_state.prices = prices
    st.session_state.returns = returns
    st.session_state.returns_smooth = returns_smooth
    st.session_state.weights_no_bench = weights_no_bench
    st.session_state.metrics_df = metrics_df
    st.session_state.port_nav = port_nav
    st.session_state.bench_nav = bench_nav

    return True
