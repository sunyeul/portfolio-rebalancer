from typing import List
import pandas as pd
import yfinance as yf
import streamlit as st
import time


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """주어진 티커 목록에 대한 가격 데이터를 조회합니다.

    # AIDEV-NOTE: caching-strategy; yfinance 호출이 느려서 @st.cache_data(ttl=300) 사용 (단점: 캐시 무효화 사용자 통제 어려움)
    # AIDEV-TODO: cache-ttl-config; TTL 설정을 사이드바 옵션으로 노출할지 검토 (현재는 고정 300초)

    Args:
        tickers: 티커 문자열 목록
        start: 시작 날짜 (YYYY-MM-DD 형식)
        end: 종료 날짜 (YYYY-MM-DD 형식)

    Returns:
        가격 데이터프레임 (날짜 인덱스, 티커 칼럼)
    """
    max_retries = 2
    base_delay = 1.0  # 초

    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers, start=start, end=end, auto_adjust=True, progress=False
            )
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                prices = data["Close"]
            else:
                prices = data
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
            return prices.dropna(how="all")
        except Exception:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2**attempt)  # 지수 백오프: 1초, 2초
                time.sleep(wait_time)
            else:
                # AIDEV-NOTE: error-propagation; 최종 실패 시 예외를 그대로 전파하여 상위에서 처리하도록 함
                raise


def ensure_tickers_exist(prices: pd.DataFrame, tickers: List[str]) -> List[str]:
    """가격 데이터에 존재하는 티커만 반환합니다.

    Args:
        prices: 가격 데이터프레임
        tickers: 확인할 티커 목록

    Returns:
        데이터에 존재하는 티커 목록
    """
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"{missing}에 대한 가격 데이터 없음. 무시됨.")
    present = [t for t in tickers if t in prices.columns]
    return present


def compute_ytd_returns(prices: pd.DataFrame) -> dict[str, float]:
    """각 티커의 YTD 수익률을 계산합니다.

    # AIDEV-NOTE: ytd-fallback-computation; return_ytd 미제공 시 가격 데이터로부터 계산하는 폴백

    Args:
        prices: 가격 데이터프레임 (칼럼=티커)

    Returns:
        {ticker: ytd_return} 딕셔너리 (소수점 형식)
    """
    result = {}
    for ticker in prices.columns:
        px = prices[ticker].dropna()
        if len(px) >= 2 and px.iloc[0] > 0:
            result[ticker] = (px.iloc[-1] - px.iloc[0]) / px.iloc[0]
        else:
            result[ticker] = float("nan")
    return result
