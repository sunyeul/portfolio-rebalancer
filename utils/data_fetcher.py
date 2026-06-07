from typing import List
import pandas as pd
import yfinance as yf
import time
from functools import lru_cache


# AIDEV-NOTE: caching-strategy; yfinance 호출이 느려서 @lru_cache 사용 (TTL은 서비스 계층에서 관리)
@lru_cache(maxsize=128)
def fetch_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """주어진 티커 목록에 대한 가격 데이터를 조회합니다.

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
            # 튜플을 리스트로 변환
            ticker_list = list(tickers)
            data = yf.download(
                ticker_list, start=start, end=end, auto_adjust=True, progress=False
            )
            if isinstance(data, pd.DataFrame) and "Close" in data.columns:
                prices = data["Close"]
            else:
                prices = data
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
            return prices.dropna(axis=1, how="all").dropna(how="all")
        except Exception:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2**attempt)  # 지수 백오프: 1초, 2초
                time.sleep(wait_time)
            else:
                # AIDEV-NOTE: error-propagation; 최종 실패 시 예외를 그대로 전파하여 상위에서 처리하도록 함
                raise


def ensure_tickers_exist(
    prices: pd.DataFrame, tickers: List[str]
) -> tuple[List[str], List[str]]:
    """가격 데이터에 존재하는 티커만 반환합니다.

    Args:
        prices: 가격 데이터프레임
        tickers: 확인할 티커 목록

    Returns:
        (존재하는 티커 목록, 누락된 티커 목록) 튜플
    """
    missing = [
        t for t in tickers if t not in prices.columns or prices[t].dropna().empty
    ]
    present = [
        t for t in tickers if t in prices.columns and not prices[t].dropna().empty
    ]
    return present, missing


def format_no_price_data_message(
    tickers: List[str], start: str, end: str, missing_tickers: List[str] | None = None
) -> str:
    """가격 데이터가 비어 있을 때 사용자가 확인할 티커와 기간을 포함한 메시지를 만듭니다."""
    ticker_text = ", ".join(tickers) if tickers else "없음"
    if missing_tickers:
        missing_text = ", ".join(missing_tickers)
        return (
            "가격 데이터를 찾지 못한 티커가 있습니다. "
            f"문제 티커: {missing_text}. "
            f"조회 기간: {start} ~ {end}. "
            "티커 오타, 상장시장 접미사(.KS/.KQ), Yahoo Finance 지원 여부를 확인하거나 "
            "해당 티커를 제외한 뒤 다시 실행하세요. "
            "참고: SK하이닉스는 000660.KS입니다. "
            f"요청한 전체 티커: {ticker_text}."
        )
    return (
        "선택한 기간에 조회 가능한 가격 데이터가 없습니다. "
        f"조회 기간: {start} ~ {end}. "
        f"요청한 티커: {ticker_text}. "
        "기간을 넓히거나 티커가 Yahoo Finance에서 지원되는지 확인하세요."
    )


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
