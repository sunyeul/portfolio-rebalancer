import json
import re
import time
from functools import lru_cache
from typing import List
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf


YAHOO_JAPAN_FINANCE_BASE_URL = "https://finance.yahoo.co.jp/quote"
YAHOO_JAPAN_BFF_BASE_URL = (
    "https://finance.yahoo.co.jp/bff-pc/v1/main/fund/price/history"
)
YAHOO_JAPAN_HISTORY_PAGE_SIZE = 100


def _normalize_japan_fund_ticker(ticker: str) -> str:
    """Yahoo Japan 투자신탁 코드 형태로 정규화합니다."""
    normalized = ticker.strip().upper().removesuffix(".JP")
    if normalized.isdigit() and len(normalized) == 7:
        return normalized.zfill(8)
    return normalized


def _is_japan_investment_trust_ticker(ticker: str) -> bool:
    """명시적 .JP suffix가 붙은 일본 투자신탁 코드인지 판별합니다.

    일반 yfinance 심볼과 충돌하지 않도록 suffix 없는 7~8자리 코드는
    자동 판별하지 않습니다. 예: 0331418A.JP, 29313233.JP, 3311187.JP
    """
    normalized = ticker.strip().upper()
    if not normalized.endswith(".JP"):
        return False
    normalized = normalized.removesuffix(".JP")
    if "." in normalized or "-" in normalized:
        return False
    if len(normalized) == 8 and normalized[:7].isdigit() and normalized[-1].isalnum():
        return True
    return normalized.isdigit() and len(normalized) == 7


def _parse_yahoo_japan_number(value: str) -> float | None:
    text = value.strip().replace(",", "").replace("+", "")
    if text in {"", "-", "--"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_yahoo_japan_fund_history_html(html: str) -> pd.DataFrame:
    """Yahoo Japan 투자신탁 시계열 HTML에서 기준가/순자산 표를 파싱합니다."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []

    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        headers = [cell.get_text(strip=True) for cell in header_cells]
        if not any("日付" in header for header in headers) or not any(
            "基準価額" in header for header in headers
        ):
            continue

        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            values = [cell.get_text(strip=True) for cell in cells]
            if len(values) < 2 or values[0] == "日付":
                continue

            date = pd.to_datetime(values[0], format="%Y年%m月%d日", errors="coerce")
            nav = _parse_yahoo_japan_number(values[1])
            if pd.isna(date) or nav is None:
                continue

            row: dict[str, object] = {"Date": date, "Close": nav}
            if len(values) >= 4:
                net_assets = _parse_yahoo_japan_number(values[3])
                if net_assets is not None:
                    row["NetAssetsMillions"] = net_assets
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Close"])

    frame = pd.DataFrame(rows).drop_duplicates(subset=["Date"]).set_index("Date")
    return frame.sort_index()


def _extract_yahoo_japan_jwt_token(html: str) -> str | None:
    match = re.search(r'"jwtToken":"([^"]+)"', html)
    return match.group(1) if match else None


def _parse_yahoo_japan_fund_history_json(payload: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in payload.get("histories", []):
        date = pd.to_datetime(item.get("date"), format="%Y年%m月%d日", errors="coerce")
        nav = _parse_yahoo_japan_number(str(item.get("price", "")))
        if pd.isna(date) or nav is None:
            continue

        row: dict[str, object] = {"Date": date, "Close": nav}
        net_assets = _parse_yahoo_japan_number(str(item.get("netAssetsBalance", "")))
        if net_assets is not None:
            row["NetAssetsMillions"] = net_assets
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Close"])

    frame = pd.DataFrame(rows).drop_duplicates(subset=["Date"]).set_index("Date")
    return frame.sort_index()


def _fetch_text(url: str, headers: dict[str, str]) -> str:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=10) as response:
        return response.read().decode("utf-8", errors="replace")


def _format_yahoo_japan_api_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _fetch_yahoo_japan_fund_history_via_api(
    ticker: str, jwt_token: str, start: str, end: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    page = 1
    total_page = 1

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
        ),
        "Referer": f"{YAHOO_JAPAN_FINANCE_BASE_URL}/{ticker}/history",
        "jwt-token": jwt_token,
    }

    while page <= total_page:
        params = urlencode(
            {
                "displayedMaxPage": 5,
                "fromDate": _format_yahoo_japan_api_date(start),
                "page": page,
                "size": YAHOO_JAPAN_HISTORY_PAGE_SIZE,
                "timeFrame": "daily",
                "toDate": _format_yahoo_japan_api_date(end),
            }
        )
        url = f"{YAHOO_JAPAN_BFF_BASE_URL}/{ticker}?{params}"
        payload = json.loads(_fetch_text(url, headers))
        frame = _parse_yahoo_japan_fund_history_json(payload)
        if not frame.empty:
            frames.append(frame)

        paging = payload.get("paging", {})
        total_page = int(paging.get("totalPage") or page)
        if not paging.get("hasNext"):
            break
        page += 1

    if not frames:
        return pd.DataFrame(columns=["Close"])
    return pd.concat(frames).sort_index()


def _fetch_yahoo_japan_fund_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Yahoo Japan에서 일본 투자신탁 기준가를 조회합니다."""
    yahoo_ticker = _normalize_japan_fund_ticker(ticker)
    url = f"{YAHOO_JAPAN_FINANCE_BASE_URL}/{yahoo_ticker}/history"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
        )
    }

    try:
        html = _fetch_text(url, headers)
        jwt_token = _extract_yahoo_japan_jwt_token(html)
        history = (
            _fetch_yahoo_japan_fund_history_via_api(
                yahoo_ticker, jwt_token, start, end
            )
            if jwt_token
            else _parse_yahoo_japan_fund_history_html(html)
        )
    except (OSError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return pd.Series(dtype=float, name=ticker)

    prices = history["Close"]
    if prices.empty:
        return pd.Series(dtype=float, name=ticker)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    prices = prices[(prices.index >= start_ts) & (prices.index < end_ts)]
    prices.name = ticker
    return prices


def _fetch_yfinance_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        prices = data["Close"]
    else:
        prices = data
    if isinstance(prices, pd.Series):
        column_name = tickers[0] if len(tickers) == 1 else prices.name
        prices = prices.to_frame(name=column_name)
    return prices


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
            ticker_list = [ticker.strip().upper() for ticker in tickers]
            japan_fund_tickers = [
                ticker
                for ticker in ticker_list
                if _is_japan_investment_trust_ticker(ticker)
            ]
            japan_fund_set = set(japan_fund_tickers)
            yfinance_tickers = [
                ticker
                for ticker in ticker_list
                if ticker not in japan_fund_set
            ]

            frames: list[pd.DataFrame] = []
            if yfinance_tickers:
                frames.append(_fetch_yfinance_prices(yfinance_tickers, start, end))

            if japan_fund_tickers:
                fund_prices = [
                    _fetch_yahoo_japan_fund_prices(ticker, start, end)
                    for ticker in japan_fund_tickers
                ]
                frames.append(pd.concat(fund_prices, axis=1))

            if not frames:
                return pd.DataFrame()

            prices = pd.concat(frames, axis=1)
            prices = prices.loc[:, ~prices.columns.duplicated()]
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
    has_prices = {
        t: t in prices.columns and not prices[t].dropna().empty
        for t in tickers
    }
    missing = [t for t, exists in has_prices.items() if not exists]
    present = [t for t, exists in has_prices.items() if exists]
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
            "일본 투자신탁은 .JP suffix를 붙인 Yahoo Japan 코드"
            "(예: 0331418A.JP, 29313233.JP)인지 확인한 뒤 "
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
