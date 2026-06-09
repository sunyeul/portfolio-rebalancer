from urllib.error import URLError

import pandas as pd

from utils import data_fetcher


YAHOO_JAPAN_HISTORY_HTML = """
<html>
  <body>
    <table>
      <thead>
        <tr>
          <th>日付</th>
          <th>基準価額</th>
          <th>前日差</th>
          <th>純資産（百万）</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>2026年6月5日</td>
          <td>38,069</td>
          <td>+81</td>
          <td>12,455,413</td>
        </tr>
        <tr>
          <td>2026年6月4日</td>
          <td>37,988</td>
          <td>-289</td>
          <td>12,410,713</td>
        </tr>
      </tbody>
    </table>
  </body>
</html>
"""

YAHOO_JAPAN_HISTORY_PAGE_HTML = """
<html>
  <body>
    <script>
      window.__PRELOADED_STATE__ = {"pageInfo":{"jwtToken":"test-jwt-token"}};
    </script>
  </body>
</html>
"""

YAHOO_JAPAN_HISTORY_JSON = """
{
  "paging": {
    "hasNext": false,
    "totalPage": 1,
    "totalSize": 2,
    "page": 1,
    "size": 100
  },
  "histories": [
    {
      "date": "2026年6月5日",
      "price": "38,069",
      "priceChange": "81",
      "netAssetsBalance": "12,455,413"
    },
    {
      "date": "2026年6月4日",
      "price": "37,988",
      "priceChange": "-289",
      "netAssetsBalance": "12,410,713"
    }
  ]
}
"""


class FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def test_parse_yahoo_japan_fund_history_html_reads_nav_series():
    frame = data_fetcher._parse_yahoo_japan_fund_history_html(
        YAHOO_JAPAN_HISTORY_HTML
    )

    assert list(frame["Close"]) == [37988.0, 38069.0]
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2026-06-04", "2026-06-05"]
    assert list(frame["NetAssetsMillions"]) == [12410713.0, 12455413.0]


def test_parse_yahoo_japan_fund_history_json_reads_nav_series():
    frame = data_fetcher._parse_yahoo_japan_fund_history_json(
        {
            "histories": [
                {
                    "date": "2026年6月5日",
                    "price": "38,069",
                    "netAssetsBalance": "12,455,413",
                }
            ]
        }
    )

    assert list(frame["Close"]) == [38069.0]
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2026-06-05"]
    assert list(frame["NetAssetsMillions"]) == [12455413.0]


def test_fetch_prices_uses_yahoo_japan_for_explicit_jp_investment_trusts(monkeypatch):
    data_fetcher.fetch_prices.cache_clear()
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        if "/bff-pc/" in request.full_url:
            assert request.headers["Jwt-token"] == "test-jwt-token"
            return FakeResponse(YAHOO_JAPAN_HISTORY_JSON)
        return FakeResponse(YAHOO_JAPAN_HISTORY_PAGE_HTML)

    def fail_yfinance_download(*args, **kwargs):
        raise AssertionError("yfinance should not be called for JP fund codes")

    monkeypatch.setattr(data_fetcher, "urlopen", fake_urlopen)
    monkeypatch.setattr(data_fetcher.yf, "download", fail_yfinance_download)

    prices = data_fetcher.fetch_prices(("0331418A.JP", "29313233.JP"), "2026-06-01", "2026-06-09")

    assert "https://finance.yahoo.co.jp/quote/0331418A/history" in requested_urls
    assert "https://finance.yahoo.co.jp/quote/29313233/history" in requested_urls
    assert any("/bff-pc/v1/main/fund/price/history/0331418A?" in url for url in requested_urls)
    assert any("/bff-pc/v1/main/fund/price/history/29313233?" in url for url in requested_urls)
    assert list(prices.columns) == ["0331418A.JP", "29313233.JP"]
    assert prices.loc[pd.Timestamp("2026-06-05"), "0331418A.JP"] == 38069.0


def test_fetch_prices_zero_pads_explicit_seven_digit_japan_fund_codes(monkeypatch):
    data_fetcher.fetch_prices.cache_clear()
    requested_urls: list[str] = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        if "/bff-pc/" in request.full_url:
            return FakeResponse(YAHOO_JAPAN_HISTORY_JSON)
        return FakeResponse(YAHOO_JAPAN_HISTORY_PAGE_HTML)

    monkeypatch.setattr(data_fetcher, "urlopen", fake_urlopen)

    prices = data_fetcher.fetch_prices(("3311187.JP",), "2026-06-01", "2026-06-09")

    assert "https://finance.yahoo.co.jp/quote/03311187/history" in requested_urls
    assert any("/bff-pc/v1/main/fund/price/history/03311187?" in url for url in requested_urls)
    assert list(prices.columns) == ["3311187.JP"]


def test_fetch_prices_combines_yfinance_and_yahoo_japan(monkeypatch):
    data_fetcher.fetch_prices.cache_clear()

    def fake_yfinance_download(tickers, start, end, auto_adjust, progress):
        assert tickers == ["VOO"]
        index = pd.to_datetime(["2026-06-04", "2026-06-05"])
        return pd.DataFrame({"VOO": [100.0, 101.0]}, index=index)

    monkeypatch.setattr(data_fetcher.yf, "download", fake_yfinance_download)
    monkeypatch.setattr(
        data_fetcher,
        "urlopen",
        lambda request, timeout: FakeResponse(
            YAHOO_JAPAN_HISTORY_JSON
            if "/bff-pc/" in request.full_url
            else YAHOO_JAPAN_HISTORY_PAGE_HTML
        ),
    )

    prices = data_fetcher.fetch_prices(("VOO", "0331418A.JP"), "2026-06-01", "2026-06-09")

    assert list(prices.columns) == ["VOO", "0331418A.JP"]
    assert prices.loc[pd.Timestamp("2026-06-05"), "VOO"] == 101.0
    assert prices.loc[pd.Timestamp("2026-06-05"), "0331418A.JP"] == 38069.0


def test_suffixless_japan_fund_shaped_symbols_stay_on_yfinance(monkeypatch):
    data_fetcher.fetch_prices.cache_clear()

    def fake_yfinance_download(tickers, start, end, auto_adjust, progress):
        assert tickers == ["0331418A", "29313233"]
        index = pd.to_datetime(["2026-06-05"])
        return pd.DataFrame({"0331418A": [42.0], "29313233": [43.0]}, index=index)

    def fail_urlopen(*args, **kwargs):
        raise AssertionError("Yahoo Japan should not be called")

    monkeypatch.setattr(data_fetcher.yf, "download", fake_yfinance_download)
    monkeypatch.setattr(data_fetcher, "urlopen", fail_urlopen)

    prices = data_fetcher.fetch_prices(("0331418A", "29313233"), "2026-06-01", "2026-06-09")

    assert list(prices.columns) == ["0331418A", "29313233"]
    assert prices.loc[pd.Timestamp("2026-06-05"), "0331418A"] == 42.0


def test_fetch_prices_drops_japan_fund_when_yahoo_japan_fails(monkeypatch):
    data_fetcher.fetch_prices.cache_clear()

    def fake_urlopen(request, timeout):
        raise URLError("network unavailable")

    monkeypatch.setattr(data_fetcher, "urlopen", fake_urlopen)

    prices = data_fetcher.fetch_prices(("0331418A.JP",), "2026-06-01", "2026-06-09")

    assert prices.empty
