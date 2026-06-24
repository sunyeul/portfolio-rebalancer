import pandas as pd

from services.analysis_service import run_analysis


def test_run_analysis_fetches_extra_layer_benchmark_components(monkeypatch):
    captured: dict[str, tuple[str, ...]] = {}

    def fake_fetch_prices(tickers, start, end):
        captured["tickers"] = tickers
        index = pd.date_range("2026-01-01", periods=8, freq="D")
        return pd.DataFrame(
            {
                "VOO": [100, 101, 102, 103, 104, 105, 106, 107],
                "SPY": [100, 101, 102, 103, 104, 105, 106, 107],
                "QQQ": [100, 102, 104, 106, 108, 110, 112, 114],
            },
            index=index,
            dtype=float,
        )

    monkeypatch.setattr("services.analysis_service.fetch_prices", fake_fetch_prices)
    asset_df = pd.DataFrame(
        {
            "ticker": ["VOO"],
            "allocation": [100.0],
            "weight": [1.0],
            "return_total": [None],
            "layer": ["core"],
            "category": ["core_market"],
            "dca_enabled": [True],
            "thesis_status": ["valid"],
        }
    )

    result = run_analysis(
        asset_df,
        1,
        0.025,
        "SPY:80,QQQ:20",
        extra_benchmarks=["QQQ", "CASH"],
    )

    assert captured["tickers"] == ("VOO", "SPY", "QQQ")
    assert "SPY:80,QQQ:20" in result.prices.columns
    assert "QQQ" in result.prices.columns
    assert "CASH" in result.prices.columns
    assert result.prices["CASH"].eq(1.0).all()
