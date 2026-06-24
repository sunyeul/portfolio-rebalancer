import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app
from services.analysis_service import AnalysisResult


client = TestClient(app)


def _fake_analysis(asset_df, period, rf, bench, extra_benchmarks=None):
    metrics_df = pd.DataFrame(
        {
            "ticker": ["VOO"],
            "CAGR": [0.1],
            "변동성": [0.15],
            "샤프": [0.7],
            "최대낙폭": [-0.1],
            "IR": [0.2],
            "베타": [1.0],
            "알파": [0.01],
            "data_start": ["2026-01-01"],
            "data_end": ["2026-06-30"],
            "observation_count": [120],
            "missing_ratio": [0.0],
            "위험기여도": [0.1],
            "수익기여도": [0.08],
            "가중치": [1.0],
            "E": [0.8],
            "return_total": [0.1],
            "layer": ["core"],
            "category": ["core_market"],
            "dca_enabled": [True],
            "thesis_status": ["valid"],
        }
    ).set_index("ticker")
    prices = pd.DataFrame(
        {
            "VOO": [100.0, 101.0, 103.0],
            "SPY:80,QQQ:20": [1.0, 1.03, 1.06],
            "QQQ": [100.0, 110.0, 120.0],
            "CASH": [1.0, 1.0, 1.0],
        }
    )
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    return AnalysisResult(
        prices=prices,
        returns=returns,
        returns_smooth=returns,
        weights_no_bench=pd.Series({"VOO": 1.0}),
        metrics_df=metrics_df,
        port_nav=pd.Series([1.0, 1.03]),
        bench_nav=pd.Series([1.0, 1.02]),
        portfolio_metrics={"cagr": 0.1, "volatility": 0.2, "sharpe": 0.5},
        benchmark_metrics={"cagr": 0.08, "volatility": 0.15, "sharpe": 0.4},
        missing_tickers=[],
    )


def test_portfolio_analysis_and_v2_evaluation(monkeypatch):
    monkeypatch.setattr("api.v1.analysis.run_analysis", _fake_analysis)

    portfolio_response = client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 100,
                    "layer": "core",
                    "category": "core_market",
                }
            ]
        },
    )
    assert portfolio_response.status_code == 200

    analysis_response = client.post(
        "/api/v1/analysis/run",
        json={"period": 3, "rf": 0.025, "bench": "SPY"},
    )
    assert analysis_response.status_code == 200

    evaluation_response = client.post(
        "/api/v1/evaluation/run",
        json={
            "period": "3M",
            "bench": "SPY",
        },
    )
    assert evaluation_response.status_code == 200
    payload = evaluation_response.json()
    assert set(payload) >= {
        "evaluation_period",
        "layer_evaluations",
        "asset_evaluations",
        "review_queue",
        "journal_draft",
        "warnings",
        "guardrails",
    }
    assert payload["evaluation_period"]["label"] == "3M"
    assert payload["layer_evaluations"]
    assert payload["asset_evaluations"]
    assert "sharpe" not in payload["asset_evaluations"][0]["output"]
    assert "sortino" not in payload["asset_evaluations"][0]["output"]
    layer_benchmarks = {record["unit"]["name"]: record["unit"]["benchmark"] for record in payload["layer_evaluations"]}
    assert layer_benchmarks["core"] == "SPY:80,QQQ:20"
    assert layer_benchmarks["satellite"] == "QQQ"
    assert layer_benchmarks["experiment"] == "QQQ"
    benchmark_returns = {
        record["unit"]["name"]: record["output"]["benchmark_return"]
        for record in payload["layer_evaluations"]
    }
    assert benchmark_returns["core"] == pytest.approx(0.06)
    assert benchmark_returns["satellite"] == pytest.approx(0.2)
    assert benchmark_returns["experiment"] == pytest.approx(0.2)


def test_v2_csv_download(monkeypatch):
    monkeypatch.setattr("api.v1.analysis.run_analysis", _fake_analysis)
    client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 100,
                    "layer": "core",
                    "category": "core_market",
                }
            ]
        },
    )
    client.post("/api/v1/analysis/run", json={"period": 3, "rf": 0.025, "bench": "SPY"})
    client.post("/api/v1/evaluation/run", json={"period": "3M"})

    response = client.get("/api/v1/evaluation/download-csv?type=asset_evaluations")

    assert response.status_code == 200
    assert "unit" in response.text


def test_unknown_csv_type_is_rejected():
    response = client.get("/api/v1/evaluation/download-csv?type=unknown")

    assert response.status_code in {400, 404}
