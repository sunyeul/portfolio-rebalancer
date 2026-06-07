import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services.analysis_service import AnalysisResult
from services.evaluation_service import EvaluationResult


def test_manual_portfolio_api_returns_normalized_assets():
    client = TestClient(app)

    response = client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {"ticker": "voo", "allocation": "40", "group": "core"},
                {"ticker": "qqq", "allocation": "60", "group": "core"},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert [row["ticker"] for row in payload["assets"]] == ["QQQ", "VOO"]
    assert sum(row["weight"] for row in payload["assets"]) == 1.0
    assert payload["warnings"] == []


def test_csv_portfolio_api_returns_normalized_assets():
    client = TestClient(app)

    response = client.post(
        "/api/v1/portfolio/csv",
        files={"file": ("portfolio.csv", b"ticker,allocation\nVOO,40\nQQQ,60\n")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["assets"]) == 2
    assert payload["assets"][0]["weight"] == 0.6


def test_analysis_requires_portfolio_first():
    client = TestClient(app)

    response = client.post("/api/v1/analysis/run", json={})

    assert response.status_code == 400
    assert "포트폴리오" in response.json()["detail"]


def test_analysis_api_stores_json_safe_metrics(monkeypatch):
    client = TestClient(app)
    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "VOO", "allocation": 100}]},
    )

    def fake_run_analysis(*args, **kwargs):
        prices = pd.DataFrame({"VOO": [1.0, 1.1]})
        returns = pd.DataFrame({"VOO": [0.1]})
        metrics_df = pd.DataFrame(
            {
                "ticker": ["VOO"],
                "CAGR": [float("nan")],
                "변동성": [0.2],
                "샤프": [1.0],
                "최대낙폭": [-0.1],
                "IR": [0.5],
                "베타": [1.0],
                "알파": [0.0],
                "위험기여도": [1.0],
                "수익기여도": [0.1],
                "가중치": [1.0],
                "E": [0.7],
                "E′": [0.75],
                "return_total": [0.2],
                "group": ["core"],
                "role": ["broad_etf"],
                "dca_enabled": [True],
                "thesis_status": ["intact"],
            }
        ).set_index("ticker")
        return AnalysisResult(
            prices=prices,
            returns=returns,
            returns_smooth=returns,
            weights_no_bench=pd.Series({"VOO": 1.0}),
            metrics_df=metrics_df,
            port_nav=pd.Series([1.0, 1.1]),
            bench_nav=None,
            portfolio_metrics={"cagr": float("nan"), "volatility": 0.2, "sharpe": 1.0},
            benchmark_metrics=None,
            missing_tickers=[],
        )

    monkeypatch.setattr("api.v1.analysis.run_analysis", fake_run_analysis)

    response = client.post("/api/v1/analysis/run", json={"bench": "SPY"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"][0]["cagr"] is None
    assert payload["portfolio_metrics"]["cagr"] is None


def test_evaluation_requires_analysis_first():
    client = TestClient(app)

    response = client.post("/api/v1/evaluation/run", json={})

    assert response.status_code == 400
    assert "데이터 분석" in response.json()["detail"]


def test_evaluation_api_and_download(monkeypatch):
    client = TestClient(app)
    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "VOO", "allocation": 100}]},
    )

    def fake_run_analysis(*args, **kwargs):
        prices = pd.DataFrame({"VOO": [1.0, 1.1]})
        returns = pd.DataFrame({"VOO": [0.1]})
        metrics_df = pd.DataFrame(
            {
                "ticker": ["VOO"],
                "가중치": [1.0],
                "위험기여도": [1.0],
                "E": [0.7],
                "E′": [0.75],
                "return_total": [0.2],
                "group": ["core"],
                "role": ["broad_etf"],
                "dca_enabled": [True],
                "thesis_status": ["intact"],
            }
        ).set_index("ticker")
        return AnalysisResult(
            prices=prices,
            returns=returns,
            returns_smooth=returns,
            weights_no_bench=pd.Series({"VOO": 1.0}),
            metrics_df=metrics_df,
            port_nav=pd.Series([1.0, 1.1]),
            bench_nav=None,
            portfolio_metrics={"cagr": 0.1, "volatility": 0.2, "sharpe": 1.0},
            benchmark_metrics=None,
            missing_tickers=[],
        )

    def fake_run_evaluation(*args, **kwargs):
        proposal = pd.DataFrame(
            {
                "ticker": ["VOO"],
                "현재%": [100.0],
                "목표%": [100.0],
                "갭%": [0.0],
                "E": [0.7],
                "E′": [0.75],
                "DCA강도점수": [0.75],
                "RC_Over%": [0.0],
                "RC_Target%": [100.0],
                "return_total%": [20.0],
                "group": ["core"],
                "role": ["broad_etf"],
                "dca_enabled": [True],
                "thesis_status": ["intact"],
                "risk_over": [False],
                "efficiency_good": [True],
                "히스테리시스제외": [True],
                "최소거래미만": [True],
                "실행": [False],
            }
        )
        return EvaluationResult(
            proposal_df=proposal,
            ips_action_df=pd.DataFrame([{"ticker": "VOO", "ips_action": "hold"}]),
            group_summary_df=pd.DataFrame([{"group": "core", "weight": 1.0}]),
            sell_list=pd.DataFrame(),
            buy_list=pd.DataFrame(),
            fine_tune_list=pd.DataFrame(),
            rc_violations=pd.DataFrame(),
        )

    monkeypatch.setattr("api.v1.analysis.run_analysis", fake_run_analysis)
    monkeypatch.setattr("api.v1.evaluation.run_evaluation", fake_run_evaluation)
    client.post("/api/v1/analysis/run", json={})

    response = client.post("/api/v1/evaluation/run", json={})
    assert response.status_code == 200
    assert response.json()["proposal"][0]["current_weight_pct"] == 100.0

    csv_response = client.get("/api/v1/evaluation/download-csv?type=proposal")
    assert csv_response.status_code == 200
    assert "VOO" in csv_response.text
