import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.evaluation import EvaluationPeriod
from main import app
from services.analysis_service import AnalysisResult
from storage.database import initialize_database


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
    monkeypatch.setattr("api.v1.evaluation.run_analysis", _fake_analysis)

    portfolio_response = client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 100,
                    "layer": "core",
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


def test_analysis_run_uses_as_of_date(monkeypatch):
    captured: dict[str, EvaluationPeriod] = {}

    def fake_analysis(asset_df, period, rf, bench, extra_benchmarks=None):
        captured["period"] = period
        return _fake_analysis(asset_df, period, rf, bench, extra_benchmarks)

    monkeypatch.setattr("api.v1.analysis.run_analysis", fake_analysis)

    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "VOO", "allocation": 100, "layer": "core"}]},
    )

    response = client.post(
        "/api/v1/analysis/run",
        json={"period": 3, "as_of_date": "2026-06-15", "rf": 0.025, "bench": "SPY"},
    )

    assert response.status_code == 200
    assert captured["period"].label == "3M"
    assert captured["period"].start_date.isoformat() == "2026-03-15"
    assert captured["period"].end_date.isoformat() == "2026-06-15"


def test_v2_csv_download(monkeypatch):
    monkeypatch.setattr("api.v1.analysis.run_analysis", _fake_analysis)
    monkeypatch.setattr("api.v1.evaluation.run_analysis", _fake_analysis)
    client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 100,
                    "layer": "core",
                }
            ]
        },
    )
    client.post("/api/v1/analysis/run", json={"period": 3, "rf": 0.025, "bench": "SPY"})
    client.post("/api/v1/evaluation/run", json={"period": "3M"})

    response = client.get("/api/v1/evaluation/download-csv?type=asset_evaluations")

    assert response.status_code == 200
    assert "unit" in response.text


def test_evaluation_run_reanalyzes_from_as_of_date(monkeypatch):
    captured: dict[str, EvaluationPeriod] = {}

    def fake_analysis(asset_df, period, rf, bench, extra_benchmarks=None):
        captured["period"] = period
        return _fake_analysis(asset_df, period, rf, bench, extra_benchmarks)

    monkeypatch.setattr("api.v1.analysis.run_analysis", _fake_analysis)
    monkeypatch.setattr("api.v1.evaluation.run_analysis", fake_analysis)

    client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 100,
                    "layer": "core",
                }
            ]
        },
    )
    client.post("/api/v1/analysis/run", json={"period": 12, "rf": 0.025, "bench": "SPY"})

    response = client.post(
        "/api/v1/evaluation/run",
        json={"period": "3M", "as_of_date": "2026-06-15", "bench": "SPY"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["evaluation_period"]["start_date"] == "2026-03-15"
    assert payload["evaluation_period"]["end_date"] == "2026-06-15"
    assert captured["period"].start_date.isoformat() == "2026-03-15"
    assert captured["period"].end_date.isoformat() == "2026-06-15"


def test_evaluation_run_inherits_numeric_analysis_period(monkeypatch):
    captured: dict[str, EvaluationPeriod] = {}

    def fake_analysis(asset_df, period, rf, bench, extra_benchmarks=None):
        captured["period"] = period
        return _fake_analysis(asset_df, period, rf, bench, extra_benchmarks)

    monkeypatch.setattr("api.v1.analysis.run_analysis", _fake_analysis)
    monkeypatch.setattr("api.v1.evaluation.run_analysis", fake_analysis)

    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "VOO", "allocation": 100, "layer": "core"}]},
    )
    client.post("/api/v1/analysis/run", json={"period": 12, "rf": 0.025, "bench": "SPY"})

    response = client.post(
        "/api/v1/evaluation/run",
        json={"as_of_date": "2026-06-15", "bench": "SPY"},
    )

    assert response.status_code == 200
    assert response.json()["evaluation_period"]["label"] == "1Y"
    assert captured["period"].start_date.isoformat() == "2025-06-15"
    assert captured["period"].end_date.isoformat() == "2026-06-15"


def test_unknown_csv_type_is_rejected():
    response = client.get("/api/v1/evaluation/download-csv?type=unknown")

    assert response.status_code in {400, 404}


def test_saved_snapshot_evaluation_runs_are_persisted_and_loaded(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )
    initialize_database()
    monkeypatch.setattr("api.v1.evaluation.run_analysis", _fake_analysis)

    portfolio_response = client.post(
        "/api/v1/portfolios",
        json={"name": "Saved evaluation account"},
    )
    assert portfolio_response.status_code == 200
    portfolio_id = portfolio_response.json()["portfolio"]["id"]

    snapshot_response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/snapshots",
        json={
            "name": "June holdings",
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 100,
                    "layer": "core",
                    "thesis_status": "valid",
                }
            ],
        },
    )
    assert snapshot_response.status_code == 200
    snapshot_id = snapshot_response.json()["snapshot"]["id"]

    first_response = client.post(
        f"/api/v1/portfolios/snapshots/{snapshot_id}/evaluations/run",
        json={"period": "3M", "bench": "SPY"},
    )
    assert first_response.status_code == 200
    first_payload = first_response.json()
    assert first_payload["evaluation_run"] is None
    assert first_payload["evaluation"]["evaluation_period"]["label"] == "3M"

    first_save_response = client.post(
        f"/api/v1/portfolios/snapshots/{snapshot_id}/evaluations"
    )
    assert first_save_response.status_code == 200
    first_run = first_save_response.json()["evaluation_run"]
    assert first_run["status"] == "active"
    assert first_run["is_stale"] is False

    second_response = client.post(
        f"/api/v1/portfolios/snapshots/{snapshot_id}/evaluations/run",
        json={"period": "1Y", "bench": "SPY"},
    )
    assert second_response.status_code == 200
    assert second_response.json()["evaluation_run"] is None

    second_save_response = client.post(
        f"/api/v1/portfolios/snapshots/{snapshot_id}/evaluations"
    )
    assert second_save_response.status_code == 200
    second_run = second_save_response.json()["evaluation_run"]

    list_response = client.get(
        f"/api/v1/portfolios/snapshots/{snapshot_id}/evaluations"
    )
    assert list_response.status_code == 200
    runs = list_response.json()["evaluation_runs"]
    assert [run["status"] for run in runs] == ["active", "superseded"]
    assert runs[0]["id"] == second_run["id"]
    assert runs[1]["id"] == first_run["id"]
    assert runs[1]["superseded_by_run_id"] == second_run["id"]

    load_response = client.post(f"/api/v1/portfolios/snapshots/{snapshot_id}/load")
    assert load_response.status_code == 200
    loaded = load_response.json()
    assert loaded["evaluation_run"]["id"] == second_run["id"]
    assert loaded["evaluation_run"]["settings"]["period"] == "1Y"
    assert loaded["evaluation_run"]["settings"]["bench"] == "SPY"
    assert loaded["evaluation"]["evaluation_period"]["label"] == "1Y"

    csv_response = client.get("/api/v1/evaluation/download-csv?type=asset_evaluations")
    assert csv_response.status_code == 200
    assert "unit" in csv_response.text


def test_saved_snapshot_patch_rejects_position_rows(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )
    initialize_database()

    portfolio_response = client.post(
        "/api/v1/portfolios",
        json={"name": "Immutable snapshot API account"},
    )
    portfolio_id = portfolio_response.json()["portfolio"]["id"]
    snapshot_response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/snapshots",
        json={
            "name": "Holdings",
            "rows": [{"ticker": "VOO", "allocation": 100, "layer": "core"}],
        },
    )
    snapshot_id = snapshot_response.json()["snapshot"]["id"]

    response = client.patch(
        f"/api/v1/portfolios/snapshots/{snapshot_id}",
        json={
            "name": "Changed",
            "rows": [{"ticker": "QQQ", "allocation": 100, "layer": "core"}],
        },
    )

    assert response.status_code == 400
    assert "새 보유현황 스냅샷" in response.json()["detail"]
