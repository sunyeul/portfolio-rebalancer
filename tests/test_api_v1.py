import math

import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services.analysis_service import AnalysisResult, run_analysis
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
                "data_start": ["2026-06-01"],
                "data_end": ["2026-06-04"],
                "observation_count": [4],
                "missing_ratio": [0.0],
                "위험기여도": [1.0],
                "수익기여도": [0.1],
                "가중치": [1.0],
                "E": [0.7],
                "return_total": [0.2],
                "group": ["core"],
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


def test_analysis_api_defaults_to_realistic_rf_and_composite_benchmark(monkeypatch):
    client = TestClient(app)
    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "VOO", "allocation": 100}]},
    )
    captured = {}

    def fake_run_analysis(asset_df, period, rf, bench):
        captured.update({"period": period, "rf": rf, "bench": bench})
        metrics_df = pd.DataFrame(
            {
                "ticker": ["VOO"],
                "CAGR": [0.1],
                "변동성": [0.2],
                "샤프": [0.5],
                "최대낙폭": [-0.1],
                "IR": [0.2],
                "베타": [1.0],
                "알파": [0.0],
                "data_start": ["2026-06-01"],
                "data_end": ["2026-06-04"],
                "observation_count": [4],
                "missing_ratio": [0.0],
                "위험기여도": [1.0],
                "수익기여도": [0.1],
                "가중치": [1.0],
                "E": [0.7],
                "return_total": [0.2],
                "group": ["core"],
                "dca_enabled": [True],
                "thesis_status": ["intact"],
            }
        ).set_index("ticker")
        return AnalysisResult(
            prices=pd.DataFrame({"VOO": [1.0, 1.1]}),
            returns=pd.DataFrame({"VOO": [0.1]}),
            returns_smooth=pd.DataFrame({"VOO": [0.1]}),
            weights_no_bench=pd.Series({"VOO": 1.0}),
            metrics_df=metrics_df,
            port_nav=pd.Series([1.0, 1.1]),
            bench_nav=None,
            portfolio_metrics={"cagr": 0.1, "volatility": 0.2, "sharpe": 0.5},
            benchmark_metrics=None,
            missing_tickers=[],
        )

    monkeypatch.setattr("api.v1.analysis.run_analysis", fake_run_analysis)

    response = client.post("/api/v1/analysis/run", json={})

    assert response.status_code == 200
    assert captured == {"period": 12, "rf": 0.025, "bench": "SPY:80,QQQ:20"}


def test_run_analysis_builds_composite_benchmark(monkeypatch):
    asset_df = pd.DataFrame(
        {
            "ticker": ["VOO"],
            "allocation": [100.0],
            "weight": [1.0],
            "return_total": [None],
            "group": ["core"],
            "dca_enabled": [True],
            "thesis_status": ["intact"],
        }
    )

    def prices_for_composite(*args, **kwargs):
        return pd.DataFrame(
            {
                "VOO": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "SPY": [100.0, 101.0, 101.0, 102.0, 103.0, 104.0],
                "QQQ": [100.0, 102.0, 103.0, 104.0, 106.0, 108.0],
            },
            index=pd.date_range("2026-06-01", periods=6),
        )

    monkeypatch.setattr("services.analysis_service.fetch_prices", prices_for_composite)

    result = run_analysis(asset_df, 1, 0.025, "SPY:80,QQQ:20")

    assert "SPY:80,QQQ:20" in result.prices.columns
    assert "SPY:80,QQQ:20" in result.returns_smooth.columns
    assert result.benchmark_metrics is not None
    assert result.bench_nav is not None
    assert math.isfinite(result.benchmark_metrics["sharpe"])
    assert math.isfinite(result.metrics_df.loc["VOO", "IR"])
    assert math.isfinite(result.metrics_df.loc["VOO", "알파"])
    assert "SPY:80,QQQ:20" not in result.metrics_df.index


def test_analysis_error_identifies_tickers_and_date_range(monkeypatch):
    client = TestClient(app)
    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "BADTICKER.KS", "allocation": 100}]},
    )

    def empty_prices(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("services.analysis_service.fetch_prices", empty_prices)

    response = client.post(
        "/api/v1/analysis/run",
        json={"period": 1, "bench": "SPY"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "BADTICKER.KS" in detail
    assert "SPY" in detail
    assert "문제 티커: BADTICKER.KS, SPY" in detail
    assert "조회 기간:" in detail
    assert "티커 오타" in detail
    assert "000660.KS" in detail


def test_analysis_ignores_all_null_tickers_and_keeps_mixed_exchange_dates(
    monkeypatch,
):
    client = TestClient(app)
    client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {"ticker": "000600.KS", "allocation": 10},
                {"ticker": "005930.KS", "allocation": 45},
                {"ticker": "VOO", "allocation": 45},
            ]
        },
    )

    def mixed_exchange_prices(*args, **kwargs):
        return pd.DataFrame(
            {
                "000600.KS": [None, None, None, None],
                "005930.KS": [100.0, None, 102.0, 103.0],
                "VOO": [200.0, 201.0, None, 203.0],
                "SPY": [300.0, 301.0, None, 303.0],
            },
            index=pd.to_datetime(
                ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]
            ),
        )

    monkeypatch.setattr("services.analysis_service.fetch_prices", mixed_exchange_prices)

    response = client.post(
        "/api/v1/analysis/run",
        json={"period": 1, "bench": "SPY"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["missing_tickers"] == ["000600.KS"]
    assert [row["ticker"] for row in payload["metrics"]] == ["005930.KS", "VOO"]
    samsung = payload["metrics"][0]
    assert samsung["data_start"] == "2026-06-01"
    assert samsung["data_end"] == "2026-06-04"
    assert samsung["observation_count"] == 3
    assert samsung["missing_ratio"] == 0.25


def test_analysis_keeps_spy_weight_when_spy_is_portfolio_asset(monkeypatch):
    client = TestClient(app)
    client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {"ticker": "SPY", "allocation": 40},
                {"ticker": "QQQ", "allocation": 60},
            ]
        },
    )

    def prices_with_spy_asset(*args, **kwargs):
        return pd.DataFrame(
            {
                "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "QQQ": [200.0, 202.0, 204.0, 206.0, 208.0, 210.0],
            },
            index=pd.date_range("2026-06-01", periods=6),
        )

    monkeypatch.setattr("services.analysis_service.fetch_prices", prices_with_spy_asset)

    response = client.post(
        "/api/v1/analysis/run",
        json={"period": 1, "bench": "SPY"},
    )

    assert response.status_code == 200
    metrics_by_ticker = {row["ticker"]: row for row in response.json()["metrics"]}
    assert metrics_by_ticker["SPY"]["weight"] == 0.4
    assert metrics_by_ticker["QQQ"]["weight"] == 0.6


def test_evaluation_requires_analysis_first():
    client = TestClient(app)

    response = client.post("/api/v1/evaluation/run", json={})

    assert response.status_code == 400
    assert "데이터 분석" in response.json()["detail"]


def test_simulation_endpoints_require_analysis_first():
    client = TestClient(app)

    counterfactual = client.post("/api/v1/simulation/counterfactual", json={})
    backtest = client.post("/api/v1/simulation/backtest", json={})

    assert counterfactual.status_code == 400
    assert backtest.status_code == 400
    assert "데이터 분석" in counterfactual.json()["detail"]
    assert "데이터 분석" in backtest.json()["detail"]


def test_simulation_counterfactual_and_backtest_api_return_json_safe_payloads(monkeypatch):
    client = TestClient(app)
    client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 55,
                    "group": "core",
                    "thesis_status": "intact",
                },
                {
                    "ticker": "QQQ",
                    "allocation": 15,
                    "group": "core",
                    "thesis_status": "intact",
                },
                {
                    "ticker": "UFO",
                    "allocation": 30,
                    "group": "satellite",
                    "thesis_status": "watch",
                },
            ]
        },
    )

    def fake_run_analysis(*args, **kwargs):
        index = pd.date_range("2025-01-31", periods=8, freq="ME")
        prices = pd.DataFrame(
            {
                "VOO": [100, 102, 101, 103, 104, 105, 106, 107],
                "QQQ": [100, 103, 101, 104, 106, 107, 108, 110],
                "UFO": [100, 108, 102, 106, 109, 107, 112, 115],
            },
            index=index,
        )
        returns = prices.pct_change(fill_method=None).dropna()
        metrics_df = pd.DataFrame(
            {
                "ticker": ["VOO", "QQQ", "UFO"],
                "가중치": [0.55, 0.15, 0.30],
                "위험기여도": [0.35, 0.15, 0.50],
                "E": [0.8, 0.7, 0.65],
                "return_total": [0.1, 0.08, 0.2],
                "group": ["core", "core", "satellite"],
                "dca_enabled": [True, True, True],
                "thesis_status": ["intact", "intact", "watch"],
                "missing_ratio": [0.0, 0.0, 0.0],
                "observation_count": [120, 120, 120],
            }
        ).set_index("ticker")
        return AnalysisResult(
            prices=prices,
            returns=returns,
            returns_smooth=returns,
            weights_no_bench=pd.Series({"VOO": 0.55, "QQQ": 0.15, "UFO": 0.30}),
            metrics_df=metrics_df,
            port_nav=pd.Series([1.0, 1.1]),
            bench_nav=None,
            portfolio_metrics={"cagr": 0.1, "volatility": 0.2, "sharpe": 1.0},
            benchmark_metrics=None,
            missing_tickers=[],
        )

    monkeypatch.setattr("api.v1.analysis.run_analysis", fake_run_analysis)
    assert client.post("/api/v1/analysis/run", json={}).status_code == 200

    counterfactual_response = client.post(
        "/api/v1/simulation/counterfactual",
        json={"scenario": "pause_satellite_new_buys"},
    )
    assert counterfactual_response.status_code == 200
    counterfactual_payload = counterfactual_response.json()
    assert counterfactual_payload["baseline"]["weights"]
    assert counterfactual_payload["deltas"]["assets"]
    assert isinstance(counterfactual_payload["warnings"], list)

    backtest_response = client.post(
        "/api/v1/simulation/backtest",
        json={"strategies": ["current_ips", "core_first_dca"], "frequency": "monthly"},
    )
    assert backtest_response.status_code == 200
    backtest_payload = backtest_response.json()
    assert len(backtest_payload["strategy_summaries"]) == 2
    assert backtest_payload["ips_fit_summary"]
    assert backtest_payload["performance_summary"]
    assert backtest_payload["strategy_summaries"][0]["strategy_label"] in {
        "현재 IPS 유지",
        "코어 부족분 우선",
    }
    assert all(row["cagr"] is not None for row in backtest_payload["strategy_summaries"])


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
                "return_total": [0.2],
                "group": ["core"],
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

    captured_evaluation_kwargs = {}

    def fake_run_evaluation(*args, **kwargs):
        captured_evaluation_kwargs.update(kwargs)
        proposal = pd.DataFrame(
            {
                "ticker": ["VOO"],
                "현재%": [100.0],
                "목표%": [100.0],
                "갭%": [0.0],
                "E": [0.7],
                "RC_Gap%": [0.0],
                "RC_Over%": [0.0],
                "RC_Target%": [100.0],
                "return_total%": [20.0],
                "group": ["core"],
                "dca_enabled": [True],
                "thesis_status": ["intact"],
                "risk_over": [False],
                "efficiency_warning": [False],
                "IPS적합도": [82.0],
                "IPS등급": ["high"],
                "히스테리시스제외": [True],
                "최소거래미만": [True],
                "실행": [False],
                "제안조정%": [0.0],
                "판단사유": ["히스테리시스 범위 및 최소 거래 미만"],
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

    response = client.post(
        "/api/v1/evaluation/run",
        json={"decision_context": "market_correction"},
    )
    assert response.status_code == 200
    assert captured_evaluation_kwargs["decision_context"] == "market_correction"
    proposal_row = response.json()["proposal"][0]
    assert proposal_row["current_weight_pct"] == 100.0
    assert proposal_row["rc_gap_pct"] == 0.0
    assert proposal_row["suggested_trade_pct"] == 0.0
    assert proposal_row["action_reason"] == "히스테리시스 범위 및 최소 거래 미만"
    assert "adjusted_gap_pct" not in proposal_row

    csv_response = client.get("/api/v1/evaluation/download-csv?type=proposal")
    assert csv_response.status_code == 200
    assert "VOO" in csv_response.text
    assert "suggested_trade_pct" in csv_response.text


def test_analysis_rerun_clears_stale_evaluation_outputs(monkeypatch):
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
                "return_total": [0.2],
                "group": ["core"],
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

    captured_evaluation_kwargs = {}

    def fake_run_evaluation(*args, **kwargs):
        captured_evaluation_kwargs.update(kwargs)
        proposal = pd.DataFrame(
            {
                "ticker": ["VOO"],
                "현재%": [100.0],
                "목표%": [100.0],
                "갭%": [0.0],
                "E": [0.7],
                "RC_Gap%": [0.0],
                "RC_Over%": [0.0],
                "RC_Target%": [100.0],
                "return_total%": [20.0],
                "group": ["core"],
                "dca_enabled": [True],
                "thesis_status": ["intact"],
                "risk_over": [False],
                "efficiency_warning": [False],
                "IPS적합도": [82.0],
                "IPS등급": ["high"],
                "히스테리시스제외": [True],
                "최소거래미만": [True],
                "실행": [False],
                "제안조정%": [0.0],
                "판단사유": ["기존 평가"],
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

    assert client.post("/api/v1/analysis/run", json={"period": 12}).status_code == 200
    assert client.post("/api/v1/evaluation/run", json={}).status_code == 200
    assert captured_evaluation_kwargs["decision_context"] == "regular_review"
    assert client.get("/api/v1/evaluation/download-csv?type=proposal").status_code == 200

    assert client.post("/api/v1/analysis/run", json={"period": 6}).status_code == 200
    response = client.get("/api/v1/evaluation/download-csv?type=proposal")

    assert response.status_code == 404
    assert "다운로드할 결과가 없습니다" in response.json()["detail"]
