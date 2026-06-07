import pandas as pd
from fastapi.testclient import TestClient

from main import app
from services.analysis_service import AnalysisResult
from services.evaluation_service import EvaluationResult
from storage.database import initialize_database
from storage.portfolio_store import save_current_state as persist_current_state


def _client_with_db(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )
    initialize_database()
    return TestClient(app)


def test_database_initialization_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )

    initialize_database()
    initialize_database()


def test_portfolio_crud_and_input_only_snapshot(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    create_response = client.post(
        "/api/v1/portfolios",
        json={"name": "장기 투자", "description": "개인 계좌"},
    )
    assert create_response.status_code == 200
    portfolio_id = create_response.json()["portfolio"]["id"]

    update_response = client.patch(
        f"/api/v1/portfolios/{portfolio_id}",
        json={"name": "장기 투자 계좌"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["portfolio"]["name"] == "장기 투자 계좌"

    client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {"ticker": "voo", "allocation": 70, "group": "core"},
                {"ticker": "ufo", "allocation": 30, "group": "satellite"},
            ]
        },
    )
    snapshot_response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/snapshots",
        json={"name": "2026-06 점검", "note": "입력 저장"},
    )
    assert snapshot_response.status_code == 200
    snapshot_id = snapshot_response.json()["snapshot"]["id"]

    list_response = client.get(f"/api/v1/portfolios/{portfolio_id}/snapshots")
    assert list_response.status_code == 200
    assert list_response.json()["snapshots"][0]["position_count"] == 2
    assert list_response.json()["snapshots"][0]["has_analysis"] is False

    load_client = TestClient(app)
    load_response = load_client.post(f"/api/v1/portfolios/snapshots/{snapshot_id}/load")
    assert load_response.status_code == 200
    payload = load_response.json()
    assert [row["ticker"] for row in payload["portfolio"]["assets"]] == ["UFO", "VOO"]
    assert [row["group"] for row in payload["portfolio"]["assets"]] == [
        "satellite",
        "core",
    ]
    assert payload["analysis"] is None


def test_snapshot_can_save_latest_rows_with_fixed_group(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)
    portfolio_id = client.post(
        "/api/v1/portfolios",
        json={"name": "장기 투자"},
    ).json()["portfolio"]["id"]

    snapshot_response = client.post(
        f"/api/v1/portfolios/{portfolio_id}/snapshots",
        json={
            "name": "입력 즉시 저장",
            "rows": [
                {"ticker": "VOO", "allocation": 70, "group": "core"},
                {"ticker": "UFO", "allocation": 30, "group": "satellite"},
            ],
        },
    )
    assert snapshot_response.status_code == 200
    snapshot_id = snapshot_response.json()["snapshot"]["id"]

    load_response = client.post(f"/api/v1/portfolios/snapshots/{snapshot_id}/load")
    assert load_response.status_code == 200
    payload = load_response.json()
    assert [row["ticker"] for row in payload["portfolio"]["assets"]] == ["UFO", "VOO"]
    assert [row["group"] for row in payload["portfolio"]["assets"]] == [
        "satellite",
        "core",
    ]
    assert payload["analysis"] is None


def test_current_state_auto_save_does_not_create_snapshot(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)
    portfolio_id = client.post(
        "/api/v1/portfolios",
        json={"name": "자동 저장 계좌"},
    ).json()["portfolio"]["id"]
    client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {"ticker": "VOO", "allocation": 70, "group": "core"},
                {"ticker": "UFO", "allocation": 30, "group": "satellite"},
            ]
        },
    )

    current_response = client.post(f"/api/v1/portfolios/{portfolio_id}/current-state")
    assert current_response.status_code == 200
    assert [row["ticker"] for row in current_response.json()["portfolio"]["assets"]] == [
        "UFO",
        "VOO",
    ]

    snapshots_response = client.get(f"/api/v1/portfolios/{portfolio_id}/snapshots")
    assert snapshots_response.status_code == 200
    assert snapshots_response.json()["snapshots"] == []

    load_client = TestClient(app)
    load_response = load_client.get(f"/api/v1/portfolios/{portfolio_id}/current-state")
    assert load_response.status_code == 200
    payload = load_response.json()
    assert [row["group"] for row in payload["portfolio"]["assets"]] == [
        "satellite",
        "core",
    ]

    snapshot_response = load_client.post(
        f"/api/v1/portfolios/{portfolio_id}/snapshots",
        json={"name": "자동 저장본에서 스냅샷"},
    )
    assert snapshot_response.status_code == 200
    assert snapshot_response.json()["snapshot"]["position_count"] == 2


def test_current_state_serializes_legacy_evaluation_defaults(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)
    portfolio_id = client.post(
        "/api/v1/portfolios",
        json={"name": "레거시 평가 계좌"},
    ).json()["portfolio"]["id"]
    persist_current_state(
        portfolio_id,
        {
            "asset_df": [
                {
                    "ticker": "VOO",
                    "allocation": 100.0,
                    "return_total": None,
                    "group": "core",
                    "dca_enabled": True,
                    "thesis_status": "intact",
                    "weight": 1.0,
                }
            ],
            "proposal_df": [
                {
                    "ticker": "VOO",
                    "현재%": 100.0,
                    "목표%": 100.0,
                    "갭%": 0.0,
                    "E": 0.7,
                    "RC_Over%": 0.0,
                    "RC_Target%": 100.0,
                    "return_total%": 20.0,
                    "group": "core",
                    "dca_enabled": True,
                    "thesis_status": "intact",
                    "risk_over": False,
                    "efficiency_good": True,
                    "히스테리시스제외": None,
                    "최소거래미만": None,
                    "실행": None,
                }
            ],
            "ips_action_df": [],
            "group_summary_df": [],
            "rc_violations": [],
        },
    )

    response = client.get(f"/api/v1/portfolios/{portfolio_id}/current-state")
    assert response.status_code == 200
    proposal_row = response.json()["evaluation"]["proposal"][0]
    assert proposal_row["rc_gap_pct"] == 0.0
    assert proposal_row["suggested_trade_pct"] == 0.0
    assert proposal_row["action_reason"] == ""
    assert proposal_row["within_hysteresis"] is False
    assert proposal_row["below_min_trade"] is False
    assert proposal_row["should_execute"] is False


def test_snapshot_metadata_update_and_delete(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)
    portfolio_id = client.post(
        "/api/v1/portfolios",
        json={"name": "장기 투자"},
    ).json()["portfolio"]["id"]
    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "VOO", "allocation": 100, "group": "core"}]},
    )
    snapshot_id = client.post(
        f"/api/v1/portfolios/{portfolio_id}/snapshots",
        json={"name": "초안", "note": "처음 저장"},
    ).json()["snapshot"]["id"]

    update_response = client.patch(
        f"/api/v1/portfolios/snapshots/{snapshot_id}",
        json={"name": "월간 점검", "note": "리밸런싱 전"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["snapshot"]["name"] == "월간 점검"
    assert update_response.json()["snapshot"]["note"] == "리밸런싱 전"

    empty_name_response = client.patch(
        f"/api/v1/portfolios/snapshots/{snapshot_id}",
        json={"name": "   "},
    )
    assert empty_name_response.status_code == 200
    assert empty_name_response.json()["snapshot"]["name"] == "저장된 스냅샷"

    list_response = client.get(f"/api/v1/portfolios/{portfolio_id}/snapshots")
    assert list_response.status_code == 200
    assert list_response.json()["snapshots"][0]["name"] == "저장된 스냅샷"
    assert list_response.json()["snapshots"][0]["note"] == "리밸런싱 전"

    missing_update = client.patch(
        "/api/v1/portfolios/snapshots/999999",
        json={"name": "없음"},
    )
    assert missing_update.status_code == 404
    assert missing_update.json()["detail"] == "스냅샷을 찾을 수 없습니다."

    delete_response = client.delete(f"/api/v1/portfolios/snapshots/{snapshot_id}")
    assert delete_response.status_code == 200
    assert delete_response.json() == {"ok": True}
    assert client.get(f"/api/v1/portfolios/{portfolio_id}/snapshots").json()["snapshots"] == []

    load_response = client.post(f"/api/v1/portfolios/snapshots/{snapshot_id}/load")
    assert load_response.status_code == 404

    missing_delete = client.delete("/api/v1/portfolios/snapshots/999999")
    assert missing_delete.status_code == 404
    assert missing_delete.json()["detail"] == "스냅샷을 찾을 수 없습니다."


def test_snapshot_update_persists_positions_and_clears_analysis(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)
    portfolio_id = client.post(
        "/api/v1/portfolios",
        json={"name": "장기 투자"},
    ).json()["portfolio"]["id"]
    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "VOO", "allocation": 100, "group": "core"}]},
    )

    def fake_run_analysis(*args, **kwargs):
        returns = pd.DataFrame({"VOO": [0.1]})
        metrics_df = pd.DataFrame(
            {
                "ticker": ["VOO"],
                "CAGR": [0.1],
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
            prices=pd.DataFrame({"VOO": [1.0, 1.1]}),
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

    monkeypatch.setattr("api.v1.analysis.run_analysis", fake_run_analysis)
    assert client.post("/api/v1/analysis/run", json={"bench": "SPY"}).status_code == 200
    snapshot_id = client.post(
        f"/api/v1/portfolios/{portfolio_id}/snapshots",
        json={"name": "수정 전"},
    ).json()["snapshot"]["id"]

    update_response = client.patch(
        f"/api/v1/portfolios/snapshots/{snapshot_id}",
        json={
            "name": "수정 후",
            "note": "상태 변경",
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 70,
                    "group": "satellite",
                    "dca_enabled": False,
                    "thesis_status": "watch",
                },
                {
                    "ticker": "QQQ",
                    "allocation": 30,
                    "group": "core",
                    "dca_enabled": True,
                    "thesis_status": "intact",
                },
            ],
        },
    )
    assert update_response.status_code == 200
    assert update_response.json()["snapshot"]["name"] == "수정 후"
    assert update_response.json()["snapshot"]["has_analysis"] is False

    load_response = client.post(f"/api/v1/portfolios/snapshots/{snapshot_id}/load")
    assert load_response.status_code == 200
    payload = load_response.json()
    assert payload["analysis"] is None
    assert payload["snapshot"]["position_count"] == 2
    assert payload["portfolio"]["assets"] == [
        {
            "ticker": "QQQ",
            "allocation": 30.0,
            "return_total": None,
            "group": "core",
            "dca_enabled": True,
            "thesis_status": "intact",
            "weight": 0.3,
        },
        {
            "ticker": "VOO",
            "allocation": 70.0,
            "return_total": None,
            "group": "satellite",
            "dca_enabled": False,
            "thesis_status": "watch",
            "weight": 0.7,
        },
    ]


def test_snapshot_persists_analysis_and_evaluation(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)
    portfolio_id = client.post(
        "/api/v1/portfolios",
        json={"name": "운용 계좌"},
    ).json()["portfolio"]["id"]
    client.post(
        "/api/v1/portfolio/manual",
        json={"rows": [{"ticker": "VOO", "allocation": 100, "group": "core"}]},
    )

    def fake_run_analysis(*args, **kwargs):
        returns = pd.DataFrame({"VOO": [0.1]})
        metrics_df = pd.DataFrame(
            {
                "ticker": ["VOO"],
                "CAGR": [0.1],
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
            prices=pd.DataFrame({"VOO": [1.0, 1.1]}),
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
                "RC_Gap%": [0.0],
                "RC_Over%": [0.0],
                "RC_Target%": [100.0],
                "return_total%": [20.0],
                "group": ["core"],
                "dca_enabled": [True],
                "thesis_status": ["intact"],
                "risk_over": [False],
                "efficiency_good": [True],
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
            ips_config_snapshot={
                "target_allocation": {"core": {"target": 0.8}},
            },
        )

    monkeypatch.setattr("api.v1.analysis.run_analysis", fake_run_analysis)
    monkeypatch.setattr("api.v1.evaluation.run_evaluation", fake_run_evaluation)

    assert client.post("/api/v1/analysis/run", json={"bench": "SPY"}).status_code == 200
    assert client.post("/api/v1/evaluation/run", json={}).status_code == 200
    snapshot_id = client.post(
        f"/api/v1/portfolios/{portfolio_id}/snapshots",
        json={"name": "평가 저장"},
    ).json()["snapshot"]["id"]

    load_client = TestClient(app)
    load_response = load_client.post(f"/api/v1/portfolios/snapshots/{snapshot_id}/load")
    assert load_response.status_code == 200
    payload = load_response.json()
    assert payload["analysis"]["metrics"][0]["ticker"] == "VOO"
    proposal_row = payload["evaluation"]["proposal"][0]
    assert proposal_row["ticker"] == "VOO"
    assert proposal_row["rc_gap_pct"] == 0.0
    assert proposal_row["suggested_trade_pct"] == 0.0
    assert proposal_row["action_reason"] == "히스테리시스 범위 및 최소 거래 미만"
    assert "adjusted_gap_pct" not in proposal_row
    assert payload["evaluation"]["ips_config_snapshot"]["target_allocation"]["core"]["target"] == 0.8

    csv_response = load_client.get("/api/v1/evaluation/download-csv?type=proposal")
    assert csv_response.status_code == 200
    assert "VOO" in csv_response.text
    assert "suggested_trade_pct" in csv_response.text
