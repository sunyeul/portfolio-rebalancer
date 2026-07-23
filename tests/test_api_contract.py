from fastapi.testclient import TestClient

from api.app import create_app
from services.dynamic_allocation import DEFAULT_ALLOCATION_REVIEW
from storage.account_observation_store import insert_snapshot
from tests.test_account_observation_store import _snapshot


def test_api_uses_read_only_sanitized_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["data"]["account_alias"] == "toss-brokerage"
    assert "TOSS_OPEN_API_CLIENT_SECRET" not in response.text
    assert response.cookies.get("ips_pilot_session")


def test_api_has_no_write_routes_and_reports_missing_account(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    client = TestClient(create_app())
    response = client.get("/api/account")
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert client.post("/api/policy", json={}).status_code == 405


def test_api_source_contract_redacts_order_and_execution_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    from storage.database import initialize_database

    initialize_database()
    insert_snapshot(_snapshot())
    client = TestClient(create_app())
    payloads = [client.get("/api/health").json(), client.get("/api/snapshots").json()]
    forbidden = (
        "order_id",
        "side",
        "quantity",
        "order_amount_native",
        "filled_quantity",
        "filled_amount_native",
    )
    encoded = str(payloads)
    assert all(field not in encoded for field in forbidden)


def test_api_can_read_anchored_performance_and_policy_versions(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    monkeypatch.setattr(
        "api.app.get_performance_run", lambda run_id: {"id": run_id, "points": []}
    )
    monkeypatch.setattr(
        "api.app.get_policy_version",
        lambda version_id: {"id": version_id, "version": version_id, "policy": {}},
    )
    client = TestClient(create_app())

    performance = client.get("/api/performance?run_id=17").json()
    policy = client.get("/api/policy?version_id=23").json()

    assert performance["data"]["run"]["id"] == 17
    assert policy["data"]["policy"]["id"] == 23


def test_market_context_passes_composite_series_and_policy_timestamp(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    active = {
        "id": 4,
        "account_alias": "toss-brokerage",
        "created_at": "2026-07-20T00:00:00+00:00",
        "policy": {
            "cash_reserve": {"target": 0.05},
            "allocation_review": DEFAULT_ALLOCATION_REVIEW,
        },
    }
    captured = {}
    monkeypatch.setattr("api.app.get_active_policy", lambda: active)
    monkeypatch.setattr("api.app.list_candles", lambda **_: [])

    def fake_evaluate(series, **kwargs):
        captured["series"] = series
        captured.update(kwargs)
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "regime": None,
            "verification_task": "verify",
        }

    monkeypatch.setattr("api.app.evaluate_dynamic_allocation", fake_evaluate)
    monkeypatch.setattr("api.app.latest_policy_candidate", lambda *args: None)
    client = TestClient(create_app())

    response = client.get("/api/market-context")

    assert response.status_code == 200
    assert set(captured["series"]) == {
        "US/SPY",
        "US/QQQ",
        "KR/KOSPI",
        "KR/KOSDAQ",
    }
    assert captured["active_policy"] is active["policy"]
    assert captured["last_change_at"] == active["created_at"]


def test_api_returns_persisted_phase5_evidence_without_reclassification(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    evaluation = {
        "id": 11,
        "snapshot_id": 7,
        "policy_version_id": 3,
        "result": {
            "state": "complete",
            "account": {
                "investment_principal_krw": 1000000.0,
                "account_profit_krw": 120000.0,
                "account_return": 0.12,
            },
            "account_profit_loss": {
                "status": "Review",
                "drawdown": {"state": "complete", "current": -0.16},
            },
            "review_queue": [
                {
                    "identity": "US/AAA",
                    "status": "Review",
                    "triggers": ["instrument_out_of_range"],
                }
            ],
        },
        "market_evidence": {"US/AAA": {"state": "complete"}},
    }
    monkeypatch.setattr("api.app.latest_evaluation_run", lambda: evaluation)
    client = TestClient(create_app())

    inspection = client.get("/api/inspection").json()

    assert (
        inspection["data"]["evaluation"]["result"]["review_queue"][0]["status"]
        == "Review"
    )
    assert (
        inspection["data"]["evaluation"]["result"]["account"][
            "investment_principal_krw"
        ]
        == 1000000.0
    )
    assert (
        inspection["data"]["evaluation"]["result"]["account"]["account_return"] == 0.12
    )
    assert (
        inspection["data"]["evaluation"]["market_evidence"]["US/AAA"]["state"]
        == "complete"
    )


def test_api_contract_gate_preserves_historical_result_without_adapter(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    evaluation = {
        "id": 4,
        "engine_version": "phase5-v1",
        "result": {
            "engine_version": "phase5-v2",
            "state": "complete",
            "review_queue": [
                {
                    "kind": "cash",
                    "identity": "cash_reserve",
                    "status": "Review",
                    "priority": 2,
                    "next_step": "legacy wording",
                }
            ],
        },
    }
    monkeypatch.setattr("api.app.latest_evaluation_run", lambda: evaluation)
    client = TestClient(create_app())

    inspection = client.get("/api/inspection").json()
    queue = client.get("/api/review-queue").json()

    assert inspection["data"]["contract_supported"] is False
    assert inspection["data"]["evaluation"]["result"]["review_queue"][0][
        "priority"
    ] == 2
    assert queue["data"]["contract_supported"] is False
    assert queue["data"]["items"][0]["next_step"] == "legacy wording"
    assert "adjustment_suggestions" not in queue["data"]


def test_api_contract_gate_exposes_v2_adjustment_suggestions(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    evaluation = {
        "id": 8,
        "engine_version": "phase5-v2",
        "result": {
            "engine_version": "phase5-v2",
            "state": "complete",
            "adjustment_suggestions": [
                {
                    "priority": "P1",
                    "suggestion": {"code": "review_increase_regular_purchase_allocation"},
                }
            ],
            "review_queue": [],
        },
    }
    monkeypatch.setattr("api.app.latest_evaluation_run", lambda: evaluation)
    client = TestClient(create_app())

    inspection = client.get("/api/inspection").json()
    queue = client.get("/api/review-queue").json()

    assert inspection["data"]["contract_supported"] is True
    assert queue["data"]["contract_supported"] is True
    assert queue["data"]["adjustment_suggestions"] == evaluation["result"][
        "adjustment_suggestions"
    ]


def test_api_projects_latest_performance_point_into_account_contract(monkeypatch):
    evaluation = {
        "performance_run_id": 9,
        "result": {
            "account": {
                "total_value_krw": 1000000.0,
                "cash_weight_gross": 0.2,
                "legacy_field": "not public",
            }
        },
    }
    monkeypatch.setattr("api.app.latest_evaluation_run", lambda: evaluation)
    monkeypatch.setattr(
        "api.app.get_performance_run",
        lambda run_id: {
            "id": run_id,
            "points": [
                {
                    "id": 3,
                    "point_at": "2026-07-23T00:00:00+00:00",
                    "evaluation_state": "evaluable",
                    "total_value_krw": 1100000.0,
                    "invested_value_krw": 900000.0,
                    "cash_value_krw": 200000.0,
                    "investment_principal_krw": 1000000.0,
                    "account_gain_krw": 100000.0,
                    "simple_return": 0.1,
                }
            ],
        },
    )
    client = TestClient(create_app())

    account = client.get("/api/inspection").json()["data"]["evaluation"]["result"][
        "account"
    ]

    assert account == {
        "total_value_krw": 1100000.0,
        "invested_value_krw": 900000.0,
        "cash_value_krw": 200000.0,
        "cash_weight_gross": 0.2,
        "investment_principal_krw": 1000000.0,
        "account_profit_krw": 100000.0,
        "account_return": 0.1,
    }
