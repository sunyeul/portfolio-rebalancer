from fastapi.testclient import TestClient

from api.app import create_app
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


def test_market_context_passes_policy_timestamp_to_cooling_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    active = {
        "id": 4,
        "account_alias": "toss-brokerage",
        "created_at": "2026-07-20T00:00:00+00:00",
        "policy": {"cash_reserve": {"target": 0.15}},
    }
    captured = {}
    monkeypatch.setattr("api.app.get_active_policy", lambda: active)
    monkeypatch.setattr("api.app.list_candles", lambda **_: [])

    def fake_evaluate(candles, **kwargs):
        captured.update(kwargs)
        return {
            "status": "Watch",
            "candidate_state": "observe",
            "history_points": 0,
            "current_target": 0.15,
            "proposed_target": None,
            "verification_task": "verify",
        }

    monkeypatch.setattr("api.app.evaluate_market_context", fake_evaluate)
    monkeypatch.setattr("api.app.latest_policy_candidate", lambda *args: None)
    client = TestClient(create_app())

    response = client.get("/api/market-context")

    assert response.status_code == 200
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
                    "status": "Action",
                    "triggers": ["broken_thesis_and_hard_maximum_breach"],
                }
            ],
        },
        "market_evidence": {"US/AAA": {"state": "complete"}},
    }
    monkeypatch.setattr("api.app.latest_evaluation_run", lambda: evaluation)
    monkeypatch.setattr(
        "api.app.list_profiles",
        lambda: [
            {
                "market_country": "US",
                "symbol": "AAA",
                "overlap_status": "review",
                "management_burden_status": "clear",
                "holdability_status": "unknown",
                "etf_substitution_status": "not_applicable",
                "review_factors_note": "review",
            }
        ],
    )
    client = TestClient(create_app())

    inspection = client.get("/api/inspection").json()
    profiles = client.get("/api/profiles").json()

    assert (
        inspection["data"]["evaluation"]["result"]["review_queue"][0]["status"]
        == "Action"
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
    assert profiles["data"]["profiles"][0]["overlap_status"] == "review"


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
