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
