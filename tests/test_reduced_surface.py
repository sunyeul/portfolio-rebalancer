from fastapi.testclient import TestClient
from typer.testing import CliRunner

from api.app import _contract_supported as api_contract_supported
from api.app import create_app
from cli import _contract_supported as cli_contract_supported
from cli import app
from storage.database import connect, initialize_database
from storage.evaluation_store import insert_evaluation_run
from storage.policy_store import get_active_policy


def test_api_exposes_only_the_reduced_read_surface(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    client = TestClient(create_app())

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/inspection").status_code == 200
    assert client.get("/api/account").status_code == 404
    assert client.get("/api/market-context").status_code == 404
    assert client.get("/api/policy-preflight").status_code == 404


def test_removed_cli_commands_keep_json_parse_errors():
    result = CliRunner().invoke(app, ["market", "context"])

    assert result.exit_code != 0
    assert '"ok": false' in result.output


def test_contract_gate_rejects_unsupported_wrapper_with_current_result():
    unsupported_wrapper = {
        "engine_version": "phase4-v1",
        "result": {
            "engine_version": "phase5-v2",
            "source": {},
            "allocation_state": "not_evaluable",
            "account": {},
            "layers": [],
            "instruments": [],
            "review_queue": [],
        },
    }

    assert api_contract_supported(unsupported_wrapper) is False
    assert cli_contract_supported(unsupported_wrapper) is False


def test_api_returns_persisted_account_without_hydration(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    initialize_database()
    with connect() as conn:
        snapshot_id = int(
            conn.execute(
                "INSERT INTO broker_account_snapshots (account_alias, sync_started_at, synced_at, state, is_current_evaluable, source_fingerprint, source_timestamps_json, data_quality_json, reconciliation_json) VALUES ('toss-brokerage','a','b','failed',0,'source','{}','{}','{}')"
            ).lastrowid
        )
    active = get_active_policy()
    expected = {
        "total_value_krw": None,
        "invested_value_krw": None,
        "cash_value_krw": None,
        "cash_weight_gross": None,
        "investment_principal_krw": None,
        "account_profit_krw": None,
        "account_return": None,
    }
    insert_evaluation_run(
        {
            "account_alias": "toss-brokerage",
            "snapshot_id": snapshot_id,
            "performance_run_id": None,
            "policy_version_id": active["id"],
            "source_fingerprint": "source",
            "performance_fingerprint": None,
            "policy_hash": active["policy_hash"],
            "engine_version": "phase5-v2",
            "state": "not_evaluable",
            "non_evaluable_reason": "missing_source",
            "result": {
                "engine_version": "phase5-v2",
                "source": {},
                "allocation_state": "not_evaluable",
                "account": expected,
                "layers": [],
                "instruments": [],
                "review_queue": [],
            },
            "market_evidence_fingerprint": "market-v1",
            "market_evidence": {},
            "evaluation_fingerprint": "unhydrated-account",
        }
    )

    client = TestClient(create_app())

    response = client.get("/api/inspection")

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["contract_supported"] is True
    assert data["evaluation"]["result"]["account"] == expected
