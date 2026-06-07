from fastapi.testclient import TestClient

from main import app
from storage.config_store import get_ips_config
from storage.database import connect, initialize_database


def _client_with_db(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PORTFOLIO_DB_PATH",
        str(tmp_path / "portfolio_rebalancer.sqlite3"),
    )
    initialize_database()
    return TestClient(app)


def test_config_options_and_ips_are_seeded_from_defaults(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    options = client.get("/api/v1/config/options").json()
    assert options["groups"][0]["value"] == "ungrouped"
    assert any(
        group["value"] == "core" and group["group_type"] == "core"
        for group in options["groups"]
    )

    ips = client.get("/api/v1/config/ips").json()
    assert ips["ips_config"]["target_allocation"]["core"]["target"] == 0.8
    assert ips["ips_config"]["groups"]["satellite_space"]["type"] == "satellite"


def test_config_option_crud_uses_soft_deactivation(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    create_response = client.post(
        "/api/v1/config/groups",
        json={
            "code": "satellite_bio",
            "label": "위성: 바이오",
            "group_type": "satellite",
            "sort_order": 90,
        },
    )
    assert create_response.status_code == 200
    assert create_response.json()["option"]["value"] == "satellite_bio"

    deactivate_response = client.patch(
        "/api/v1/config/groups/satellite_bio/active",
        json={"is_active": False},
    )
    assert deactivate_response.status_code == 200
    assert deactivate_response.json()["option"]["is_active"] is False

    active_only = client.get("/api/v1/config/options?include_inactive=false").json()
    assert all(group["value"] != "satellite_bio" for group in active_only["groups"])


def test_app_defined_config_values_are_read_only(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    role_response = client.post(
        "/api/v1/config/roles",
        json={"code": "custom_role", "label": "사용자 역할"},
    )
    assert role_response.status_code == 403

    thesis_response = client.patch(
        "/api/v1/config/thesis_statuses/intact/active",
        json={"is_active": False},
    )
    assert thesis_response.status_code == 403

    priority_response = client.put(
        "/api/v1/config/ips/action-priorities",
        json=[
            {
                "action_code": "custom_action",
                "label": "사용자 액션",
                "priority": 99,
                "is_active": True,
            }
        ],
    )
    assert priority_response.status_code == 403


def test_ips_edits_are_loaded_by_runtime_config(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    response = client.put(
        "/api/v1/config/ips/target-allocations",
        json=[
            {"group_type": "core", "min": 0.6, "target": 0.7, "max": 0.8},
            {"group_type": "satellite", "min": 0.2, "target": 0.3, "max": 0.4},
        ],
    )
    assert response.status_code == 200
    assert get_ips_config()["target_allocation"]["core"]["target"] == 0.7


def test_unknown_input_metadata_warns_and_does_not_create_options(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    response = client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 100,
                    "group": "typo_group",
                    "role": "typo_role",
                    "thesis_status": "typo_status",
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["assets"][0]["group"] == "ungrouped"
    assert payload["assets"][0]["role"] == "unknown"
    assert payload["assets"][0]["thesis_status"] == "unknown"
    assert len(payload["warnings"]) == 3

    with connect() as conn:
        row = conn.execute(
            "SELECT id FROM groups WHERE code = ?",
            ("typo_group",),
        ).fetchone()
    assert row is None
