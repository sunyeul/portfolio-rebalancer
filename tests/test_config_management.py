from fastapi.testclient import TestClient

from main import app
from storage.config_store import get_ips_config
from storage.database import initialize_database


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
    assert set(options) == {"thesis_statuses"}
    assert [row["value"] for row in options["thesis_statuses"]] == [
        "unknown",
        "intact",
        "watch",
        "broken",
    ]

    ips = client.get("/api/v1/config/ips").json()
    assert ips["ips_config"]["target_allocation"]["core"]["target"] == 0.8
    assert list(ips["ips_config"]["action_priority"]) == [
        "block_action",
        "rebalance_sell_review",
        "risk_control_review",
        "review_before_action",
        "reduce_or_pause_dca",
        "increase_dca",
        "hold_observe",
    ]


def test_app_defined_config_values_are_read_only(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    thesis_response = client.patch(
        "/api/v1/config/thesis_statuses/intact/active",
        json={"is_active": False},
    )
    assert thesis_response.status_code == 405

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
            {"group": "core", "min": 0.6, "target": 0.7, "max": 0.8},
            {"group": "satellite", "min": 0.2, "target": 0.3, "max": 0.4},
        ],
    )
    assert response.status_code == 200
    assert get_ips_config()["target_allocation"]["core"]["target"] == 0.7


def test_invalid_target_group_is_rejected(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    response = client.put(
        "/api/v1/config/ips/target-allocations",
        json=[{"group": "cash", "min": 0, "target": 0, "max": 0}],
    )

    assert response.status_code == 400


def test_unknown_input_group_warns_and_uses_unclassified(monkeypatch, tmp_path):
    client = _client_with_db(monkeypatch, tmp_path)

    response = client.post(
        "/api/v1/portfolio/manual",
        json={
            "rows": [
                {
                    "ticker": "VOO",
                    "allocation": 100,
                    "group": "typo_group",
                    "thesis_status": "typo_status",
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["assets"][0]["group"] == "unclassified"
    assert payload["assets"][0]["thesis_status"] == "unknown"
    assert len(payload["warnings"]) == 2
