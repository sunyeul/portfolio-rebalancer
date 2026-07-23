import json

from typer.testing import CliRunner

from cli import app


runner = CliRunner()


class FakeTossService:
    def health(self):
        return {
            "ok": True,
            "checks": {
                "config": "ok",
                "oauth": "ok",
                "account_discovery": "ok",
                "account_match": "ok",
            },
            "account_count": 1,
            "error": None,
        }

    def sync(self, **kwargs):
        return {
            "id": 7,
            "state": "complete",
            "account_alias": "toss-brokerage",
            "source_fingerprint": "safe-fingerprint",
            "holdings": [],
            "cash": [],
            "fx_rate": None,
            "orders": [],
        }


def test_toss_health_emits_one_sanitized_json_object(monkeypatch):
    monkeypatch.setattr("cli._build_toss_service", lambda: (FakeTossService(), None))

    result = runner.invoke(app, ["toss-health"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "toss-health"
    assert payload["checks"]["oauth"] == "ok"
    assert "12345678901" not in result.stdout
    assert "access-token" not in result.stdout


def test_toss_sync_emits_snapshot_and_uses_requested_order_window(monkeypatch):
    service = FakeTossService()
    captured = {}
    original_sync = service.sync

    def fake_sync(**kwargs):
        captured.update(kwargs)
        return original_sync()

    service.sync = fake_sync
    monkeypatch.setattr("cli._build_toss_service", lambda: (service, None))
    monkeypatch.setattr("cli.initialize_database", lambda: None)

    result = runner.invoke(
        app,
        [
            "toss-sync",
            "--from",
            "2026-01-01",
            "--to",
            "2026-07-23",
            "--max-order-pages",
            "3",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["snapshot_id"] == 7
    assert payload["state"] == "complete"
    assert captured == {
        "from_date": "2026-01-01",
        "to_date": "2026-07-23",
        "max_order_pages": 3,
    }


def test_toss_snapshots_latest_reads_local_store_only(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr(
        "cli.list_account_snapshots",
        lambda limit=20: [
            {"id": 9, "state": "partial", "account_alias": "toss-brokerage"}
        ],
    )

    result = runner.invoke(app, ["toss-snapshots", "--latest"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["snapshots"]["id"] == 9
    assert payload["snapshots"]["state"] == "partial"


def test_toss_health_returns_machine_readable_config_error(monkeypatch):
    def fail_build():
        raise RuntimeError(
            "Missing required environment variable: TOSS_OPEN_API_CLIENT_SECRET"
        )

    monkeypatch.setattr("cli._build_toss_service", fail_build)

    result = runner.invoke(app, ["toss-health"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["command"] == "toss-health"
    assert "TOSS_OPEN_API_CLIENT_SECRET" in payload["error"]["message"]
    assert "access-token" not in result.stdout
