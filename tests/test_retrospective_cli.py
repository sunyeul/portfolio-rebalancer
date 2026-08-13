import json

from typer.testing import CliRunner

from cli import app


runner = CliRunner()


def test_retrospective_commands_emit_one_json_object(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr(
        "cli.eligible_retrospective",
        lambda: {"evaluation_run_id": 5, "snapshot_id": 10, "items": []},
    )
    monkeypatch.setattr(
        "cli.start_case",
        lambda **_: {"id": 3, "queue_kind": "cash", "queue_identity": "cash_reserve"},
    )
    monkeypatch.setattr("cli.list_case_summaries", lambda state: [])
    monkeypatch.setattr(
        "cli.preview_retrospective",
        lambda case_id, horizon: {"state": "not_ready", "case_id": case_id, "horizon": horizon, "evidence": None, "evidence_fingerprint": None},
    )
    monkeypatch.setattr(
        "cli.confirm_retrospective",
        lambda **_: {"id": 7, "revision": 1, "horizon": "1m"},
    )
    monkeypatch.setattr(
        "cli.show_retrospective",
        lambda case_id: {"case": {"id": case_id}, "reviews": [], "latest_by_horizon": {}},
    )

    commands = [
        ["retrospective", "eligible"],
        ["retrospective", "start", "--kind", "cash", "--identity", "cash_reserve", "--disposition", "adopted"],
        ["retrospective", "list", "--state", "all"],
        ["retrospective", "preview", "--case-id", "3", "--horizon", "1m"],
        ["retrospective", "confirm", "--case-id", "3", "--horizon", "1m", "--evidence-fingerprint", "fingerprint", "--judgment", "supported", "--execution", "aligned", "--policy", "maintain"],
        ["retrospective", "show", "--case-id", "3"],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert payload["ok"] is True
        assert payload["command"].startswith("retrospective ")
