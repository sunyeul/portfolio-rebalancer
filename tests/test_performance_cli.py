import json

from typer.testing import CliRunner

from cli import app


runner = CliRunner()


def _run():
    return {
        "id": 3,
        "baseline_id": 1,
        "through_snapshot_id": 2,
        "state": "complete",
        "data_quality": {"issues": []},
        "points": [],
        "candidates": [],
        "execution_count": 0,
    }


def test_baseline_preview_emits_one_sanitized_json_object(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr(
        "cli.preview_baseline",
        lambda snapshot_id: {
            "id": snapshot_id,
            "account_alias": "toss-brokerage",
            "state": "complete",
            "is_current_evaluable": 1,
            "total_value_krw": 120802745.17802304,
        },
    )

    result = runner.invoke(
        app, ["performance", "baseline-preview", "--snapshot-id", "4"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "performance baseline-preview"
    assert payload["snapshot"]["id"] == 4
    assert "accountNo" not in result.stdout


def test_baseline_confirm_emits_baseline(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr(
        "cli.create_baseline",
        lambda snapshot_id, expected_principal_krw: {
            "id": 1,
            "baseline_snapshot_id": snapshot_id,
            "initial_principal_krw": expected_principal_krw,
        },
    )

    result = runner.invoke(
        app,
        [
            "performance",
            "baseline-confirm",
            "--snapshot-id",
            "4",
            "--expected-principal-krw",
            "120802745.17802304",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["baseline"]["baseline_snapshot_id"] == 4


def test_refresh_and_history_emit_run(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr("cli.refresh_performance", _run)
    monkeypatch.setattr("cli.latest_performance_run", _run)

    refreshed = runner.invoke(app, ["performance", "refresh"])
    history = runner.invoke(app, ["performance", "history", "--latest"])

    assert refreshed.exit_code == 0
    assert history.exit_code == 0
    assert json.loads(refreshed.stdout)["run_id"] == 3
    assert json.loads(history.stdout)["run"]["state"] == "complete"


def test_candidates_and_decision_emit_safe_json(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)
    monkeypatch.setattr(
        "cli.get_performance_run",
        lambda run_id: {**_run(), "id": run_id, "candidates": [{"id": 9}]},
    )
    monkeypatch.setattr(
        "cli.append_cash_flow_decision",
        lambda candidate_id, **kwargs: {
            "id": 4,
            "candidate_id": candidate_id,
            "classification": kwargs["classification"],
        },
    )

    candidates = runner.invoke(app, ["performance", "candidates", "--run-id", "3"])
    decision = runner.invoke(
        app,
        [
            "performance",
            "decide-flow",
            "--candidate-id",
            "9",
            "--classification",
            "internal_fx",
        ],
    )

    assert candidates.exit_code == 0
    assert decision.exit_code == 0
    assert json.loads(candidates.stdout)["candidates"] == [{"id": 9}]
    assert json.loads(decision.stdout)["decision"]["classification"] == "internal_fx"


def test_history_rejects_latest_and_run_id_together(monkeypatch):
    monkeypatch.setattr("cli.initialize_database", lambda: None)

    result = runner.invoke(
        app,
        ["performance", "history", "--latest", "--run-id", "3"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "input"
    assert "access-token" not in result.stdout
