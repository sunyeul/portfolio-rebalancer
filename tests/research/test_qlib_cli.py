import json

import pytest

from research.qlib_validation.cli import main


def test_cli_writes_one_json_object_to_stdout(monkeypatch, capsys, tmp_path):
    def noisy_run(**kwargs):
        print("dependency progress noise")
        return {
            "run_id": "fixed-run",
            "regime_signal_verdict": "inconclusive",
            "target_policy_verdict": "inconclusive",
        }

    monkeypatch.setattr("research.qlib_validation.cli.run_stage1", noisy_run)
    code = main(
        [
            "stage1",
            "--db",
            str(tmp_path / "db.sqlite3"),
            "--as-of",
            "2026-07-28T00:00:00+00:00",
            "--output",
            str(tmp_path / "artifacts"),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "fixed-run"


def test_cli_passes_current_holdings_universe(monkeypatch, capsys, tmp_path):
    seen = {}

    def run(**kwargs):
        seen.update(kwargs)
        return {"run_id": "fixed-run"}

    monkeypatch.setattr("research.qlib_validation.cli.run_stage1", run)

    code = main(
        [
            "stage1",
            "--db",
            str(tmp_path / "db.sqlite3"),
            "--as-of",
            "2026-07-28T00:00:00+00:00",
            "--output",
            str(tmp_path / "artifacts"),
            "--universe",
            "current-holdings",
        ]
    )

    assert code == 0
    assert seen["universe"] == "current-holdings"
    assert json.loads(capsys.readouterr().out)["run_id"] == "fixed-run"


def test_forecast_cli_writes_one_json_object_to_stdout(monkeypatch, capsys, tmp_path):
    def noisy_run(**kwargs):
        print("dependency progress noise")
        return {"run_id": "forecast-fixed-run", "prediction_horizon_sessions": 20}

    monkeypatch.setattr("research.qlib_validation.cli.run_forecast", noisy_run)
    code = main(
        [
            "forecast",
            "--db",
            str(tmp_path / "db.sqlite3"),
            "--as-of",
            "2026-07-28T00:00:00+00:00",
            "--output",
            str(tmp_path / "artifacts"),
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "forecast-fixed-run"
    assert payload["prediction_horizon_sessions"] == 20


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["stage1"],
        ["stage1", "--db", "fixture.sqlite3"],
        ["unknown"],
    ],
)
def test_cli_argument_errors_are_one_json_object_on_stdout(argv, capsys):
    code = main(argv)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 2
    assert payload["ok"] is False
    assert payload["error"] == "ArgumentError"
    assert payload["message"]
    assert captured.err == ""
