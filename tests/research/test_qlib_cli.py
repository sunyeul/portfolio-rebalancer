import json

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
