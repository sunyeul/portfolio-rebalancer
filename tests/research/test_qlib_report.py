from datetime import UTC, datetime

from research.qlib_validation.report import run_stage1


def test_stage1_summary_never_claims_stage2_or_emits_execution_fields(
    monkeypatch, snapshot_factory, tmp_path
):
    snapshot = snapshot_factory(days=80)

    def fake_write_inputs(value, run_dir, *, repository_root):
        assert value is snapshot
        run_dir.mkdir(parents=True)
        return {
            "relevant_source_dirty": False,
            "source_manifest": {"git_commit": "test", "files": {}},
            "input_manifest": {"policy_hash": "test", "files": {}},
        }

    monkeypatch.setattr(
        "research.qlib_validation.report.load_snapshot",
        lambda *args, **kwargs: snapshot,
    )
    monkeypatch.setattr(
        "research.qlib_validation.report.write_inputs", fake_write_inputs
    )
    monkeypatch.setattr(
        "research.qlib_validation.report.assess_capability",
        lambda value: {
            "data_adapter_suitable": False,
            "backtest_engine_suitable": False,
            "reasons": ["factor_unavailable"],
        },
    )
    monkeypatch.setattr(
        "research.qlib_validation.report.environment_info",
        lambda: {"python": "3.12.test", "pyqlib": "0.9.7"},
    )
    monkeypatch.setattr(
        "research.qlib_validation.report.replay_regimes",
        lambda value, **kwargs: [],
    )
    monkeypatch.setattr(
        "research.qlib_validation.report.build_forward_observations",
        lambda *args, **kwargs: {
            "rows": [],
            "missing": [],
            "analysis": {
                "effects": {},
                "effects_serializable": {},
                "risk_off_episodes": 0,
            },
        },
    )
    summary = run_stage1(
        database=tmp_path / "ignored.sqlite3",
        as_of=datetime(2026, 7, 28, tzinfo=UTC),
        output=tmp_path / "artifacts",
    )
    assert summary["regime_signal_verdict"] in {
        "supported",
        "inconclusive",
        "not_supported",
    }
    assert summary["target_policy_verdict"] == "inconclusive"
    assert summary["target_policy_reason"] == "stage2_not_run"
    assert (tmp_path / "artifacts" / summary["run_id"] / "manifest.json").is_file()

    forbidden = {"buy", "sell", "execute", "order_size", "status"}

    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value), set())
        return set()

    assert forbidden.isdisjoint(keys(summary))
