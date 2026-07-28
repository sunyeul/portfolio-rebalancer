import json
from pathlib import Path

import pytest

from research.qlib_validation.artifacts import (
    ArtifactError,
    canonical_bytes,
    staged_run,
    write_inputs,
)


def test_write_inputs_is_canonical_hashed_and_never_overwrites(
    snapshot_factory, tmp_path
):
    snapshot = snapshot_factory()
    repository_root = Path(__file__).resolve().parents[2]
    result = write_inputs(snapshot, tmp_path / "run", repository_root=repository_root)
    policy = (tmp_path / "run" / "policy.json").read_bytes()
    manifest = json.loads((tmp_path / "run" / "input-manifest.json").read_text())

    assert policy == canonical_bytes(snapshot.policy_record["policy"])
    assert manifest["policy_hash"] == snapshot.policy_record["policy_hash"]
    assert manifest["files"]["candles.jsonl"]["sha256"] == result["candles_sha256"]
    assert manifest["series"][snapshot.benchmark_specs[0].key]["rows"] == 1
    source = json.loads((tmp_path / "run" / "source-manifest.json").read_text())
    assert "research/qlib_validation/protocol.json" in source["files"]
    assert "research/qlib_validation/uv.lock" in source["files"]
    assert "services/policy_validation.py" in source["files"]
    assert "storage/policy_store.py" in source["files"]

    with pytest.raises(ArtifactError, match="already exists"):
        write_inputs(snapshot, tmp_path / "run", repository_root=tmp_path)


def test_staged_run_only_publishes_complete_results(tmp_path):
    final = tmp_path / "artifacts" / "fixed-run"

    with pytest.raises(RuntimeError, match="mid-run failure"):
        with staged_run(final) as staging:
            staging.mkdir()
            (staging / "partial.json").write_text("partial")
            raise RuntimeError("mid-run failure")

    assert not final.exists()

    with staged_run(final) as staging:
        staging.mkdir()
        (staging / "summary.json").write_text("complete")

    assert (final / "summary.json").read_text() == "complete"
    with pytest.raises(ArtifactError, match="already exists"):
        with staged_run(final):
            pass
