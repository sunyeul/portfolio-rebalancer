from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from research.qlib_validation.contracts import SourceSnapshot


RELEVANT_SOURCE_PATHS = (
    "services/dynamic_allocation.py",
    "research/qlib_validation",
)

FIXED_REPRODUCIBILITY_FILES = (
    "services/dynamic_allocation.py",
    "research/qlib_validation/protocol.json",
    "research/qlib_validation/pyproject.toml",
    "research/qlib_validation/uv.lock",
)


class ArtifactError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write(path, canonical_bytes(value))


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _source_manifest(repository_root: Path) -> dict[str, Any]:
    files = set((repository_root / "research/qlib_validation").rglob("*.py"))
    files.update(repository_root / value for value in FIXED_REPRODUCIBILITY_FILES)
    hashes = {
        str(path.relative_to(repository_root)): sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
        if path.is_file()
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", *RELEVANT_SOURCE_PATHS],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_commit": commit, "relevant_source_dirty": dirty, "files": hashes}


def write_inputs(
    snapshot: SourceSnapshot, run_dir: Path, *, repository_root: Path
) -> dict[str, Any]:
    if run_dir.exists():
        raise ArtifactError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    policy_payload = canonical_bytes(snapshot.policy_record["policy"])
    candle_lines = b"".join(canonical_bytes(item.record()) for item in snapshot.candles)
    _atomic_write(run_dir / "policy.json", policy_payload)
    _atomic_write(run_dir / "candles.jsonl", candle_lines)
    source = _source_manifest(repository_root)
    _atomic_write(run_dir / "source-manifest.json", canonical_bytes(source))
    by_key = {
        spec.key: sorted(
            snapshot.candles_for(spec.key), key=lambda item: item.available_at
        )
        for spec in (*snapshot.benchmark_specs, *snapshot.policy_specs)
    }
    manifest = {
        "policy_hash": snapshot.policy_record["policy_hash"],
        "series": {
            key: {
                "rows": len(items),
                "min_available_at": items[0].available_at.isoformat()
                if items
                else None,
                "max_available_at": items[-1].available_at.isoformat()
                if items
                else None,
            }
            for key, items in sorted(by_key.items())
        },
        "files": {
            "policy.json": {"sha256": _digest(policy_payload)},
            "candles.jsonl": {
                "sha256": _digest(candle_lines),
                "rows": len(snapshot.candles),
            },
        },
    }
    _atomic_write(run_dir / "input-manifest.json", canonical_bytes(manifest))
    return {
        "policy_sha256": manifest["files"]["policy.json"]["sha256"],
        "candles_sha256": manifest["files"]["candles.jsonl"]["sha256"],
        "relevant_source_dirty": source["relevant_source_dirty"],
        "source_manifest": source,
        "input_manifest": manifest,
    }
