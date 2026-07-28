from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from research.qlib_validation.artifacts import canonical_bytes, write_inputs, write_json
from research.qlib_validation.capability import assess_capability
from research.qlib_validation.contracts import SourceSnapshot
from research.qlib_validation.environment import environment_info
from research.qlib_validation.metrics import build_forward_observations, signal_verdict
from research.qlib_validation.replay import replay_regimes
from research.qlib_validation.source import load_snapshot


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("protocol.json")


def _protocol() -> tuple[dict[str, Any], str]:
    value = json.loads(PROTOCOL_PATH.read_text())
    return value, sha256(canonical_bytes(value)).hexdigest()


def _stale_series(
    snapshot: SourceSnapshot,
    *,
    as_of: datetime,
    maximum_staleness_days: int,
) -> list[str]:
    cutoff = as_of - timedelta(days=maximum_staleness_days)
    specs = {
        spec.key: spec for spec in (*snapshot.benchmark_specs, *snapshot.policy_specs)
    }
    return sorted(
        key
        for key in specs
        if not snapshot.candles_for(key)
        or max(item.available_at for item in snapshot.candles_for(key)) < cutoff
    )


def run_stage1(*, database: Path, as_of: datetime, output: Path) -> dict[str, Any]:
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    as_of = as_of.astimezone(UTC)
    protocol, protocol_hash = _protocol()
    snapshot = load_snapshot(database, as_of=as_of)
    input_fingerprint = sha256(
        canonical_bytes(
            {
                "as_of": as_of.isoformat(),
                "policy_hash": snapshot.policy_record["policy_hash"],
                "protocol_hash": protocol_hash,
                "candles": [item.record() for item in snapshot.candles],
            }
        )
    ).hexdigest()
    run_id = f"{as_of.strftime('%Y%m%dT%H%M%SZ')}-{input_fingerprint[:12]}"
    run_dir = output / run_id
    input_manifest = write_inputs(snapshot, run_dir, repository_root=ROOT)
    capability = assess_capability(snapshot)
    points = replay_regimes(
        snapshot,
        minimum_history=int(protocol["minimum_history"]),
    )
    forward = build_forward_observations(
        snapshot,
        points,
        tuple(protocol["horizons"]),
        **protocol["bootstrap"],
    )
    analysis = forward["analysis"]
    stale_series = _stale_series(
        snapshot,
        as_of=as_of,
        maximum_staleness_days=int(protocol["maximum_staleness_days"]),
    )
    unclassified_months = [
        point.month
        for point in points
        if point.regime not in {"risk_on", "neutral", "risk_off"}
    ]
    verdict = signal_verdict(
        analysis["effects"],
        risk_off_episodes=analysis["risk_off_episodes"],
        complete_coverage=not any(item["blocking"] for item in forward["missing"]),
        reproducible=not input_manifest["relevant_source_dirty"],
        source_fresh=not stale_series,
        replay_complete=not unclassified_months,
        minimum_risk_off_episodes=int(protocol["minimum_risk_off_episodes"]),
    )
    manifest = {
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "protocol_hash": protocol_hash,
        "policy_hash": snapshot.policy_record["policy_hash"],
        "environment": environment_info(),
        "qlib_capability": capability,
        "source_manifest": input_manifest["source_manifest"],
        "input_manifest": input_manifest["input_manifest"],
    }
    summary = {
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "protocol_hash": protocol_hash,
        "policy_hash": snapshot.policy_record["policy_hash"],
        "qlib_capability": capability,
        "regime_signal_verdict": verdict["verdict"],
        "regime_signal_reason": verdict["reason"],
        "target_policy_verdict": "inconclusive",
        "target_policy_reason": "stage2_not_run",
        "coverage_missing": forward["missing"],
        "stale_series": stale_series,
        "unclassified_months": unclassified_months,
        "effects": analysis["effects_serializable"],
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "replay.json", [item.record() for item in points])
    write_json(run_dir / "stage1-metrics.json", forward["rows"])
    write_json(run_dir / "summary.json", summary)
    return summary
