"""Application orchestration for persisted Toss inspection runs."""

from __future__ import annotations

from typing import Any, Mapping

from services.account_projection import (
    AccountProjectionError,
    build_account_projection,
    layer_map_from_policy,
)
from services.inspection_engine import (
    ENGINE_VERSION,
    evaluation_fingerprint,
    evaluate_inspection,
)
from services.risk_evidence import build_risk_evidence
from storage.account_observation_store import get_snapshot, list_snapshots
from storage.database import initialize_database
from storage.evaluation_store import insert_evaluation_run
from storage.market_store import list_adjusted_stock_candles
from storage.performance_store import latest_performance_run
from storage.policy_store import get_active_policy, policy_hash


def _select_market_candles(
    projection: dict[str, Any], risk_policy: dict[str, Any]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Select local Toss candles for each held identity at one snapshot."""
    identities = sorted(
        {
            (
                str(position.get("market_country", "")).upper(),
                str(position.get("symbol", "")).upper(),
            )
            for position in projection.get("positions", [])
            if position.get("market_country") and position.get("symbol")
        }
    )
    return {
        identity: list_adjusted_stock_candles(
            market_country=identity[0],
            symbol=identity[1],
            through_at=str(projection["synced_at"]),
            limit=int(risk_policy["lookback_sessions"]),
        )
        for identity in identities
    }


def _prepare_inputs(
    *,
    policy: dict[str, Any],
    policy_hash_value: str,
    policy_version_id: int | None,
    snapshot_id: int | None,
    account_alias: str,
    allow_missing_source: bool,
) -> dict[str, Any]:
    if "risk_review" not in policy:
        raise RuntimeError(
            "active IPS policy lacks risk_review; validate and explicitly activate a Phase 5 policy"
        )
    layer_map = layer_map_from_policy(policy)
    performance: dict[str, Any] | None = None
    projection: dict[str, Any] | None = None
    source_error: str | None = None
    source_snapshot = get_snapshot(snapshot_id) if snapshot_id is not None else None
    if source_snapshot is None and snapshot_id is None:
        recent = list_snapshots(limit=1)
        source_snapshot = recent[0] if recent else None
    try:
        projection = build_account_projection(
            snapshot_id=snapshot_id,
            account_alias=account_alias,
            layer_map=layer_map,
        )
    except AccountProjectionError as exc:
        source_error = str(exc)
        if source_snapshot is None and not allow_missing_source:
            raise RuntimeError(f"snapshot_id={snapshot_id} not found")
        if snapshot_id is None and source_snapshot is not None:
            source_error = (
                f"latest snapshot {source_snapshot['id']} is not evaluable: "
                f"{source_snapshot['state']}"
            )
    if projection is not None:
        performance = latest_performance_run(
            through_snapshot_id=projection["snapshot_id"]
        )
    candles = (
        _select_market_candles(projection, policy["risk_review"])
        if projection is not None
        else {}
    )
    risk_evidence = build_risk_evidence(
        projection,
        performance,
        candles,
        policy["risk_review"],
    )
    source_fingerprint = (
        projection.get("source_fingerprint")
        if projection
        else source_snapshot.get("source_fingerprint")
        if source_snapshot
        else "missing"
    )
    evidence_refs = {
        "snapshot_id": projection["snapshot_id"]
        if projection
        else source_snapshot["id"]
        if source_snapshot
        else None,
        "performance_run_id": performance["id"] if performance else None,
        "policy_version_id": policy_version_id,
        "policy_hash": policy_hash_value,
        "market_evidence_fingerprint": risk_evidence[
            "market_evidence_fingerprint"
        ],
    }
    result = evaluate_inspection(
        projection,
        performance,
        policy,
        risk_evidence=risk_evidence,
        evidence_refs=evidence_refs,
        source_error=source_error,
    )
    if projection is None and source_snapshot is not None:
        result["source"]["snapshot_id"] = source_snapshot["id"]
        result["source"]["source_fingerprint"] = source_snapshot[
            "source_fingerprint"
        ]
    fingerprint = evaluation_fingerprint(
        source_fingerprint=source_fingerprint,
        performance_fingerprint=performance.get("input_fingerprint")
        if performance
        else None,
        policy_hash=policy_hash_value,
        market_evidence_fingerprint=risk_evidence[
            "market_evidence_fingerprint"
        ],
    )
    allocation_state = result.get("allocation_state", "not_evaluable")
    persisted_state = (
        "complete" if allocation_state in {"complete", "partial"} else "not_evaluable"
    )
    return {
        "snapshot_id": projection["snapshot_id"]
        if projection
        else source_snapshot["id"]
        if source_snapshot
        else 0,
        "performance_run_id": performance["id"] if performance else None,
        "source_fingerprint": source_fingerprint,
        "performance_fingerprint": performance.get("input_fingerprint")
        if performance
        else None,
        "policy_version_id": policy_version_id,
        "policy_hash": policy_hash_value,
        # The persistence wrapper retains its historical state column while
        # the result itself uses the explicit allocation evaluability axis.
        "state": persisted_state,
        "non_evaluable_reason": result.get("allocation_reason")
        if persisted_state == "not_evaluable"
        else None,
        "result": result,
        "market_evidence_fingerprint": risk_evidence[
            "market_evidence_fingerprint"
        ],
        "market_evidence": risk_evidence["market_evidence"],
        "evaluation_fingerprint": fingerprint,
    }


def run_inspection(
    snapshot_id: int | None = None,
    account_alias: str = "toss-brokerage",
) -> dict[str, Any]:
    """Create or reuse one deterministic evaluation for the requested snapshot."""
    initialize_database()
    active = get_active_policy(account_alias)
    if active is None:
        raise RuntimeError("active IPS policy is not configured")
    prepared = _prepare_inputs(
        policy=active["policy"],
        policy_hash_value=active["policy_hash"],
        policy_version_id=active["id"],
        snapshot_id=snapshot_id,
        account_alias=account_alias,
        allow_missing_source=False,
    )
    prepared["account_alias"] = account_alias
    prepared["engine_version"] = ENGINE_VERSION
    return insert_evaluation_run(prepared)


def preview_inspection(
    policy: Mapping[str, Any],
    *,
    snapshot_id: int | None = None,
    account_alias: str = "toss-brokerage",
) -> dict[str, Any]:
    """Evaluate a proposed policy without activation or persistence."""
    initialize_database()
    normalized_policy = dict(policy)
    prepared = _prepare_inputs(
        policy=normalized_policy,
        policy_hash_value=policy_hash(normalized_policy),
        policy_version_id=None,
        snapshot_id=snapshot_id,
        account_alias=account_alias,
        allow_missing_source=True,
    )
    return {
        "persisted": False,
        "account_alias": account_alias,
        "policy_version_id": None,
        "policy_hash": prepared["policy_hash"],
        "snapshot_id": prepared["snapshot_id"],
        "evaluation_fingerprint": prepared["evaluation_fingerprint"],
        "market_evidence_fingerprint": prepared[
            "market_evidence_fingerprint"
        ],
        "market_evidence": prepared["market_evidence"],
        "evaluation": prepared["result"],
    }
