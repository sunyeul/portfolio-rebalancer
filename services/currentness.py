"""Single currentness contract for IPS source and persisted evaluations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


MAX_SNAPSHOT_AGE_SECONDS = 72 * 60 * 60


def utc_now() -> datetime:
    """Return the evaluation clock as an injectable UTC boundary."""
    return datetime.now(timezone.utc)


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def evaluate_currentness(
    *,
    snapshot: Mapping[str, Any] | None,
    active_policy: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None = None,
    evaluated_at: datetime | None = None,
    require_evaluation: bool = False,
) -> dict[str, Any]:
    """Return one fail-closed currentness result for every consumer.

    ``require_evaluation`` distinguishes source validation before a new run from
    checking a persisted evaluation in API/CLI read paths.
    """
    now = (evaluated_at or utc_now()).astimezone(timezone.utc)
    snapshot_at = _timestamp(snapshot.get("synced_at")) if snapshot else None
    snapshot_age_seconds = (
        int((now - snapshot_at).total_seconds()) if snapshot_at is not None else None
    )
    reasons: list[str] = []
    if snapshot is None:
        reasons.append("current_snapshot_missing")
    else:
        if snapshot.get("state") != "complete":
            reasons.append("snapshot_not_complete")
        if snapshot.get("is_current_evaluable") is not True:
            reasons.append("snapshot_not_evaluable")
        if snapshot_at is None:
            reasons.append("snapshot_timestamp_invalid")
        elif snapshot_at > now:
            reasons.append("snapshot_timestamp_future")
        elif (
            snapshot_age_seconds is not None
            and snapshot_age_seconds > MAX_SNAPSHOT_AGE_SECONDS
        ):
            reasons.append("snapshot_stale")
    if active_policy is None:
        reasons.append("active_policy_missing")
    if require_evaluation and evaluation is None:
        reasons.append("evaluation_missing")
    if evaluation is not None and snapshot is not None:
        if evaluation.get("snapshot_id") != snapshot.get("id"):
            reasons.append("snapshot_mismatch")
    if evaluation is not None and active_policy is not None:
        if evaluation.get("policy_version_id") != active_policy.get("id"):
            reasons.append("policy_version_mismatch")
    return {
        "is_current": not reasons,
        "reasons": reasons,
        "max_snapshot_age_seconds": MAX_SNAPSHOT_AGE_SECONDS,
        "snapshot_age_seconds": snapshot_age_seconds,
        "evaluation_snapshot_id": evaluation.get("snapshot_id")
        if evaluation is not None
        else None,
        "current_snapshot_id": snapshot.get("id") if snapshot is not None else None,
        "evaluation_policy_version_id": evaluation.get("policy_version_id")
        if evaluation is not None
        else None,
        "active_policy_version_id": active_policy.get("id")
        if active_policy is not None
        else None,
    }
