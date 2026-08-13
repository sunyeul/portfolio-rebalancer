"""Evidence-backed, human-confirmed IPS retrospective workflow."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from services.account_performance import canonical_executions
from services.account_projection import AccountProjectionError, build_account_projection, layer_map_from_policy
from services.currentness import evaluate_currentness
from storage.account_observation_store import list_complete_snapshots, list_snapshots
from storage.evaluation_store import (
    current_v2_result,
    get_evaluation_run,
    latest_evaluation_run,
)
from storage.performance_store import latest_performance_run_for_snapshots
from storage.policy_store import get_active_policy, get_policy_version
from storage.retrospective_store import (
    RetrospectiveStorageError,
    append_review,
    create_case,
    get_case,
    list_cases as stored_cases,
    list_reviews,
)


HORIZON_DAYS = {"1m": 30, "3m": 90, "12m": 365}
DISPOSITIONS = frozenset({"adopted", "deferred", "declined"})
JUDGMENTS = frozenset({"supported", "mixed", "challenged", "insufficient_evidence"})
EXECUTIONS = frozenset({"aligned", "partially_aligned", "not_aligned", "not_applicable", "insufficient_evidence"})
POLICIES = frozenset({"maintain", "review_flag", "insufficient_evidence"})


class RetrospectiveError(RuntimeError):
    """Raised for a safe, user-visible retrospective workflow failure."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise RetrospectiveError("invalid retrospective timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _current_evaluation(account_alias: str) -> dict[str, Any]:
    evaluation = latest_evaluation_run(account_alias)
    snapshots = list_snapshots(limit=1, account_alias=account_alias)
    currentness = evaluate_currentness(
        evaluation=evaluation,
        snapshot=snapshots[0] if snapshots else None,
        active_policy=get_active_policy(account_alias),
        require_evaluation=True,
    )
    if (
        evaluation is None
        or not current_v2_result(evaluation.get("result"))
        or not currentness["is_current"]
    ):
        raise RetrospectiveError("latest inspection evaluation is not current")
    return evaluation


def _eligible_items(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    queue = evaluation.get("result", {}).get("review_queue", [])
    return [
        dict(item)
        for item in queue
        if isinstance(item, Mapping)
        and item.get("queue_class") != "blocking"
        and isinstance(item.get("suggestion"), Mapping)
        and item.get("kind")
        and item.get("identity")
    ]


def eligible(account_alias: str = "toss-brokerage") -> dict[str, Any]:
    evaluation = _current_evaluation(account_alias)
    return {
        "evaluation_run_id": evaluation["id"],
        "snapshot_id": evaluation["snapshot_id"],
        "items": _eligible_items(evaluation),
    }


def start_case(
    *,
    kind: str,
    identity: str,
    disposition: str,
    note: str = "",
    account_alias: str = "toss-brokerage",
) -> dict[str, Any]:
    if disposition not in DISPOSITIONS:
        raise RetrospectiveError("invalid retrospective disposition")
    if disposition in {"deferred", "declined"} and not note.strip():
        raise RetrospectiveError("decision note is required for deferred or declined")
    evaluation = _current_evaluation(account_alias)
    item = next(
        (
            candidate
            for candidate in _eligible_items(evaluation)
            if candidate["kind"] == kind and candidate["identity"] == identity
        ),
        None,
    )
    if item is None:
        raise RetrospectiveError("eligible review queue item not found")
    try:
        return create_case(
            account_alias=account_alias,
            evaluation_run_id=int(evaluation["id"]),
            queue_item=item,
            disposition=disposition,
            decision_note=note.strip(),
        )
    except RetrospectiveStorageError as exc:
        raise RetrospectiveError(str(exc)) from exc


def _due_at(case: Mapping[str, Any], horizon: str) -> datetime:
    if horizon not in HORIZON_DAYS:
        raise RetrospectiveError("invalid retrospective horizon")
    return _parse_timestamp(case["decided_at"]) + timedelta(days=HORIZON_DAYS[horizon])


def _latest_by_horizon(reviews: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for review in reviews:
        latest.setdefault(review["horizon"], review)
    return latest


def _first_observation(case: Mapping[str, Any], due_at: datetime) -> dict[str, Any] | None:
    for snapshot in list_complete_snapshots(str(case["account_alias"])):
        if _parse_timestamp(snapshot["synced_at"]) >= due_at:
            return snapshot
    return None


def _unit_current(
    item: Mapping[str, Any], projection: Mapping[str, Any]
) -> float | None:
    kind, identity = str(item["kind"]), str(item["identity"])
    if kind == "cash":
        return _finite(projection.get("cash_weight_gross"))
    if kind == "layer":
        return _finite((projection.get("layer_weights_invested") or {}).get(identity))
    if kind == "instrument":
        for position in projection.get("positions", []):
            if f"{position.get('market_country')}/{position.get('symbol')}" == identity:
                return _finite(position.get("invested_weight"))
        return 0.0 if projection.get("invested_weights_evaluable") else None
    return None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _performance_evidence(
    account_alias: str, start_snapshot_id: int, end_snapshot_id: int
) -> dict[str, Any]:
    run = latest_performance_run_for_snapshots(
        account_alias=account_alias,
        start_snapshot_id=start_snapshot_id,
        end_snapshot_id=end_snapshot_id,
    )
    if run is None:
        return {"state": "unavailable", "run_id": None, "period_twr": None, "maximum_drawdown": None, "point_count": 0}
    points = list(run.get("points", []))
    start_index = next((index for index, point in enumerate(points) if point.get("snapshot_id") == start_snapshot_id), None)
    end_index = next((index for index, point in enumerate(points) if point.get("snapshot_id") == end_snapshot_id), None)
    if start_index is None or end_index is None or end_index < start_index:
        return {"state": "unavailable", "run_id": run["id"], "period_twr": None, "maximum_drawdown": None, "point_count": 0}
    selected = points[start_index : end_index + 1]
    intervals = selected[1:]
    values = [_finite(point.get("interval_twr")) for point in intervals]
    if any(point.get("evaluation_state") != "evaluable" for point in selected) or any(value is None or value <= -1 for value in values):
        return {"state": "unavailable", "run_id": run["id"], "period_twr": None, "maximum_drawdown": None, "point_count": len(selected)}
    curve = peak = 1.0
    drawdowns = [0.0]
    for value in values:
        curve *= 1.0 + float(value)
        peak = max(peak, curve)
        drawdowns.append(curve / peak - 1.0)
    return {"state": "complete", "run_id": run["id"], "period_twr": curve - 1.0, "maximum_drawdown": min(drawdowns), "point_count": len(selected)}


def _action_evidence(
    *,
    item: Mapping[str, Any],
    baseline: Mapping[str, Any],
    observation: Mapping[str, Any],
    before_current: float | None,
    after_current: float | None,
) -> dict[str, Any]:
    issues: list[str] = []
    try:
        executions = canonical_executions(list_complete_snapshots(str(baseline["account_alias"])))
    except Exception:
        executions = []
        issues.append("execution_evidence_unavailable")
    start, end = _parse_timestamp(baseline["synced_at"]), _parse_timestamp(observation["synced_at"])
    kind, identity = str(item["kind"]), str(item["identity"])
    if kind == "layer":
        issues.append("execution_layer_mapping_unavailable")
    count = 0
    for execution in executions:
        filled_at = execution.get("filled_at")
        if not filled_at:
            continue
        try:
            occurred = _parse_timestamp(filled_at)
        except RetrospectiveError:
            issues.append("execution_timestamp_invalid")
            continue
        relevant = (
            kind in {"cash", "account_risk"}
            or (kind == "instrument" and str(execution.get("symbol", "")).upper() == identity.rsplit("/", 1)[-1])
        )
        if relevant and start < occurred <= end:
            count += 1
    return {
        "state": "partial" if issues else "complete",
        "related_filled_order_count": count,
        "allocation_changed": (
            before_current is not None
            and after_current is not None
            and not math.isclose(before_current, after_current, abs_tol=1e-12)
        ),
        "issues": sorted(set(issues)),
    }


def preview(case_id: int, horizon: str) -> dict[str, Any]:
    case = get_case(case_id)
    if case is None:
        raise RetrospectiveError(f"case_id={case_id} not found")
    due_at = _due_at(case, horizon)
    now = _now()
    if now < due_at:
        return {"state": "not_ready", "case_id": case_id, "horizon": horizon, "due_at": due_at.isoformat(), "evidence": None, "evidence_fingerprint": None}
    observation = _first_observation(case, due_at)
    if observation is None:
        return {"state": "not_ready", "case_id": case_id, "horizon": horizon, "due_at": due_at.isoformat(), "evidence": None, "evidence_fingerprint": None}
    evaluation = get_evaluation_run(int(case["evaluation_run_id"]))
    if evaluation is None or evaluation.get("account_alias") != case["account_alias"]:
        return {"state": "blocked", "case_id": case_id, "horizon": horizon, "due_at": due_at.isoformat(), "evidence": None, "evidence_fingerprint": None, "reason": "anchored_evaluation_missing"}
    policy_version = get_policy_version(int(evaluation["policy_version_id"]), str(case["account_alias"]))
    if policy_version is None:
        return {"state": "blocked", "case_id": case_id, "horizon": horizon, "due_at": due_at.isoformat(), "evidence": None, "evidence_fingerprint": None, "reason": "anchored_policy_missing"}
    try:
        projection = build_account_projection(
            snapshot_id=int(observation["id"]),
            account_alias=str(case["account_alias"]),
            layer_map=layer_map_from_policy(policy_version["policy"]),
            require_current_evaluable=False,
        )
    except AccountProjectionError as exc:
        return {"state": "blocked", "case_id": case_id, "horizon": horizon, "due_at": due_at.isoformat(), "evidence": None, "evidence_fingerprint": None, "reason": str(exc)}
    item = case["queue_item"]
    before_current = _finite(item.get("current"))
    after_current = _unit_current(item, projection)
    target = _finite(item.get("target"))
    baseline_snapshot_id = int(evaluation["snapshot_id"])
    performance = _performance_evidence(
        str(case["account_alias"]), baseline_snapshot_id, int(observation["id"])
    )
    baseline = next((snapshot for snapshot in list_complete_snapshots(str(case["account_alias"])) if int(snapshot["id"]) == baseline_snapshot_id), None)
    if baseline is None:
        return {"state": "blocked", "case_id": case_id, "horizon": horizon, "due_at": due_at.isoformat(), "evidence": None, "evidence_fingerprint": None, "reason": "anchored_snapshot_missing"}
    evidence = {
        "schema_version": 1,
        "case_id": case_id,
        "horizon": horizon,
        "due_at": due_at.isoformat(),
        "baseline": {"evaluation_run_id": case["evaluation_run_id"], "snapshot_id": baseline_snapshot_id, "policy_version_id": policy_version["id"], "queue_item": item},
        "observation": {"snapshot_id": int(observation["id"]), "synced_at": observation["synced_at"], "lag_days": ( _parse_timestamp(observation["synced_at"]) - due_at).days},
        "allocation": {
            "state": "not_applicable" if after_current is None and before_current is None else "complete",
            "before_current": before_current,
            "after_current": after_current,
            "target": target,
            "signed_gap_before": before_current - target if before_current is not None and target is not None else None,
            "signed_gap_after": after_current - target if after_current is not None and target is not None else None,
            "absolute_gap_change": (abs(after_current - target) - abs(before_current - target)) if before_current is not None and after_current is not None and target is not None else None,
        },
        "performance": performance,
        "action_evidence": _action_evidence(item=item, baseline=baseline, observation=observation, before_current=before_current, after_current=after_current),
        "data_quality": {"issues": []},
    }
    if performance["state"] != "complete":
        evidence["data_quality"]["issues"].append("performance_evidence_unavailable")
    evidence["data_quality"]["issues"].extend(evidence["action_evidence"]["issues"])
    state = "ready" if not evidence["data_quality"]["issues"] else "partial"
    return {"state": state, "case_id": case_id, "horizon": horizon, "due_at": due_at.isoformat(), "evidence": evidence, "evidence_fingerprint": _fingerprint(evidence)}


def confirm(
    *,
    case_id: int,
    horizon: str,
    evidence_fingerprint: str,
    judgment: str,
    execution: str,
    policy: str,
    note: str = "",
) -> dict[str, Any]:
    if judgment not in JUDGMENTS or execution not in EXECUTIONS or policy not in POLICIES:
        raise RetrospectiveError("invalid retrospective assessment")
    if ({judgment, execution, policy} & {"mixed", "challenged", "partially_aligned", "not_aligned", "review_flag"}) and not note.strip():
        raise RetrospectiveError("review note is required for the selected assessment")
    candidate = preview(case_id, horizon)
    if candidate["state"] not in {"ready", "partial"}:
        raise RetrospectiveError("retrospective evidence is not confirmable")
    if candidate["evidence_fingerprint"] != evidence_fingerprint:
        raise RetrospectiveError("retrospective evidence fingerprint mismatch")
    evidence = candidate["evidence"]
    try:
        return append_review(
            case_id=case_id,
            horizon=horizon,
            due_at=candidate["due_at"],
            observation_snapshot_id=int(evidence["observation"]["snapshot_id"]),
            performance_run_id=evidence["performance"]["run_id"],
            evidence=evidence,
            evidence_fingerprint=evidence_fingerprint,
            judgment_assessment=judgment,
            execution_assessment=execution,
            policy_assessment=policy,
            review_note=note.strip(),
        )
    except RetrospectiveStorageError as exc:
        raise RetrospectiveError(str(exc)) from exc


def show(case_id: int) -> dict[str, Any]:
    case = get_case(case_id)
    if case is None:
        raise RetrospectiveError(f"case_id={case_id} not found")
    reviews = list_reviews(case_id)
    return {"case": case, "reviews": reviews, "latest_by_horizon": _latest_by_horizon(reviews)}


def list_case_summaries(state: str = "all", account_alias: str = "toss-brokerage") -> list[dict[str, Any]]:
    if state not in {"all", "due", "flagged"}:
        raise RetrospectiveError("invalid retrospective list state")
    now = _now()
    summaries = []
    for case in stored_cases(account_alias):
        reviews = _latest_by_horizon(list_reviews(int(case["id"])))
        due = [horizon for horizon in HORIZON_DAYS if now >= _due_at(case, horizon) and horizon not in reviews]
        flagged = any(review["policy_assessment"] == "review_flag" for review in reviews.values())
        if state == "due" and not due:
            continue
        if state == "flagged" and not flagged:
            continue
        summaries.append({"case": case, "due_horizons": due, "latest_reviews": reviews, "policy_review_flagged": flagged})
    return summaries
