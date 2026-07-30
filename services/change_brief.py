"""Read-only differences between the two newest IPS inspection evaluations."""

from __future__ import annotations

from typing import Any, Mapping


def _identity(evaluation: Mapping[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(evaluation, Mapping):
        return None
    run_id = evaluation.get("id")
    snapshot_id = evaluation.get("snapshot_id")
    if not isinstance(run_id, int) or not isinstance(snapshot_id, int):
        return None
    return {"run_id": run_id, "snapshot_id": snapshot_id}


def _result(evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    value = evaluation.get("result")
    return value if isinstance(value, Mapping) else {}


def _queue(evaluation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = _result(evaluation).get("review_queue")
    return (
        [item for item in value if isinstance(item, Mapping)]
        if isinstance(value, list)
        else []
    )


def _queue_key(item: Mapping[str, Any]) -> tuple[str, str]:
    return str(item.get("kind") or ""), str(item.get("identity") or "")


def _suggestion_code(item: Mapping[str, Any]) -> str | None:
    suggestion = item.get("suggestion")
    return suggestion.get("code") if isinstance(suggestion, Mapping) else None


def _queue_signature(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("status"),
        item.get("priority"),
        item.get("queue_class"),
        _suggestion_code(item),
        tuple(sorted(str(trigger) for trigger in item.get("triggers", []))),
    )


def _brief_item(item: Mapping[str, Any], change: str) -> dict[str, Any]:
    return {
        "change": change,
        "kind": item.get("kind"),
        "identity": item.get("identity"),
        "status": item.get("status"),
        "priority": item.get("priority"),
        "priority_label": item.get("priority_label"),
        "queue_class": item.get("queue_class"),
        "suggestion": item.get("suggestion"),
        "triggers": list(item.get("triggers") or []),
        "verification_task": item.get("verification_task"),
    }


def _source_alert(current: Mapping[str, Any]) -> dict[str, str] | None:
    result = _result(current)
    source = result.get("source")
    source_state = source.get("state") if isinstance(source, Mapping) else "unknown"
    if source_state != "complete":
        return {
            "state": str(source_state),
            "message": "최신 Toss 원천이 complete 상태가 아닙니다.",
        }
    if result.get("allocation_state") == "not_evaluable":
        return {
            "state": "not_evaluable",
            "message": "최신 Toss 원천으로 비중 조정 판단을 평가할 수 없습니다.",
        }
    return None


def build_change_brief(
    current: Mapping[str, Any] | None,
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compare persisted queue evidence without reclassifying any item."""
    if not isinstance(current, Mapping):
        return {
            "state": "no_evaluation",
            "current": None,
            "previous": None,
            "changes": [],
            "source_alert": None,
        }
    if not isinstance(previous, Mapping):
        return {
            "state": "baseline",
            "current": _identity(current),
            "previous": None,
            "changes": [],
            "source_alert": _source_alert(current),
        }

    current_queue = _queue(current)
    previous_queue = _queue(previous)
    previous_by_key = {_queue_key(item): item for item in previous_queue}
    current_keys = {_queue_key(item) for item in current_queue}
    changes: list[dict[str, Any]] = []
    for item in current_queue:
        prior = previous_by_key.get(_queue_key(item))
        if prior is None:
            changes.append(_brief_item(item, "new"))
        elif _queue_signature(item) != _queue_signature(prior):
            changes.append(_brief_item(item, "changed"))
    for item in previous_queue:
        if _queue_key(item) not in current_keys:
            changes.append(_brief_item(item, "resolved"))
    return {
        "state": "changes",
        "current": _identity(current),
        "previous": _identity(previous),
        "changes": changes,
        "source_alert": _source_alert(current),
    }
