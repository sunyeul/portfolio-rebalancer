"""Single source of truth for the Phase 5 allocation-action vocabulary.

The contract intentionally describes inspection policy changes rather than
orders.  Consumers must not infer an order side, quantity, price, or execution
flag from these values.
"""

from __future__ import annotations

from typing import Any, Mapping


PRIORITY_LABELS = {
    "P1": "다음 정기매수 전",
    "P2": "이번 월간 점검",
    "P3": "다음 정기매수 배분 반영",
    "P4": "관찰 유지",
}

SUGGESTION_LABELS = {
    "review_increase_regular_purchase_pace": "향후 정기매수 속도 확대 검토",
    "review_reduce_or_pause_regular_purchase_pace": "향후 정기매수 속도 축소·중단 검토",
    "review_increase_regular_purchase_allocation": "향후 정기매수 배분 확대 검토",
    "review_overweight_normalization": "정기매수 축소·중단 우선 후 초과비중 정상화 검토",
    "review_thesis_or_constraints": "논지와 제약 조건 검토",
    "inspect_exceptional_intervention": "예외적 개입 가능성 검토",
    "hold_and_observe": "관찰 유지",
}

SUGGESTION = {code: {"code": code, "label": label} for code, label in SUGGESTION_LABELS.items()}


def suggestion(code: str) -> dict[str, str]:
    """Return a copy of one closed-vocabulary suggestion object."""
    if code not in SUGGESTION:
        raise ValueError(f"unknown inspection suggestion: {code}")
    return dict(SUGGESTION[code])


def priority_label(priority: str | None) -> str | None:
    return PRIORITY_LABELS.get(priority) if priority else None


def unit_priority(
    *,
    kind: str,
    status: str,
    current: float | None,
    minimum: float | None,
    maximum: float | None,
    triggers: list[str],
    thesis_status: str | None = None,
    eligible_for_increase: bool = False,
) -> str:
    """Choose the nearest relative checkpoint for one evaluable unit."""
    if status == "Action" or "broken_thesis_and_hard_maximum_breach" in triggers:
        return "P1"
    if kind == "cash":
        if current is not None and minimum is not None and current < minimum:
            return "P1"
        if current is not None and maximum is not None and current > maximum:
            return "P2"
        return "P4"
    if maximum is not None and current is not None and current > maximum:
        return "P2"
    if (
        minimum is not None
        and current is not None
        and current < minimum
        and eligible_for_increase
    ):
        return "P3"
    if thesis_status in {"unknown", "watch", "broken"} or any(
        trigger.endswith("_review") or trigger.endswith("_unknown")
        for trigger in triggers
    ):
        return "P2"
    return "P4"


def unit_suggestion(
    *,
    kind: str,
    status: str,
    current: float | None,
    minimum: float | None,
    maximum: float | None,
    triggers: list[str],
    thesis_status: str | None = None,
    eligible_for_increase: bool = False,
) -> tuple[str, dict[str, str]]:
    """Apply the documented precedence and return ``(priority, suggestion)``."""
    if status == "Action" or "broken_thesis_and_hard_maximum_breach" in triggers:
        return "P1", suggestion("inspect_exceptional_intervention")
    if kind == "cash":
        if current is not None and minimum is not None and current < minimum:
            return "P1", suggestion("review_reduce_or_pause_regular_purchase_pace")
        if current is not None and maximum is not None and current > maximum:
            return "P2", suggestion("review_increase_regular_purchase_pace")
    if maximum is not None and current is not None and current > maximum:
        return "P2", suggestion("review_overweight_normalization")
    if (
        minimum is not None
        and current is not None
        and current < minimum
        and eligible_for_increase
    ):
        return "P3", suggestion("review_increase_regular_purchase_allocation")
    if thesis_status in {"unknown", "watch", "broken"} or any(
        trigger.endswith("_review") or trigger.endswith("_unknown")
        for trigger in triggers
    ):
        return "P2", suggestion("review_thesis_or_constraints")
    return "P4", suggestion("hold_and_observe")


def attach_decision(
    item: Mapping[str, Any],
    *,
    eligible_for_increase: bool = False,
    thesis_status: str | None = None,
) -> dict[str, Any]:
    """Add backend-owned priority and suggestion fields to an evaluable unit."""
    values = dict(item)
    priority, selected = unit_suggestion(
        kind=str(values.get("kind", "")),
        status=str(values.get("status", "OK")),
        current=values.get("current"),
        minimum=values.get("minimum"),
        maximum=values.get("maximum"),
        triggers=list(values.get("triggers") or []),
        thesis_status=thesis_status,
        eligible_for_increase=eligible_for_increase,
    )
    values["priority"] = priority
    values["priority_label"] = priority_label(priority)
    values["suggestion"] = selected
    return values


def queue_class(priority: str | None, *, blocking: bool = False) -> str:
    if blocking:
        return "blocking"
    return "adjustment" if priority in {"P1", "P2", "P3"} else "observation"


def priority_rank(priority: str | None) -> int:
    return {"P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(priority, 99)
