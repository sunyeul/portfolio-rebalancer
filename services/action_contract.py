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
    "review_increase_regular_purchase_pace": "현금 비중 과다: 향후 정기매수 속도 확대 검토",
    "review_reduce_or_pause_regular_purchase_pace": "현금 비중 확보: 향후 정기매수 축소·중단 검토",
    "review_increase_regular_purchase_allocation": "비중 확대: 향후 정기매수 배분 확대 검토",
    "review_overweight_normalization": "초과비중 완화: 향후 정기매수 축소·중단 검토",
    "inspect_exceptional_intervention": "예외 개입 가능성 점검",
    "hold_and_observe": "관찰 유지",
}

SUGGESTION = {
    code: {"code": code, "label": label} for code, label in SUGGESTION_LABELS.items()
}


def suggestion(code: str) -> dict[str, str]:
    """Return a copy of one closed-vocabulary suggestion object."""
    if code not in SUGGESTION:
        raise ValueError(f"unknown inspection suggestion: {code}")
    return dict(SUGGESTION[code])


def priority_label(priority: str | None) -> str | None:
    return PRIORITY_LABELS.get(priority) if priority else None


def unit_suggestion(
    *,
    kind: str,
    status: str,
    current: float | None,
    minimum: float | None,
    maximum: float | None,
    triggers: list[str],
    eligible_for_increase: bool = False,
) -> tuple[str, dict[str, str]]:
    """Apply the documented precedence and return ``(priority, suggestion)``."""
    if status == "Action":
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
    if status == "Review":
        return "P2", suggestion("hold_and_observe")
    return "P4", suggestion("hold_and_observe")


def attach_decision(
    item: Mapping[str, Any],
    *,
    eligible_for_increase: bool = False,
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
        eligible_for_increase=eligible_for_increase,
    )
    values["priority"] = priority
    values["priority_label"] = priority_label(priority)
    values["suggestion"] = selected
    return values


def queue_class(
    priority: str | None,
    *,
    blocking: bool = False,
    suggestion_code: str | None = None,
) -> str:
    if blocking:
        return "blocking"
    if suggestion_code == "hold_and_observe":
        return "observation"
    return "adjustment" if priority in {"P1", "P2", "P3"} else "observation"


def priority_rank(priority: str | None) -> int:
    return {"P1": 1, "P2": 2, "P3": 3, "P4": 4}.get(priority, 99)
