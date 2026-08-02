"""Explicitly separate unavailable policy-candidate analysis from IPS results."""

from __future__ import annotations

from typing import Any, Mapping


def unavailable_policy_candidate_assessment(
    active_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Expose the reduced-core candidate boundary without inventing evidence."""
    return {
        "state": "not_implemented",
        "reason": "market_candidate_evidence_unavailable",
        "active_policy_version_id": active_policy.get("id")
        if active_policy is not None
        else None,
        "candidate": None,
        "feasibility": None,
        "may_change_primary_evaluation": False,
        "verification_task": (
            "완전한 Toss 시장·기술 근거와 후보 정책의 실행 가능성 검증을 제공하기 전에는 "
            "정책 재검토 후보를 만들지 않습니다."
        ),
    }
