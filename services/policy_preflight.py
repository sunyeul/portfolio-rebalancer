"""Read-only questions required before preparing an IPS policy change."""

from __future__ import annotations

from typing import Any, Mapping


def _evaluation_identity(evaluation: Mapping[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(evaluation, Mapping):
        return None
    run_id = evaluation.get("id")
    snapshot_id = evaluation.get("snapshot_id")
    if not isinstance(run_id, int) or not isinstance(snapshot_id, int):
        return None
    return {"run_id": run_id, "snapshot_id": snapshot_id}


def build_policy_preflight(
    active_policy: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None,
    *,
    evaluation_is_current: bool = True,
) -> dict[str, Any]:
    """Return questions only; never persist answers or modify active policy."""
    if not isinstance(active_policy, Mapping):
        return {
            "state": "policy_missing",
            "policy_version_id": None,
            "evaluation": _evaluation_identity(evaluation),
            "questions": [
                {
                    "id": "active_policy",
                    "title": "활성 IPS 정책이 있나요?",
                    "prompt": "정책 변경 전 활성 정책과 대상 계좌를 먼저 확인하세요.",
                    "required": True,
                }
            ],
        }

    result = (
        evaluation.get("result") if isinstance(evaluation, Mapping) else None
    ) or {}
    allocation_blocked = not evaluation_is_current or (
        isinstance(result, Mapping)
        and result.get("allocation_state") == "not_evaluable"
    )
    questions: list[dict[str, Any]] = []
    if allocation_blocked:
        prompt = (
            "저장된 평가가 최신 Toss 스냅샷과 활성 정책에 맞는지 확인한 뒤 "
            "inspection run을 다시 실행하세요."
            if not evaluation_is_current
            else "현재 Toss 원천으로 비중 판단을 평가할 수 없습니다. 최근 동기화·조정 상태와 정책 커버리지를 확인하세요."
        )
        questions.append(
            {
                "id": "source_verification",
                "title": "현재 원천을 먼저 확인할까요?",
                "prompt": prompt,
                "required": True,
            }
        )
    questions.extend(
        [
            {
                "id": "change_scope",
                "title": "무엇을 바꾸려 하나요?",
                "prompt": "현금, core, satellite, experiment, 또는 특정 종목 중 변경 범위를 적으세요.",
                "required": True,
            },
            {
                "id": "objective",
                "title": "무엇을 지키거나 개선하려 하나요?",
                "prompt": "수익 목표가 아니라 역할, 위험 한도, 현금 여력처럼 유지할 제약을 적으세요.",
                "required": True,
            },
            {
                "id": "duration",
                "title": "이 변경은 언제까지 유효한가요?",
                "prompt": "일시적 관찰인지 다음 정책 검토까지의 변경인지 적으세요.",
                "required": True,
            },
            {
                "id": "evidence",
                "title": "어떤 근거가 이 변경을 뒷받침하나요?",
                "prompt": "현재 Toss 평가, 활성 정책, 또는 명시된 분석 가정을 구분해 적으세요.",
                "required": True,
            },
            {
                "id": "reversal_condition",
                "title": "어떤 증거가 결론을 바꿀까요?",
                "prompt": "변경을 보류하거나 다시 검토하게 만들 확인 조건을 적으세요.",
                "required": True,
            },
        ]
    )
    return {
        "state": "source_verification_required" if allocation_blocked else "ready",
        "policy_version_id": active_policy.get("id"),
        "evaluation": _evaluation_identity(evaluation),
        "questions": questions,
    }
