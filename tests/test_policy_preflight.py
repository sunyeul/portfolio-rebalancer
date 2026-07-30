from services.policy_preflight import build_policy_preflight


def test_policy_preflight_asks_source_question_first_when_current_allocation_is_blocked():
    policy = {
        "id": 4,
        "policy": {"layers": {"core": {}, "satellite": {}, "experiment": {}}},
    }
    evaluation = {
        "id": 11,
        "snapshot_id": 6,
        "result": {
            "allocation_state": "not_evaluable",
            "allocation_reason": "source_not_current_evaluable",
        },
    }

    preflight = build_policy_preflight(policy, evaluation)

    assert preflight["state"] == "source_verification_required"
    assert preflight["policy_version_id"] == 4
    assert preflight["evaluation"] == {"run_id": 11, "snapshot_id": 6}
    assert preflight["questions"][0] == {
        "id": "source_verification",
        "title": "현재 원천을 먼저 확인할까요?",
        "prompt": "현재 Toss 원천으로 비중 판단을 평가할 수 없습니다. 최근 동기화·조정 상태와 정책 커버리지를 확인하세요.",
        "required": True,
    }
    assert [item["id"] for item in preflight["questions"][1:]] == [
        "change_scope",
        "objective",
        "duration",
        "evidence",
        "reversal_condition",
    ]


def test_policy_preflight_never_persists_or_applies_answers():
    policy = {"id": 4, "policy": {"layers": {"core": {}}}}

    preflight = build_policy_preflight(policy, None)

    assert preflight["state"] == "ready"
    assert preflight["evaluation"] is None
    assert "answers" not in preflight
    assert "activate" not in str(preflight).lower()


def test_policy_preflight_requires_current_evaluation_before_policy_reasoning():
    policy = {"id": 4, "policy": {"layers": {"core": {}}}}
    evaluation = {
        "id": 11,
        "snapshot_id": 6,
        "result": {"allocation_state": "complete"},
    }

    preflight = build_policy_preflight(policy, evaluation, evaluation_is_current=False)

    assert preflight["state"] == "source_verification_required"
    assert preflight["questions"][0]["id"] == "source_verification"
    assert "최신 Toss 스냅샷" in preflight["questions"][0]["prompt"]
