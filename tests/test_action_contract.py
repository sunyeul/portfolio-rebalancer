from services.action_contract import (
    PRIORITY_LABELS,
    SUGGESTION_LABELS,
    priority_rank,
    suggestion,
    unit_suggestion,
)


def test_priority_labels_and_suggestion_vocabulary_are_closed_and_exact():
    assert PRIORITY_LABELS == {
        "P1": "다음 정기매수 전",
        "P2": "이번 월간 점검",
        "P3": "다음 정기매수 배분 반영",
        "P4": "관찰 유지",
    }
    assert SUGGESTION_LABELS == {
        "review_increase_regular_purchase_pace": "향후 정기매수 속도 확대 검토",
        "review_reduce_or_pause_regular_purchase_pace": "향후 정기매수 속도 축소·중단 검토",
        "review_increase_regular_purchase_allocation": "향후 정기매수 배분 확대 검토",
        "review_overweight_normalization": "정기매수 축소·중단 우선 후 초과비중 정상화 검토",
        "review_thesis_or_constraints": "논지와 제약 조건 검토",
        "inspect_exceptional_intervention": "예외적 개입 가능성 검토",
        "hold_and_observe": "관찰 유지",
    }
    assert priority_rank("P1") < priority_rank("P2") < priority_rank("P3") < priority_rank("P4")


def test_action_precedence_requires_same_instrument_broken_thesis_and_maximum():
    priority, selected = unit_suggestion(
        kind="instrument",
        status="Action",
        current=0.31,
        minimum=0.1,
        maximum=0.3,
        triggers=["broken_thesis_and_hard_maximum_breach"],
        thesis_status="broken",
    )
    assert priority == "P1"
    assert selected == suggestion("inspect_exceptional_intervention")


def test_cash_and_allocation_precedence_use_policy_review_language():
    assert unit_suggestion(
        kind="cash",
        status="Review",
        current=0.05,
        minimum=0.10,
        maximum=0.20,
        triggers=["cash_reserve_out_of_range"],
    ) == ("P1", suggestion("review_reduce_or_pause_regular_purchase_pace"))
    assert unit_suggestion(
        kind="instrument",
        status="Review",
        current=0.31,
        minimum=0.10,
        maximum=0.30,
        triggers=["instrument_out_of_range", "overlap_review"],
        thesis_status="valid",
    ) == ("P2", suggestion("review_overweight_normalization"))
    assert unit_suggestion(
        kind="instrument",
        status="Review",
        current=0.05,
        minimum=0.10,
        maximum=0.30,
        triggers=["instrument_out_of_range"],
        thesis_status="valid",
        eligible_for_increase=True,
    ) == ("P3", suggestion("review_increase_regular_purchase_allocation"))
