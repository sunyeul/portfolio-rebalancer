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
        "review_increase_regular_purchase_pace": "현금 비중 과다: 향후 정기매수 속도 확대 검토",
        "review_reduce_or_pause_regular_purchase_pace": "현금 비중 확보: 향후 정기매수 축소·중단 검토",
        "review_increase_regular_purchase_allocation": "비중 확대: 향후 정기매수 배분 확대 검토",
        "review_overweight_normalization": "초과비중 완화: 향후 정기매수 축소·중단 검토",
        "inspect_exceptional_intervention": "예외 개입 가능성 점검",
        "hold_and_observe": "관찰 유지",
    }
    assert priority_rank("P1") < priority_rank("P2") < priority_rank("P3") < priority_rank("P4")


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
        triggers=["instrument_out_of_range"],
    ) == ("P2", suggestion("review_overweight_normalization"))
    assert unit_suggestion(
        kind="instrument",
        status="Review",
        current=0.05,
        minimum=0.10,
        maximum=0.30,
        triggers=["instrument_out_of_range"],
        eligible_for_increase=True,
    ) == ("P3", suggestion("review_increase_regular_purchase_allocation"))
