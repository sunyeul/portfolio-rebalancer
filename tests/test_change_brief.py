from services.change_brief import build_change_brief


def _queue_item(identity, status="Review", priority="P2", triggers=None):
    return {
        "kind": "instrument",
        "identity": identity,
        "status": status,
        "priority": priority,
        "priority_label": "이번 월간 점검",
        "queue_class": "adjustment",
        "suggestion": {"code": "hold_and_observe", "label": "관찰 유지"},
        "triggers": triggers or ["out_of_range"],
        "verification_task": f"{identity} 근거를 확인합니다.",
    }


def _evaluation(run_id, snapshot_id, queue, *, source_state="complete"):
    return {
        "id": run_id,
        "snapshot_id": snapshot_id,
        "result": {
            "source": {"state": source_state},
            "allocation_state": "complete",
            "review_queue": queue,
        },
    }


def test_change_brief_reports_new_changed_and_resolved_queue_items_in_source_order():
    previous = _evaluation(
        10,
        5,
        [_queue_item("US/AAA", status="Watch", priority="P4"), _queue_item("US/CCC")],
    )
    current = _evaluation(
        11,
        6,
        [_queue_item("US/AAA"), _queue_item("US/BBB")],
    )

    brief = build_change_brief(current, previous)

    assert brief["state"] == "changes"
    assert brief["current"] == {"run_id": 11, "snapshot_id": 6}
    assert brief["previous"] == {"run_id": 10, "snapshot_id": 5}
    assert [(item["change"], item["identity"]) for item in brief["changes"]] == [
        ("changed", "US/AAA"),
        ("new", "US/BBB"),
        ("resolved", "US/CCC"),
    ]
    assert brief["changes"][0]["status"] == "Review"
    assert brief["changes"][2]["status"] == "Review"
    assert brief["source_alert"] is None


def test_change_brief_marks_baseline_and_current_source_problem_without_trading_direction():
    current = _evaluation(11, 6, [], source_state="failed")

    brief = build_change_brief(current, None)

    assert brief["state"] == "baseline"
    assert brief["previous"] is None
    assert brief["changes"] == []
    assert brief["source_alert"] == {
        "state": "failed",
        "message": "최신 Toss 원천이 complete 상태가 아닙니다.",
    }
    assert "buy" not in str(brief).lower()
    assert "sell" not in str(brief).lower()
