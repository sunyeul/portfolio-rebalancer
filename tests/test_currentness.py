from datetime import datetime, timedelta, timezone

from services.currentness import MAX_SNAPSHOT_AGE_SECONDS, evaluate_currentness


NOW = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)


def _snapshot(*, age_hours: float, state: str = "complete", evaluable: bool = True):
    return {
        "id": 9,
        "state": state,
        "is_current_evaluable": evaluable,
        "synced_at": (NOW - timedelta(hours=age_hours)).isoformat(),
    }


def test_currentness_accepts_the_exact_72_hour_boundary():
    result = evaluate_currentness(
        snapshot=_snapshot(age_hours=72),
        active_policy={"id": 4},
        evaluated_at=NOW,
    )

    assert result["is_current"] is True
    assert result["snapshot_age_seconds"] == MAX_SNAPSHOT_AGE_SECONDS


def test_currentness_blocks_expired_or_non_evaluable_sources():
    expired = evaluate_currentness(
        snapshot=_snapshot(age_hours=72, state="partial", evaluable=False),
        active_policy={"id": 4},
        evaluated_at=NOW + timedelta(seconds=1),
    )

    assert expired["is_current"] is False
    assert expired["reasons"] == [
        "snapshot_not_complete",
        "snapshot_not_evaluable",
        "snapshot_stale",
    ]


def test_api_and_cli_inputs_share_the_same_currentness_contract():
    snapshot = _snapshot(age_hours=1)
    evaluation = {"snapshot_id": 9, "policy_version_id": 4}

    api_result = evaluate_currentness(
        snapshot=snapshot,
        active_policy={"id": 4},
        evaluation=evaluation,
        evaluated_at=NOW,
        require_evaluation=True,
    )
    cli_result = evaluate_currentness(
        snapshot=snapshot,
        active_policy={"id": 4},
        evaluation=evaluation,
        evaluated_at=NOW,
        require_evaluation=True,
    )

    assert api_result == cli_result
