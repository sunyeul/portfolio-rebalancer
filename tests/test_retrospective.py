from datetime import datetime, timezone

import pytest

from integrations.toss.observation import (
    NormalizedCash,
    NormalizedHolding,
    NormalizedSnapshot,
    SyncState,
)
from services.retrospective import (
    RetrospectiveError,
    confirm,
    eligible,
    preview,
    show,
    start_case,
)
from storage.account_observation_store import insert_snapshot
from storage.database import initialize_database
from storage.evaluation_store import insert_evaluation_run
from storage.performance_store import (
    create_baseline,
    latest_performance_run_for_snapshots,
    refresh_performance,
)
from storage.policy_store import get_active_policy
from storage.retrospective_store import create_case


def _snapshot(
    *, fingerprint: str, synced_at: str, invested: float = 90.0,
    account_alias: str = "toss-brokerage",
):
    return NormalizedSnapshot(
        account_alias=account_alias,
        sync_started_at=synced_at,
        synced_at=synced_at,
        state=SyncState.COMPLETE,
        holdings=(
            NormalizedHolding(
                symbol="AAA",
                name="AAA",
                market_country="US",
                currency="USD",
                quantity=1.0,
                last_price=invested,
                average_purchase_price=80.0,
                market_value_native=invested,
                market_value_krw=invested,
                cost_native=80.0,
                cost_krw=80.0,
                profit_loss_native=invested - 80.0,
                profit_loss_krw=invested - 80.0,
                daily_profit_loss_native=0.0,
                daily_profit_loss_krw=0.0,
            ),
        ),
        cash=(NormalizedCash("KRW", 10.0, 10.0),),
        fx_rate=None,
        orders=(),
        source_timestamps={},
        data_quality={"issues": []},
        reconciliation={"holdings": {"all_within_tolerance": True}},
        total_value_krw=invested + 10.0,
        invested_value_krw=invested,
        cash_value_krw=10.0,
        fingerprint=fingerprint,
    )


def _evaluation(snapshot_id: int, policy_id: int, fingerprint: str):
    queue_item = {
        "kind": "cash",
        "identity": "cash_reserve",
        "status": "Review",
        "queue_class": "adjustment",
        "current": 0.10,
        "target": 0.15,
        "minimum": 0.10,
        "maximum": 0.20,
        "triggers": ["cash_reserve_out_of_range"],
        "suggestion": {"code": "review_increase_regular_purchase_pace"},
        "evidence_refs": {"snapshot_id": snapshot_id, "policy_version_id": policy_id},
    }
    return {
        "account_alias": "toss-brokerage",
        "snapshot_id": snapshot_id,
        "performance_run_id": None,
        "policy_version_id": policy_id,
        "source_fingerprint": fingerprint,
        "performance_fingerprint": None,
        "policy_hash": "policy",
        "engine_version": "phase5-v2",
        "state": "complete",
        "result": {
            "engine_version": "phase5-v2",
            "source": {},
            "allocation_state": "complete",
            "account": {},
            "layers": [],
            "instruments": [],
            "review_queue": [queue_item],
        },
        "market_evidence_fingerprint": "market",
        "market_evidence": {},
        "evaluation_fingerprint": fingerprint,
    }


@pytest.fixture()
def database(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "retrospective.sqlite3"))
    initialize_database()


def test_eligible_and_start_require_current_nonblocking_item(database):
    now = datetime.now(timezone.utc).isoformat()
    snapshot = insert_snapshot(_snapshot(fingerprint="current", synced_at=now))
    active = get_active_policy()
    evaluation = _evaluation(snapshot["id"], active["id"], "current-evaluation")
    evaluation["policy_hash"] = active["policy_hash"]
    persisted = insert_evaluation_run(evaluation)

    assert eligible()["evaluation_run_id"] == persisted["id"]
    case = start_case(
        kind="cash", identity="cash_reserve", disposition="adopted"
    )
    assert case["evaluation_run_id"] == persisted["id"]
    assert start_case(
        kind="cash", identity="cash_reserve", disposition="adopted"
    )["id"] == case["id"]
    with pytest.raises(RetrospectiveError, match="conflict"):
        start_case(kind="cash", identity="cash_reserve", disposition="declined", note="not now")


def test_preview_and_confirm_preserve_evidence_and_revision(database):
    baseline = insert_snapshot(
        _snapshot(fingerprint="baseline", synced_at="2025-01-01T00:00:00+00:00")
    )
    create_baseline(baseline["id"], 90.0)
    outcome = insert_snapshot(
        _snapshot(fingerprint="outcome", synced_at="2025-02-05T00:00:00+00:00", invested=95.0)
    )
    insert_snapshot(
        _snapshot(fingerprint="later", synced_at="2025-03-10T00:00:00+00:00", invested=97.0)
    )
    refresh_performance()
    active = get_active_policy()
    evaluation = _evaluation(baseline["id"], active["id"], "historic-evaluation")
    evaluation["policy_hash"] = active["policy_hash"]
    persisted = insert_evaluation_run(evaluation)
    case = create_case(
        account_alias="toss-brokerage",
        evaluation_run_id=persisted["id"],
        queue_item=evaluation["result"]["review_queue"][0],
        disposition="adopted",
        decision_note="",
        decided_at="2025-01-01T00:00:00+00:00",
    )

    candidate = preview(case["id"], "1m")
    assert candidate["state"] == "ready"
    assert candidate["evidence"]["observation"]["snapshot_id"] == outcome["id"]
    assert candidate["evidence"]["performance"]["period_twr"] == pytest.approx(0.05)
    assert "side" not in str(candidate["evidence"])

    first = confirm(
        case_id=case["id"],
        horizon="1m",
        evidence_fingerprint=candidate["evidence_fingerprint"],
        judgment="supported",
        execution="aligned",
        policy="maintain",
    )
    second = confirm(
        case_id=case["id"],
        horizon="1m",
        evidence_fingerprint=candidate["evidence_fingerprint"],
        judgment="mixed",
        execution="partially_aligned",
        policy="review_flag",
        note="more observations are needed",
    )
    payload = show(case["id"])
    assert (first["revision"], second["revision"]) == (1, 2)
    assert payload["latest_by_horizon"]["1m"]["id"] == second["id"]


def test_preview_uses_evaluation_anchor_not_optional_queue_references(database):
    baseline = insert_snapshot(
        _snapshot(fingerprint="anchor-baseline", synced_at="2025-01-01T00:00:00+00:00")
    )
    create_baseline(baseline["id"], 90.0)
    insert_snapshot(
        _snapshot(fingerprint="anchor-outcome", synced_at="2025-02-05T00:00:00+00:00", invested=95.0)
    )
    refresh_performance()
    active = get_active_policy()
    evaluation = _evaluation(baseline["id"], active["id"], "anchor-evaluation")
    evaluation["policy_hash"] = active["policy_hash"]
    evaluation["result"]["review_queue"][0].pop("evidence_refs")
    persisted = insert_evaluation_run(evaluation)
    case = create_case(
        account_alias="toss-brokerage",
        evaluation_run_id=persisted["id"],
        queue_item=evaluation["result"]["review_queue"][0],
        disposition="adopted",
        decision_note="",
        decided_at="2025-01-01T00:00:00+00:00",
    )

    candidate = preview(case["id"], "1m")

    assert candidate["state"] == "ready"
    assert candidate["evidence"]["baseline"]["snapshot_id"] == baseline["id"]
    assert candidate["evidence"]["baseline"]["policy_version_id"] == active["id"]


def test_partial_action_evidence_makes_preview_partial(database):
    baseline = insert_snapshot(
        _snapshot(fingerprint="partial-baseline", synced_at="2025-01-01T00:00:00+00:00")
    )
    create_baseline(baseline["id"], 90.0)
    insert_snapshot(
        _snapshot(fingerprint="partial-outcome", synced_at="2025-02-05T00:00:00+00:00", invested=95.0)
    )
    refresh_performance()
    active = get_active_policy()
    evaluation = _evaluation(baseline["id"], active["id"], "partial-evaluation")
    evaluation["policy_hash"] = active["policy_hash"]
    item = evaluation["result"]["review_queue"][0]
    item["kind"] = "layer"
    item["identity"] = "core"
    persisted = insert_evaluation_run(evaluation)
    case = create_case(
        account_alias="toss-brokerage",
        evaluation_run_id=persisted["id"],
        queue_item=item,
        disposition="adopted",
        decision_note="",
        decided_at="2025-01-01T00:00:00+00:00",
    )

    candidate = preview(case["id"], "1m")

    assert candidate["state"] == "partial"
    assert "execution_layer_mapping_unavailable" in candidate["evidence"]["data_quality"]["issues"]


def test_performance_evidence_run_is_scoped_to_account_and_both_snapshots(database):
    first_a = insert_snapshot(
        _snapshot(fingerprint="a-first", synced_at="2025-01-01T00:00:00+00:00", account_alias="account-a")
    )
    create_baseline(first_a["id"], 90.0)
    last_a = insert_snapshot(
        _snapshot(fingerprint="a-last", synced_at="2025-02-05T00:00:00+00:00", invested=95.0, account_alias="account-a")
    )
    run_a = refresh_performance("account-a")
    first_b = insert_snapshot(
        _snapshot(fingerprint="b-first", synced_at="2025-01-01T00:00:00+00:00", account_alias="account-b")
    )
    create_baseline(first_b["id"], 90.0)
    insert_snapshot(
        _snapshot(fingerprint="b-last", synced_at="2025-02-05T00:00:00+00:00", invested=99.0, account_alias="account-b")
    )
    refresh_performance("account-b")

    selected = latest_performance_run_for_snapshots(
        account_alias="account-a",
        start_snapshot_id=first_a["id"],
        end_snapshot_id=last_a["id"],
    )

    assert selected is not None
    assert selected["id"] == run_a["id"]


def test_preview_is_not_ready_before_due_and_confirm_rejects_it(database):
    now = datetime.now(timezone.utc).isoformat()
    snapshot = insert_snapshot(_snapshot(fingerprint="current", synced_at=now))
    active = get_active_policy()
    evaluation = _evaluation(snapshot["id"], active["id"], "future-evaluation")
    evaluation["policy_hash"] = active["policy_hash"]
    persisted = insert_evaluation_run(evaluation)
    case = create_case(
        account_alias="toss-brokerage",
        evaluation_run_id=persisted["id"],
        queue_item=evaluation["result"]["review_queue"][0],
        disposition="adopted",
        decision_note="",
    )

    candidate = preview(case["id"], "1m")
    assert candidate["state"] == "not_ready"
    with pytest.raises(RetrospectiveError, match="not confirmable"):
        confirm(
            case_id=case["id"], horizon="1m", evidence_fingerprint="x",
            judgment="supported", execution="aligned", policy="maintain",
        )
