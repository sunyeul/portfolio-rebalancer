import hashlib
import json
import sqlite3

import pytest
from typer.testing import CliRunner

from cli import app

from integrations.toss.observation import (
    NormalizedCash,
    NormalizedHolding,
    NormalizedSnapshot,
    SyncState,
)
from services.policy_candidate_assessment import (
    CandidateScenarioError,
    assess_candidate_history,
    normalize_candidate_scenario,
)
from services.policy_validation import validate_policy
from storage.account_observation_store import insert_snapshot
from storage.database import (
    DatabaseIntegrityError,
    connect_readonly,
    initialize_database,
)
from storage.policy_store import activate_policy


runner = CliRunner()


def _policy(core: tuple[float, float, float]) -> dict[str, object]:
    satellite_target = 1.0 - core[1]
    satellite = (1.0 - core[2], satellite_target, 1.0 - core[0])
    return {
        "cash_reserve": {"minimum": 0.03, "target": 0.05, "maximum": 0.10},
        "performance": {
            "annual_return_target": 0.1,
            "measurement": "ytd_twr",
            "minimum_history_days": 365,
        },
        "risk_review": {
            "lookback_sessions": 252,
            "minimum_history_points": 200,
            "max_data_age_days": 7,
            "max_gap_days": 7,
            "account_drawdown_review": -0.15,
            "instrument_drawdown_review": {
                "core": -0.25,
                "satellite": -0.20,
                "experiment": -0.15,
            },
        },
        "layers": {
            "core": dict(zip(("minimum", "target", "maximum"), core)),
            "satellite": dict(zip(("minimum", "target", "maximum"), satellite)),
            "experiment": {"minimum": 0.0, "target": 0.0, "maximum": 0.0},
        },
        "instruments": [
            {
                "market_country": "KR",
                "symbol": "AAA",
                "layer": "core",
                **dict(zip(("minimum", "target", "maximum"), core)),
            },
            {
                "market_country": "KR",
                "symbol": "BBB",
                "layer": "satellite",
                **dict(zip(("minimum", "target", "maximum"), satellite)),
            },
        ],
    }


def _holding(symbol: str, value: float) -> NormalizedHolding:
    return NormalizedHolding(
        symbol=symbol,
        name=symbol,
        market_country="KR",
        currency="KRW",
        quantity=value,
        last_price=1.0,
        average_purchase_price=1.0,
        market_value_native=value,
        market_value_krw=value,
        cost_native=value,
        cost_krw=value,
        profit_loss_native=0.0,
        profit_loss_krw=0.0,
        daily_profit_loss_native=0.0,
        daily_profit_loss_krw=0.0,
    )


def _snapshot(
    *,
    at: str,
    fingerprint: str,
    core: float = 80.0,
    satellite: float = 20.0,
    extra: float = 0.0,
    reconciled: bool = True,
    state: SyncState = SyncState.COMPLETE,
) -> NormalizedSnapshot:
    invested = core + satellite + extra
    cash = 10.0
    holdings = [_holding("AAA", core), _holding("BBB", satellite)]
    if extra:
        holdings.append(_holding("CCC", extra))
    return NormalizedSnapshot(
        account_alias="toss-brokerage",
        sync_started_at=at,
        synced_at=at,
        state=state,
        holdings=tuple(holdings),
        cash=(NormalizedCash("KRW", cash, cash),),
        fx_rate=None,
        orders=(),
        source_timestamps={},
        data_quality={"issues": []},
        reconciliation={"holdings": {"all_within_tolerance": reconciled}},
        total_value_krw=invested + cash,
        invested_value_krw=invested,
        cash_value_krw=cash,
        fingerprint=fingerprint,
    )


def _scenario(cost: float = 0.0) -> dict[str, object]:
    return {
        "monthly_contribution_krw": 20.0,
        "horizon_months": 6,
        "review_interval_days": 30,
        "transaction_cost_bps": cost,
    }


def _prepared_database(monkeypatch, tmp_path):
    path = tmp_path / "candidate.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))
    initialize_database()
    insert_snapshot(_snapshot(at="2026-01-01T00:00:00+00:00", fingerprint="first"))
    insert_snapshot(_snapshot(at="2026-01-09T00:00:00+00:00", fingerprint="second"))
    active = validate_policy(_policy((0.6, 0.7, 0.8)), [("KR", "AAA"), ("KR", "BBB")])
    activate_policy(active, expected_current_version=1)
    candidate = validate_policy(
        _policy((0.4, 0.5, 0.6)), [("KR", "AAA"), ("KR", "BBB")]
    )
    return path, candidate


def _assert_no_order_semantics(value: object) -> None:
    forbidden = {"status", "priority", "suggestion", "order_quantity", "price"}
    if isinstance(value, dict):
        assert not forbidden.intersection(value)
        for child in value.values():
            _assert_no_order_semantics(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_order_semantics(child)


def test_candidate_history_is_deterministic_read_only_and_reports_short_cadence(
    monkeypatch, tmp_path
):
    path, candidate = _prepared_database(monkeypatch, tmp_path)
    for state in (SyncState.PARTIAL, SyncState.STALE, SyncState.FAILED):
        insert_snapshot(
            _snapshot(
                at=f"2026-01-{10 + len(state):02d}T00:00:00+00:00",
                fingerprint=f"excluded-{state}",
                state=state,
            )
        )
    with sqlite3.connect(path) as conn:
        before_counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "ips_policy_versions",
                "ips_policy_candidates",
                "ips_evaluation_runs",
            )
        }
    before = (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_size,
        path.stat().st_mtime_ns,
    )

    first = assess_candidate_history(candidate, _scenario())
    second = assess_candidate_history(candidate, _scenario())

    with sqlite3.connect(path) as conn:
        after_counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before_counts
        }
    after = (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_size,
        path.stat().st_mtime_ns,
    )
    assert first == second
    assert before == after
    assert before_counts == after_counts
    assert first["persisted"] is False
    assert first["may_change_primary_evaluation"] is False
    assert first["state"] == "partial"
    assert first["cadence"]["state"] == "insufficient_history"
    assert first["analysis_period"]["used_snapshot_ids"] == [1, 2]
    assert {
        item["state"] for item in first["analysis_period"]["excluded_snapshots"]
    } == {
        "partial",
        "stale",
        "failed",
    }
    assert first["policy_differences"] == sorted(
        first["policy_differences"], key=lambda item: item["path"]
    )
    assert first["snapshots"][0]["candidate"]["cash"]["band_state"] == "within"
    assert first["snapshots"][0]["candidate"]["instruments"][0]["band_state"] == "above"
    assert first["snapshots"][0]["candidate"]["recovery"]["earliest_month"] == 2
    _assert_no_order_semantics(first)


def test_costs_delay_candidate_recovery(monkeypatch, tmp_path):
    _, candidate = _prepared_database(monkeypatch, tmp_path)

    free = assess_candidate_history(candidate, _scenario(0.0))
    costly = assess_candidate_history(candidate, _scenario(5_000.0))

    free_month = free["snapshots"][0]["candidate"]["recovery"]["earliest_month"]
    costly_month = costly["snapshots"][0]["candidate"]["recovery"]["earliest_month"]
    assert free_month == 2
    assert costly_month == 4


def test_incomplete_coverage_reconciliation_and_cash_only_are_not_evaluable(
    monkeypatch, tmp_path
):
    _, candidate = _prepared_database(monkeypatch, tmp_path)
    insert_snapshot(
        _snapshot(
            at="2026-02-01T00:00:00+00:00",
            fingerprint="uncovered",
            extra=5.0,
        )
    )
    insert_snapshot(
        _snapshot(
            at="2026-02-02T00:00:00+00:00",
            fingerprint="unreconciled",
            reconciled=False,
        )
    )
    insert_snapshot(
        _snapshot(
            at="2026-02-03T00:00:00+00:00",
            fingerprint="cash-only",
            core=0.0,
            satellite=0.0,
        )
    )

    result = assess_candidate_history(candidate, _scenario())
    by_id = {entry["snapshot_id"]: entry["candidate"] for entry in result["snapshots"]}

    assert by_id[3]["reason"] == "policy_coverage_incomplete"
    assert by_id[4]["reason"] == "snapshot reconciliation failed"
    assert by_id[5]["reason"] == "invested_denominator_unavailable"
    assert all(
        by_id[snapshot_id]["recovery"]["state"] == "not_evaluable"
        for snapshot_id in (3, 4, 5)
    )


def test_scenario_validation_is_strict():
    with pytest.raises(CandidateScenarioError, match="missing scenario fields"):
        normalize_candidate_scenario({})
    with pytest.raises(CandidateScenarioError, match="unknown scenario fields"):
        normalize_candidate_scenario({**_scenario(), "extra": 1})
    with pytest.raises(CandidateScenarioError, match="less than 10000"):
        normalize_candidate_scenario({**_scenario(), "transaction_cost_bps": 10_000})


def test_candidate_preview_cli_emits_one_json_object(monkeypatch, tmp_path):
    _, candidate = _prepared_database(monkeypatch, tmp_path)
    policy_file = tmp_path / "candidate.json"
    scenario_file = tmp_path / "scenario.json"
    policy_file.write_text(json.dumps(candidate), encoding="utf-8")
    scenario_file.write_text(json.dumps(_scenario()), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "inspection",
            "candidate-preview",
            "--policy-file",
            str(policy_file),
            "--scenario-file",
            str(scenario_file),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "inspection candidate-preview"
    assert payload["assessment"]["persisted"] is False
    assert payload["assessment"]["cadence"]["state"] == "insufficient_history"


def test_candidate_preview_does_not_create_a_missing_database(monkeypatch, tmp_path):
    db_path = tmp_path / "missing" / "candidate.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(db_path))
    policy_file = tmp_path / "candidate.json"
    scenario_file = tmp_path / "scenario.json"
    policy_file.write_text(json.dumps(_policy((0.4, 0.5, 0.6))), encoding="utf-8")
    scenario_file.write_text(json.dumps(_scenario()), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "inspection",
            "candidate-preview",
            "--policy-file",
            str(policy_file),
            "--scenario-file",
            str(scenario_file),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["error"]["stage"] == "persistence"
    assert not db_path.exists()


def test_active_policy_missing_returns_a_complete_read_only_contract(
    monkeypatch, tmp_path
):
    path = tmp_path / "no-active-policy.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))
    initialize_database()
    insert_snapshot(_snapshot(at="2026-01-01T00:00:00+00:00", fingerprint="only"))
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE ips_policy_versions SET superseded_at = '2026-01-01T00:00:00+00:00'"
        )

    result = assess_candidate_history(_policy((0.4, 0.5, 0.6)), _scenario())

    assert result["state"] == "not_evaluable"
    assert result["reason"] == "active_policy_missing"
    assert result["active_policy"] == {"id": None, "hash": None}
    assert result["candidate_policy"]["hash"]
    assert result["analysis_fingerprint"]
    assert result["snapshots"] == []


def test_readonly_connection_rejects_an_unsupported_schema_without_mutation(
    monkeypatch, tmp_path
):
    path = tmp_path / "unsupported.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel VALUES ('unchanged')")
        conn.execute("PRAGMA user_version = 999")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))

    with pytest.raises(DatabaseIntegrityError, match="current database schema"):
        connect_readonly()

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
