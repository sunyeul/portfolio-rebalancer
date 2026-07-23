"""Persistence helpers for Phase 2 account performance projections."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from services.account_performance import (
    build_projection,
)
from storage.account_observation_store import list_complete_snapshots
from storage.database import connect


DECISION_CLASSIFICATIONS = frozenset(
    {
        "external_deposit",
        "external_withdrawal",
        "investment_income",
        "fee_or_tax",
        "internal_fx",
        "rounding_or_false_positive",
        "other_non_external",
    }
)


class PerformanceStorageError(RuntimeError):
    """Raised when a Phase 2 persistence operation is invalid."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decimal(value: Any, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise PerformanceStorageError(f"invalid {field}") from exc
    if not parsed.is_finite():
        raise PerformanceStorageError(f"invalid {field}")
    return parsed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _confirmation_fingerprint(
    account_alias: str, snapshot_id: int, principal_krw: Decimal
) -> str:
    payload = f"{account_alias}|{snapshot_id}|{principal_krw.normalize()}".encode()
    return hashlib.sha256(payload).hexdigest()


def preview_baseline(snapshot_id: int) -> dict[str, Any] | None:
    """Return safe baseline confirmation data for one account snapshot."""
    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, account_alias, synced_at, state, is_current_evaluable,
                   total_value_krw, invested_value_krw, cash_value_krw
            FROM broker_account_snapshots
            WHERE id = ?
            """,
            (snapshot_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def create_baseline(
    snapshot_id: int,
    expected_principal_krw: float,
    *,
    confirmed_at: str | None = None,
) -> dict[str, Any]:
    """Create one immutable tracking baseline after exact confirmation."""
    expected = _decimal(expected_principal_krw, "expected_principal_krw")
    if expected < 0:
        raise PerformanceStorageError("expected_principal_krw must be nonnegative")
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM broker_account_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise PerformanceStorageError(f"snapshot_id={snapshot_id} not found")
        if row["state"] != "complete" or not row["is_current_evaluable"]:
            raise PerformanceStorageError(
                "baseline requires a current complete snapshot"
            )
        if row["total_value_krw"] is None:
            raise PerformanceStorageError("baseline snapshot has no total value")
        actual = _decimal(row["total_value_krw"], "snapshot.total_value_krw")
        if abs(actual - expected) > Decimal("0.01"):
            raise PerformanceStorageError("expected principal does not match snapshot")
        existing = conn.execute(
            "SELECT id FROM account_tracking_baselines WHERE account_alias = ?",
            (row["account_alias"],),
        ).fetchone()
        if existing is not None:
            raise PerformanceStorageError("tracking baseline already exists")
        fingerprint = _confirmation_fingerprint(
            str(row["account_alias"]), snapshot_id, actual
        )
        fx_row = conn.execute(
            "SELECT rate FROM broker_exchange_rates WHERE snapshot_id = ? LIMIT 1",
            (snapshot_id,),
        ).fetchone()
        cursor = conn.execute(
            """
            INSERT INTO account_tracking_baselines (
                account_alias, baseline_snapshot_id, tracking_started_at,
                initial_principal_krw, baseline_fx_rate, confirmed_at,
                confirmation_fingerprint
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["account_alias"],
                snapshot_id,
                row["synced_at"],
                float(actual),
                fx_row["rate"] if fx_row is not None else None,
                confirmed_at or _now_iso(),
                fingerprint,
            ),
        )
        baseline_id = int(cursor.lastrowid)
        return _row_to_baseline(conn, baseline_id)


def _row_to_baseline(conn: sqlite3.Connection, baseline_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM account_tracking_baselines WHERE id = ?", (baseline_id,)
    ).fetchone()
    if row is None:
        raise PerformanceStorageError(f"baseline_id={baseline_id} not found")
    return dict(row)


def get_baseline(account_alias: str = "toss-brokerage") -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM account_tracking_baselines
            WHERE account_alias = ?
            ORDER BY id DESC LIMIT 1
            """,
            (account_alias,),
        ).fetchone()
        return dict(row) if row is not None else None


def insert_cash_flow_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Insert one immutable candidate or return its fingerprint match."""
    required = (
        "baseline_id",
        "from_snapshot_id",
        "to_snapshot_id",
        "currency",
        "observed_delta_native",
        "explained_trade_delta_native",
        "residual_native",
        "residual_krw",
        "materiality_threshold_krw",
        "bridge_basis",
        "candidate_fingerprint",
    )
    missing = [key for key in required if key not in candidate]
    if missing:
        raise PerformanceStorageError(f"candidate missing fields: {', '.join(missing)}")
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM account_cash_flow_candidates WHERE candidate_fingerprint = ?",
            (candidate["candidate_fingerprint"],),
        ).fetchone()
        if existing is None:
            cursor = conn.execute(
                """
                INSERT INTO account_cash_flow_candidates (
                    baseline_id, from_snapshot_id, to_snapshot_id, currency,
                    observed_delta_native, explained_trade_delta_native,
                    residual_native, residual_krw, materiality_threshold_krw,
                    bridge_basis, candidate_fingerprint
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(candidate[key] for key in required),
            )
            candidate_id = int(cursor.lastrowid)
        else:
            candidate_id = int(existing["id"])
        row = conn.execute(
            "SELECT * FROM account_cash_flow_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
        return dict(row)


def list_cash_flow_candidates(baseline_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM account_cash_flow_candidates
            WHERE baseline_id = ?
            ORDER BY from_snapshot_id, to_snapshot_id, currency, id
            """,
            (baseline_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def append_cash_flow_decision(
    candidate_id: int,
    *,
    classification: str,
    confirmed_amount_native: float | None = None,
    confirmed_amount_krw: float | None = None,
    effective_at: str | None = None,
    note: str = "",
) -> dict[str, Any]:
    if classification not in DECISION_CLASSIFICATIONS:
        raise PerformanceStorageError("invalid cash-flow classification")
    with connect() as conn:
        if (
            conn.execute(
                "SELECT id FROM account_cash_flow_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            is None
        ):
            raise PerformanceStorageError(f"candidate_id={candidate_id} not found")
        cursor = conn.execute(
            """
            INSERT INTO account_cash_flow_decisions (
                candidate_id, classification, confirmed_amount_native,
                confirmed_amount_krw, effective_at, note
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                classification,
                confirmed_amount_native,
                confirmed_amount_krw,
                effective_at,
                note,
            ),
        )
        row = conn.execute(
            "SELECT * FROM account_cash_flow_decisions WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        return dict(row)


def latest_cash_flow_decisions(
    candidate_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    if not candidate_ids:
        return {}
    placeholders = ",".join("?" for _ in candidate_ids)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT d.* FROM account_cash_flow_decisions d
            JOIN (
                SELECT candidate_id, MAX(id) AS id
                FROM account_cash_flow_decisions
                WHERE candidate_id IN ({placeholders})
                GROUP BY candidate_id
            ) latest ON latest.id = d.id
            """,
            tuple(candidate_ids),
        ).fetchall()
        return {int(row["candidate_id"]): dict(row) for row in rows}


def insert_performance_run(projection: Any) -> dict[str, Any]:
    """Insert one immutable projection and hydrate its safe public shape."""
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM account_performance_runs WHERE input_fingerprint = ?",
            (projection.input_fingerprint,),
        ).fetchone()
        if existing is not None:
            return _hydrate_run(conn, int(existing["id"]))
        cursor = conn.execute(
            """
            INSERT INTO account_performance_runs (
                baseline_id, through_snapshot_id, input_fingerprint,
                engine_version, state, data_quality_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                projection.baseline_id,
                projection.through_snapshot_id,
                projection.input_fingerprint,
                projection.engine_version,
                projection.state,
                _json(projection.data_quality),
            ),
        )
        run_id = int(cursor.lastrowid)
        point_keys = (
            "snapshot_id",
            "previous_snapshot_id",
            "point_at",
            "evaluation_state",
            "evaluation_reason",
            "total_value_krw",
            "invested_value_krw",
            "cash_value_krw",
            "current_cost_basis_krw",
            "unrealized_pnl_krw",
            "investment_principal_krw",
            "cumulative_external_flow_krw",
            "account_gain_krw",
            "simple_return",
            "interval_twr",
            "segment_id",
            "segment_twr",
            "tracked_realized_pnl_krw",
            "actual_realized_pnl_krw",
            "fx_remeasurement_krw",
        )
        point_sql = f"""
            INSERT INTO account_performance_points (
                run_id, {", ".join(point_keys)}
            ) VALUES ({", ".join("?" for _ in range(len(point_keys) + 1))})
        """
        for point in projection.points:
            conn.execute(
                point_sql,
                (run_id, *(point.get(key) for key in point_keys)),
            )
        execution_keys = (
            "source_snapshot_id",
            "order_id",
            "symbol",
            "currency",
            "side",
            "filled_at",
            "settlement_date",
            "filled_quantity_native",
            "filled_amount_native",
            "commission_native",
            "tax_native",
            "actual_basis_before_native",
            "tracking_basis_before_native",
            "actual_realized_pnl_native",
            "tracking_realized_pnl_native",
            "realized_pnl_krw",
            "krw_conversion_snapshot_id",
        )
        execution_sql = f"""
            INSERT INTO account_execution_ledger (
                run_id, {", ".join(execution_keys)}
            ) VALUES ({", ".join("?" for _ in range(len(execution_keys) + 1))})
        """
        for execution in projection.executions:
            conn.execute(
                execution_sql,
                (run_id, *(execution.get(key) for key in execution_keys)),
            )
        return _hydrate_run(conn, run_id)


def refresh_performance(account_alias: str = "toss-brokerage") -> dict[str, Any]:
    """Build and persist one immutable performance projection."""
    baseline = get_baseline(account_alias)
    if baseline is None:
        raise PerformanceStorageError("tracking baseline is not configured")
    snapshots = list_complete_snapshots(account_alias)
    if not snapshots:
        raise PerformanceStorageError("no complete account snapshots available")
    first_projection = build_projection(baseline, snapshots, {})
    candidate_ids: dict[str, int] = {}
    for candidate in first_projection.candidates:
        row = insert_cash_flow_candidate(candidate.as_dict())
        candidate_ids[candidate.candidate_fingerprint] = int(row["id"])
    if candidate_ids:
        candidate_rows = list_cash_flow_candidates(int(baseline["id"]))
        decision_rows = latest_cash_flow_decisions(
            [int(row["id"]) for row in candidate_rows]
        )
    else:
        decision_rows = {}
    projection = build_projection(
        baseline,
        snapshots,
        decision_rows,
        candidate_ids,
    )
    return insert_performance_run(projection)


def _hydrate_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM account_performance_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise PerformanceStorageError(f"run_id={run_id} not found")
    points = [
        dict(point)
        for point in conn.execute(
            """
            SELECT * FROM account_performance_points
            WHERE run_id = ? ORDER BY point_at, snapshot_id
            """,
            (run_id,),
        ).fetchall()
    ]
    candidate_rows = conn.execute(
        """
        SELECT c.*,
               d.classification AS latest_classification,
               d.effective_at AS latest_effective_at
        FROM account_cash_flow_candidates c
        LEFT JOIN account_cash_flow_decisions d ON d.id = (
            SELECT MAX(id) FROM account_cash_flow_decisions
            WHERE candidate_id = c.id
        )
        WHERE c.baseline_id = ?
        ORDER BY c.from_snapshot_id, c.to_snapshot_id, c.currency, c.id
        """,
        (row["baseline_id"],),
    ).fetchall()
    return {
        "id": int(row["id"]),
        "baseline_id": int(row["baseline_id"]),
        "through_snapshot_id": int(row["through_snapshot_id"]),
        "input_fingerprint": row["input_fingerprint"],
        "engine_version": row["engine_version"],
        "state": row["state"],
        "data_quality": json.loads(row["data_quality_json"]),
        "points": points,
        "candidates": [dict(candidate) for candidate in candidate_rows],
        "execution_count": int(
            conn.execute(
                "SELECT COUNT(*) FROM account_execution_ledger WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
        ),
    }


def get_performance_run(run_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id FROM account_performance_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _hydrate_run(conn, run_id) if row is not None else None


def latest_performance_run(
    baseline_id: int | None = None,
    through_snapshot_id: int | None = None,
) -> dict[str, Any] | None:
    with connect() as conn:
        if through_snapshot_id is not None:
            row = conn.execute(
                """
                SELECT id FROM account_performance_runs
                WHERE through_snapshot_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (through_snapshot_id,),
            ).fetchone()
        elif baseline_id is None:
            row = conn.execute(
                "SELECT id FROM account_performance_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id FROM account_performance_runs
                WHERE baseline_id = ? ORDER BY id DESC LIMIT 1
                """,
                (baseline_id,),
            ).fetchone()
        return _hydrate_run(conn, int(row["id"])) if row is not None else None
