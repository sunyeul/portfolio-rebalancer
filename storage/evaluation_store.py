"""Immutable persistence for Toss inspection evaluations."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from storage.database import connect


ENGINE_VERSION = "phase5-v2"
REQUIRED_RESULT_FIELDS = frozenset(
    {
        "engine_version",
        "source",
        "allocation_state",
        "account",
        "layers",
        "instruments",
        "review_queue",
    }
)
ALLOCATION_STATES = frozenset({"complete", "partial", "not_evaluable"})


class EvaluationStorageError(RuntimeError):
    """Raised when an evaluation run cannot be persisted or read."""


def current_v2_result(result: Any) -> bool:
    """Return whether a result is a complete current phase5-v2 payload."""
    if not isinstance(result, dict):
        return False
    if result.get("engine_version") != ENGINE_VERSION:
        return False
    if not REQUIRED_RESULT_FIELDS <= result.keys():
        return False
    return (
        result.get("allocation_state") in ALLOCATION_STATES
        and isinstance(result.get("source"), dict)
        and isinstance(result.get("account"), dict)
        and isinstance(result.get("layers"), list)
        and isinstance(result.get("instruments"), list)
        and isinstance(result.get("review_queue"), list)
    )


def _invalid_current_v2_result_field(result: Any) -> str:
    """Name the first invalid result field for a machine-readable store error."""
    if not isinstance(result, dict):
        return "result"
    for field in sorted(REQUIRED_RESULT_FIELDS - result.keys()):
        return f"result.{field}"
    if result.get("engine_version") != ENGINE_VERSION:
        return "result.engine_version"
    if result.get("allocation_state") not in ALLOCATION_STATES:
        return "result.allocation_state"
    for field, expected_type in (
        ("source", dict),
        ("account", dict),
        ("layers", list),
        ("instruments", list),
        ("review_queue", list),
    ):
        if not isinstance(result.get(field), expected_type):
            return f"result.{field}"
    return "result"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def insert_evaluation_run(evaluation: dict[str, Any]) -> dict[str, Any]:
    required = (
        "account_alias",
        "snapshot_id",
        "policy_version_id",
        "source_fingerprint",
        "policy_hash",
        "engine_version",
        "state",
        "result",
        "market_evidence_fingerprint",
        "market_evidence",
        "evaluation_fingerprint",
    )
    missing = [key for key in required if key not in evaluation]
    if missing:
        raise EvaluationStorageError(f"evaluation missing fields: {', '.join(missing)}")
    if evaluation["state"] not in {"complete", "not_evaluable", "failed"}:
        raise EvaluationStorageError("invalid evaluation state")
    if evaluation["engine_version"] != ENGINE_VERSION:
        raise EvaluationStorageError("unsupported evaluation engine_version")
    if not current_v2_result(evaluation["result"]):
        raise EvaluationStorageError(
            f"invalid phase5-v2 evaluation {_invalid_current_v2_result_field(evaluation['result'])}"
        )
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM ips_evaluation_runs WHERE evaluation_fingerprint = ?",
            (evaluation["evaluation_fingerprint"],),
        ).fetchone()
        if existing is not None:
            return get_evaluation_run(int(existing["id"]))  # type: ignore[return-value]
        cursor = conn.execute(
            """
            INSERT INTO ips_evaluation_runs (
                account_alias, snapshot_id, performance_run_id, policy_version_id,
                source_fingerprint, performance_fingerprint, policy_hash,
                engine_version, state,
                non_evaluable_reason, result_json, market_evidence_fingerprint,
                market_evidence_json, evaluation_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation["account_alias"],
                evaluation["snapshot_id"],
                evaluation.get("performance_run_id"),
                evaluation["policy_version_id"],
                evaluation["source_fingerprint"],
                evaluation.get("performance_fingerprint"),
                evaluation["policy_hash"],
                evaluation["engine_version"],
                evaluation["state"],
                evaluation.get("non_evaluable_reason"),
                _json(evaluation["result"]),
                evaluation["market_evidence_fingerprint"],
                _json(evaluation["market_evidence"]),
                evaluation["evaluation_fingerprint"],
            ),
        )
        return get_evaluation_run(int(cursor.lastrowid), conn=conn)  # type: ignore[return-value]


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["result"] = json.loads(result.pop("result_json"))
    result["market_evidence"] = json.loads(result.pop("market_evidence_json"))
    result["id"] = int(result["id"])
    result["snapshot_id"] = int(result["snapshot_id"])
    result["policy_version_id"] = int(result["policy_version_id"])
    if result["performance_run_id"] is not None:
        result["performance_run_id"] = int(result["performance_run_id"])
    return result


def get_evaluation_run(
    run_id: int, *, conn: sqlite3.Connection | None = None
) -> dict[str, Any] | None:
    if conn is not None:
        row = conn.execute(
            "SELECT * FROM ips_evaluation_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _decode(row) if row is not None else None
    with connect() as owned:
        row = owned.execute(
            "SELECT * FROM ips_evaluation_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _decode(row) if row is not None else None


def latest_evaluation_run(
    account_alias: str = "toss-brokerage",
) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM ips_evaluation_runs WHERE account_alias = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            (account_alias,),
        ).fetchone()
        return _decode(row) if row is not None else None


def list_evaluation_runs(
    *, limit: int = 20, account_alias: str = "toss-brokerage"
) -> list[dict[str, Any]]:
    """List immutable evaluation runs from newest to oldest."""
    if limit < 1:
        return []
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ips_evaluation_runs WHERE account_alias = ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (account_alias, limit),
        ).fetchall()
        return [_decode(row) for row in rows]
