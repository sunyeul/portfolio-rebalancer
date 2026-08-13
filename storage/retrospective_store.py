"""Append-only persistence for IPS retrospective cases and reviews."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping

from storage.database import connect


class RetrospectiveStorageError(RuntimeError):
    """Raised when a retrospective record cannot be stored safely."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decode_case(row: Any) -> dict[str, Any]:
    value = dict(row)
    value["id"] = int(value["id"])
    value["evaluation_run_id"] = int(value["evaluation_run_id"])
    value["queue_item"] = json.loads(value.pop("queue_item_json"))
    return value


def _decode_review(row: Any) -> dict[str, Any]:
    value = dict(row)
    for key in ("id", "case_id", "revision", "observation_snapshot_id"):
        value[key] = int(value[key])
    if value["performance_run_id"] is not None:
        value["performance_run_id"] = int(value["performance_run_id"])
    value["evidence"] = json.loads(value.pop("evidence_json"))
    return value


def create_case(
    *,
    account_alias: str,
    evaluation_run_id: int,
    queue_item: Mapping[str, Any],
    disposition: str,
    decision_note: str,
    decided_at: str | None = None,
) -> dict[str, Any]:
    kind = str(queue_item.get("kind") or "")
    identity = str(queue_item.get("identity") or "")
    if not kind or not identity:
        raise RetrospectiveStorageError("queue item identity is missing")
    with connect() as conn:
        existing = conn.execute(
            """
            SELECT * FROM ips_retrospective_cases
            WHERE evaluation_run_id = ? AND queue_kind = ? AND queue_identity = ?
            """,
            (evaluation_run_id, kind, identity),
        ).fetchone()
        if existing is not None:
            case = _decode_case(existing)
            if (
                case["disposition"] != disposition
                or case["decision_note"] != decision_note
            ):
                raise RetrospectiveStorageError("retrospective case decision conflict")
            return case
        cursor = conn.execute(
            """
            INSERT INTO ips_retrospective_cases (
                account_alias, evaluation_run_id, queue_kind, queue_identity,
                queue_item_json, disposition, decision_note, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_alias,
                evaluation_run_id,
                kind,
                identity,
                _json(dict(queue_item)),
                disposition,
                decision_note,
                decided_at or _now_iso(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM ips_retrospective_cases WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return _decode_case(row)


def get_case(case_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM ips_retrospective_cases WHERE id = ?", (case_id,)
        ).fetchone()
        return _decode_case(row) if row is not None else None


def list_cases(account_alias: str = "toss-brokerage") -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM ips_retrospective_cases
            WHERE account_alias = ?
            ORDER BY decided_at DESC, id DESC
            """,
            (account_alias,),
        ).fetchall()
        return [_decode_case(row) for row in rows]


def list_reviews(case_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM ips_retrospective_reviews
            WHERE case_id = ?
            ORDER BY horizon, revision DESC, id DESC
            """,
            (case_id,),
        ).fetchall()
        return [_decode_review(row) for row in rows]


def append_review(
    *,
    case_id: int,
    horizon: str,
    due_at: str,
    observation_snapshot_id: int,
    performance_run_id: int | None,
    evidence: Mapping[str, Any],
    evidence_fingerprint: str,
    judgment_assessment: str,
    execution_assessment: str,
    policy_assessment: str,
    review_note: str,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    with connect() as conn:
        if conn.execute(
            "SELECT id FROM ips_retrospective_cases WHERE id = ?", (case_id,)
        ).fetchone() is None:
            raise RetrospectiveStorageError(f"case_id={case_id} not found")
        revision = int(
            conn.execute(
                """
                SELECT COALESCE(MAX(revision), 0) + 1
                FROM ips_retrospective_reviews WHERE case_id = ? AND horizon = ?
                """,
                (case_id, horizon),
            ).fetchone()[0]
        )
        cursor = conn.execute(
            """
            INSERT INTO ips_retrospective_reviews (
                case_id, horizon, revision, due_at, observation_snapshot_id,
                performance_run_id, evidence_json, evidence_fingerprint,
                judgment_assessment, execution_assessment, policy_assessment,
                review_note, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                horizon,
                revision,
                due_at,
                observation_snapshot_id,
                performance_run_id,
                _json(dict(evidence)),
                evidence_fingerprint,
                judgment_assessment,
                execution_assessment,
                policy_assessment,
                review_note,
                reviewed_at or _now_iso(),
            ),
        )
        row = conn.execute(
            "SELECT * FROM ips_retrospective_reviews WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return _decode_review(row)
