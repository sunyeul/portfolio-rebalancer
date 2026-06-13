"""Persistence helpers for IPS decision journal entries."""

from __future__ import annotations

import json
from typing import Any

from storage.database import connect, initialize_database
from storage.portfolio_store import StorageError


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _json_load(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _row_to_journal(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "snapshot_id": row["snapshot_id"],
        "date": row["date"],
        "decision_context": row["decision_context"],
        "playbook_code": row["playbook_code"],
        "dca_changes_considered": _json_load(
            row["dca_changes_considered_json"],
            [],
        ),
        "review_items": _json_load(row["review_items_json"], []),
        "decision_note": row["decision_note"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def get_journal(snapshot_id: int) -> dict[str, Any] | None:
    initialize_database()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM journal_entries
            WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchone()
    return _row_to_journal(row) if row else None


def upsert_journal(snapshot_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    initialize_database()
    with connect() as conn:
        snapshot = conn.execute(
            "SELECT id FROM portfolio_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        if snapshot is None:
            raise StorageError("스냅샷을 찾을 수 없습니다.")

        conn.execute(
            """
            INSERT INTO journal_entries (
                snapshot_id,
                date,
                decision_context,
                playbook_code,
                dca_changes_considered_json,
                review_items_json,
                decision_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(snapshot_id) DO UPDATE SET
                date = excluded.date,
                decision_context = excluded.decision_context,
                playbook_code = excluded.playbook_code,
                dca_changes_considered_json = excluded.dca_changes_considered_json,
                review_items_json = excluded.review_items_json,
                decision_note = excluded.decision_note,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                snapshot_id,
                payload["date"],
                payload["decision_context"],
                payload.get("playbook_code"),
                _json_dump(payload.get("dca_changes_considered") or []),
                _json_dump(payload.get("review_items") or []),
                payload.get("decision_note") or "",
            ),
        )

    journal = get_journal(snapshot_id)
    if journal is None:
        raise StorageError("저널 저장 결과를 찾을 수 없습니다.")
    return journal


def update_journal(snapshot_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    current = get_journal(snapshot_id)
    if current is None:
        raise StorageError("수정할 저널을 찾을 수 없습니다.")

    merged = {
        "date": payload.get("date", current["date"]),
        "decision_context": payload.get(
            "decision_context",
            current["decision_context"],
        ),
        "playbook_code": payload.get("playbook_code", current["playbook_code"]),
        "dca_changes_considered": payload.get(
            "dca_changes_considered",
            current["dca_changes_considered"],
        ),
        "review_items": payload.get("review_items", current["review_items"]),
        "decision_note": payload.get("decision_note", current["decision_note"]),
    }
    return upsert_journal(snapshot_id, merged)
