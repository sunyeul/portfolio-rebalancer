"""Decision journal JSON API."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from storage.journal_store import get_journal, update_journal, upsert_journal
from storage.portfolio_store import StorageError

router = APIRouter()


DecisionContext = Literal[
    "regular_review",
    "market_correction",
    "sharp_drop_review",
    "rebalance_review",
]


class JournalWriteRequest(BaseModel):
    date: str = Field(default_factory=lambda: date.today().isoformat())
    decision_context: DecisionContext = "regular_review"
    playbook_code: str | None = None
    dca_changes_considered: list[dict[str, Any]] = Field(default_factory=list)
    review_items: list[dict[str, Any]] = Field(default_factory=list)
    decision_note: str = ""


class JournalPatchRequest(BaseModel):
    date: str | None = None
    decision_context: DecisionContext | None = None
    playbook_code: str | None = None
    dca_changes_considered: list[dict[str, Any]] | None = None
    review_items: list[dict[str, Any]] | None = None
    decision_note: str | None = None


@router.get("/snapshots/{snapshot_id}")
async def get_snapshot_journal(snapshot_id: int):
    """Return the latest journal entry for a snapshot, if any."""
    return {"journal": get_journal(snapshot_id)}


@router.post("/snapshots/{snapshot_id}")
async def save_snapshot_journal(snapshot_id: int, payload: JournalWriteRequest):
    """Create or replace the latest journal entry for a snapshot."""
    try:
        return {"journal": upsert_journal(snapshot_id, payload.model_dump())}
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/snapshots/{snapshot_id}")
async def patch_snapshot_journal(snapshot_id: int, payload: JournalPatchRequest):
    """Partially update the latest journal entry for a snapshot."""
    try:
        updates = payload.model_dump(exclude_unset=True)
        return {"journal": update_journal(snapshot_id, updates)}
    except StorageError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
