"""Application configuration JSON API."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from storage.config_store import (
    ConfigError,
    get_ips_management_config,
    list_options,
    replace_action_priorities,
    replace_rules,
    replace_target_allocations,
    set_option_active,
    upsert_option,
)

router = APIRouter()

OptionTable = Literal["groups", "roles", "thesis_statuses"]


class OptionRequest(BaseModel):
    code: str
    label: str
    sort_order: int = 999
    is_active: bool = True
    group_type: str | None = None


class ActiveRequest(BaseModel):
    is_active: bool


class TargetAllocationRequest(BaseModel):
    group_type: str
    min: float = Field(ge=0, le=1)
    target: float = Field(ge=0, le=1)
    max: float = Field(ge=0, le=1)


class ActionPriorityRequest(BaseModel):
    action_code: str
    label: str
    priority: int
    is_active: bool = True


class RuleRequest(BaseModel):
    key: str
    value: Any


@router.get("/options")
async def get_options(include_inactive: bool = True):
    """Return dropdown option lookups."""
    return list_options(include_inactive=include_inactive)


@router.get("/ips")
async def get_ips_config():
    """Return editable IPS configuration and runtime config shape."""
    return get_ips_management_config()


@router.post("/{table}")
async def save_option(table: OptionTable, payload: OptionRequest):
    """Create or update a config option."""
    try:
        return {
            "option": upsert_option(
                table,
                payload.code,
                payload.label,
                payload.sort_order,
                payload.is_active,
                payload.group_type,
            )
        }
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/{table}/{code}/active")
async def update_option_active(table: OptionTable, code: str, payload: ActiveRequest):
    """Soft-delete or reactivate an option."""
    try:
        return {"option": set_option_active(table, code, payload.is_active)}
    except ConfigError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/ips/target-allocations")
async def save_target_allocations(payload: list[TargetAllocationRequest]):
    """Replace IPS target allocations."""
    try:
        return {
            "target_allocations": replace_target_allocations(
                [row.model_dump() for row in payload]
            )
        }
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/ips/action-priorities")
async def save_action_priorities(payload: list[ActionPriorityRequest]):
    """Replace IPS action priorities."""
    try:
        return {
            "action_priorities": replace_action_priorities(
                [row.model_dump() for row in payload]
            )
        }
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/ips/rules")
async def save_rules(payload: list[RuleRequest]):
    """Replace IPS rules."""
    try:
        return {"rules": replace_rules([row.model_dump() for row in payload])}
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
