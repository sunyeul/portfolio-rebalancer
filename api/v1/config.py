"""Application configuration JSON API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from storage.config_store import (
    ConfigError,
    get_ips_management_config,
    list_options,
    replace_rules,
    replace_target_allocations,
)

router = APIRouter()

class TargetAllocationRequest(BaseModel):
    group: str
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
    """Reject edits to app-defined IPS action priorities."""
    raise HTTPException(status_code=403, detail="액션 우선순위는 읽기 전용입니다.")


@router.put("/ips/rules")
async def save_rules(payload: list[RuleRequest]):
    """Replace IPS rules."""
    try:
        return {"rules": replace_rules([row.model_dump() for row in payload])}
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
