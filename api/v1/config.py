"""Application configuration JSON API."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from storage.config_store import (
    ConfigError,
    get_ips_management_config,
    list_options,
    replace_target_allocations,
)

router = APIRouter()

class TargetAllocationRequest(BaseModel):
    layer: str
    min: float = Field(ge=0, le=1)
    target: float = Field(ge=0, le=1)
    max: float = Field(ge=0, le=1)


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
