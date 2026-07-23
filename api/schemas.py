"""Sanitized response schemas for the local Toss dashboard."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ApiEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")
    ok: bool = True
    data: Any = None
    error: dict[str, Any] | None = None
