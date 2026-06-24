"""Evaluation Framework v2 domain models."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


EvaluationLevel = Literal["layer", "asset"]
EvaluationStatus = Literal["OK", "Watch", "Review", "Action"]
ThesisStatus = Literal["valid", "watch", "broken", "unknown"]
BurdenLevel = Literal["low", "medium", "high"]


class EvaluationPeriod(BaseModel):
    """Date window used for v2 performance evaluation."""

    label: Literal["1M", "3M", "6M", "YTD", "1Y", "Max", "custom"]
    start_date: date
    end_date: date

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, value: date, info) -> date:
        start_date = info.data.get("start_date")
        if start_date is not None and value < start_date:
            raise ValueError("end_date must be on or after start_date")
        return value


class BenchmarkConfig(BaseModel):
    """Benchmark definition for an evaluation unit."""

    label: str
    components: dict[str, float] = Field(default_factory=dict)


class EvaluationUnit(BaseModel):
    """Layer or asset inspected by the common v2 evaluation frame."""

    level: EvaluationLevel
    name: str
    parent_layer: str | None = None
    benchmark: str | BenchmarkConfig | None = None
    target_weight: float | None = None
    allowed_mdd: float | None = None
    allowed_volatility: float | None = None
    max_weight: float | None = None
    min_efficiency: float | None = None
    thesis: str | None = None
    counter_scenario: str | None = None
    check_frequency: str | None = None
    manual_intervention_allowed: bool | None = None
    evaluation_period: EvaluationPeriod


class EvaluationOutput(BaseModel):
    """Computed v2 evaluation values for a unit."""

    current_weight: float
    weight_gap: float | None = None
    layer_internal_weight: float | None = None
    period_return: float | None = None
    cagr: float | None = None
    benchmark_return: float | None = None
    benchmark_excess_return: float | None = None
    mdd: float | None = None
    volatility: float | None = None
    concentration: float | None = None
    risk_contribution: float | None = None
    return_mdd_ratio: float | None = None
    cagr_mdd_ratio: float | None = None
    thesis_status: ThesisStatus = "unknown"
    burden: BurdenLevel = "medium"
    status: EvaluationStatus
    triggered_by: list[str] = Field(default_factory=list)


class ReviewItem(BaseModel):
    """Queue item generated from non-OK v2 evaluations."""

    level: EvaluationLevel
    name: str
    parent_layer: str | None = None
    status: Literal["Watch", "Review", "Action"]
    triggered_by: list[str] = Field(default_factory=list)
    metrics_snapshot: dict[str, Any] = Field(default_factory=dict)
    thesis: str | None = None
    counter_scenario: str | None = None
    suggested_next_step: str
