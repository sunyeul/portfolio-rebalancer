from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Literal


@dataclass(frozen=True)
class SeriesSpec:
    key: str
    source_kind: str
    market_country: str
    symbol: str
    weight: float
    role: Literal["benchmark", "policy_instrument"]


@dataclass(frozen=True)
class Candle:
    key: str
    source_kind: str
    market_country: str
    symbol: str
    session_date: date
    candle_at: datetime
    available_at: datetime
    currency: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    adjusted: bool
    adjusted_supported: bool
    factor: float | None

    def evaluator_row(self) -> dict[str, Any]:
        return {
            "candle_at": self.candle_at.isoformat(),
            "close_price": self.close_price,
        }

    def record(self) -> dict[str, Any]:
        value = asdict(self)
        value["session_date"] = self.session_date.isoformat()
        value["candle_at"] = self.candle_at.isoformat()
        value["available_at"] = self.available_at.isoformat()
        return value


@dataclass(frozen=True)
class SourceSnapshot:
    policy_record: dict[str, Any]
    benchmark_specs: tuple[SeriesSpec, ...]
    policy_specs: tuple[SeriesSpec, ...]
    candles: tuple[Candle, ...]

    def candles_for(self, key: str) -> tuple[Candle, ...]:
        return tuple(item for item in self.candles if item.key == key)


@dataclass(frozen=True)
class ReplayPoint:
    month: str
    decision_timestamp: datetime
    regime: str | None
    reason: str
    cutoffs: dict[str, str]

    def record(self) -> dict[str, Any]:
        return {
            "month": self.month,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "regime": self.regime,
            "reason": self.reason,
            "cutoffs": dict(sorted(self.cutoffs.items())),
        }
