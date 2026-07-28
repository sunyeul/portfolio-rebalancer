from copy import deepcopy
from datetime import UTC, datetime, timedelta
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from research.qlib_validation.contracts import (  # noqa: E402
    Candle,
    SeriesSpec,
    SourceSnapshot,
)
from services.dynamic_allocation import (  # noqa: E402
    allocation_benchmarks,
    build_neutral_policy,
)
from storage.policy_store import DEFAULT_POLICY, policy_hash  # noqa: E402


@pytest.fixture
def snapshot_factory():
    def build(days: int = 1) -> SourceSnapshot:
        policy = deepcopy(DEFAULT_POLICY)
        policy["instruments"] = [
            {"market_country": "US", "symbol": "SPY", "layer": "core", "target": 0.70},
            {"market_country": "US", "symbol": "GLD", "layer": "core", "target": 0.10},
            {
                "market_country": "US",
                "symbol": "QQQ",
                "layer": "satellite",
                "target": 0.20,
            },
            {
                "market_country": "US",
                "symbol": "TQQQ",
                "layer": "experiment",
                "target": 0.10,
            },
        ]
        policy = build_neutral_policy(policy)
        benchmark_specs = tuple(
            SeriesSpec(
                key=item["key"],
                source_kind=item["source_kind"],
                market_country=item["market_country"],
                symbol=item["symbol"],
                weight=float(item["weight"]),
                role="benchmark",
            )
            for item in allocation_benchmarks(policy)
        )
        instruments = tuple(policy["instruments"])
        invested_total = sum(float(item["target"]) for item in instruments)
        policy_specs = tuple(
            SeriesSpec(
                key=f"{item['market_country']}/{item['symbol']}",
                source_kind="stock",
                market_country=item["market_country"],
                symbol=item["symbol"],
                weight=float(item["target"]) / invested_total,
                role="policy_instrument",
            )
            for item in instruments
        )
        unique = {item.key: item for item in (*benchmark_specs, *policy_specs)}
        sessions = []
        cursor = datetime(2023, 1, 2, 21, 0, tzinfo=UTC)
        while len(sessions) < days:
            if cursor.weekday() < 5:
                sessions.append(cursor)
            cursor += timedelta(days=1)
        candles = tuple(
            Candle(
                key=spec.key,
                source_kind=spec.source_kind,
                market_country=spec.market_country,
                symbol=spec.symbol,
                session_date=point.date(),
                candle_at=point,
                available_at=point,
                currency="KRW" if spec.market_country == "KR" else "USD",
                open_price=100.0 + index,
                high_price=101.0 + index,
                low_price=99.0 + index,
                close_price=100.5 + index,
                volume=1000.0 + index,
                adjusted=spec.source_kind == "stock",
                adjusted_supported=spec.source_kind == "stock",
                factor=None,
            )
            for spec in unique.values()
            for index, point in enumerate(sessions)
        )
        return SourceSnapshot(
            policy_record={
                "id": 1,
                "account_alias": "toss-brokerage",
                "version": 1,
                "policy": policy,
                "policy_hash": policy_hash(policy),
                "created_at": "2023-01-01T00:00:00+00:00",
            },
            benchmark_specs=benchmark_specs,
            policy_specs=policy_specs,
            candles=candles,
        )

    return build


@pytest.fixture
def long_snapshot(snapshot_factory):
    return snapshot_factory(days=340)
