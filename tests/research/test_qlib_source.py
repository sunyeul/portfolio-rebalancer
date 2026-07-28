from copy import deepcopy
from datetime import UTC, datetime
import sqlite3

import pytest

from services.dynamic_allocation import allocation_benchmarks, build_neutral_policy
from storage.database import initialize_database
from storage.market_store import insert_candles
from storage.policy_store import (
    DEFAULT_POLICY,
    canonical_policy_json,
    get_active_policy,
    policy_hash,
)


def _seed_dynamic_policy(database):
    policy = deepcopy(DEFAULT_POLICY)
    policy["instruments"] = [
        {"market_country": "US", "symbol": "SPY", "layer": "core", "target": 0.7},
        {"market_country": "US", "symbol": "GLD", "layer": "core", "target": 0.1},
        {
            "market_country": "US",
            "symbol": "QQQ",
            "layer": "satellite",
            "target": 0.2,
        },
        {
            "market_country": "US",
            "symbol": "TQQQ",
            "layer": "experiment",
            "target": 0.1,
        },
    ]
    dynamic = build_neutral_policy(policy)
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE ips_policy_versions SET policy_json = ?, policy_hash = ? "
            "WHERE account_alias = 'toss-brokerage' AND superseded_at IS NULL",
            (canonical_policy_json(dynamic), policy_hash(dynamic)),
        )
    return dynamic


def test_snapshot_reads_active_policy_and_never_opens_database_for_write(
    tmp_path, monkeypatch
):
    from research.qlib_validation.source import load_snapshot, open_readonly

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    _seed_dynamic_policy(database)
    active = get_active_policy()
    assert active is not None

    for spec in allocation_benchmarks(active["policy"]):
        insert_candles(
            [
                {
                    "source_kind": spec["source_kind"],
                    "market_country": spec["market_country"],
                    "symbol": spec["symbol"],
                    "interval": "1d",
                    "candle_at": (
                        "2026-01-30T09:00:00+09:00"
                        if spec["market_country"] == "KR"
                        else "2026-01-30T09:30:00-05:00"
                    ),
                    "currency": "KRW" if spec["market_country"] == "KR" else "USD",
                    "open_price": 100.0,
                    "high_price": 101.0,
                    "low_price": 99.0,
                    "close_price": 100.5,
                    "volume": 1000.0,
                    "adjusted": spec["source_kind"] == "stock",
                    "adjusted_supported": spec["source_kind"] == "stock",
                }
            ]
        )

    snapshot = load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))
    assert snapshot.policy_record["policy_hash"] == active["policy_hash"]
    assert {item.key for item in snapshot.benchmark_specs} == {
        "US/SPY",
        "US/QQQ",
        "KR/KOSPI",
        "KR/KOSDAQ",
    }
    assert all(candle.factor is None for candle in snapshot.candles)

    with open_readonly(database) as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE forbidden_write (id INTEGER)")


def test_default_policy_without_dynamic_allocation_fails_closed(tmp_path, monkeypatch):
    from research.qlib_validation.source import SourceError, load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()

    with pytest.raises(SourceError, match="dynamic allocation policy configuration"):
        load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))


def test_malformed_dynamic_policy_duplicates_fail_closed():
    from research.qlib_validation.source import SourceError, _specs

    policy = deepcopy(DEFAULT_POLICY)
    policy["instruments"] = [
        {"market_country": "US", "symbol": "SPY", "layer": "core", "target": 0.7},
        {"market_country": "US", "symbol": "GLD", "layer": "core", "target": 0.1},
        {
            "market_country": "US",
            "symbol": "QQQ",
            "layer": "satellite",
            "target": 0.2,
        },
        {
            "market_country": "US",
            "symbol": "TQQQ",
            "layer": "experiment",
            "target": 0.1,
        },
    ]
    dynamic = build_neutral_policy(policy)
    dynamic["instruments"].append(deepcopy(dynamic["instruments"][0]))

    with pytest.raises(SourceError, match="duplicate identities"):
        _specs(dynamic)

    benchmark_duplicate = deepcopy(dynamic)
    benchmark_duplicate["instruments"].pop()
    benchmark_duplicate["allocation_review"]["benchmarks"].append(
        deepcopy(benchmark_duplicate["allocation_review"]["benchmarks"][0])
    )
    with pytest.raises(SourceError, match="duplicate keys"):
        _specs(benchmark_duplicate)


def test_availability_rules_fail_closed_for_naive_time_and_unknown_market():
    from research.qlib_validation.source import SourceError, _available_at

    with pytest.raises(SourceError, match="timezone-aware"):
        _available_at(datetime(2026, 1, 30, 9, 0), "KR")
    with pytest.raises(SourceError, match="unsupported market"):
        _available_at(datetime(2026, 1, 30, 9, 0, tzinfo=UTC), "JP")
