from copy import deepcopy
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
import sqlite3
from zoneinfo import ZoneInfo

import pytest

from services.dynamic_allocation import allocation_benchmarks, build_neutral_policy
from integrations.toss.observation import NormalizedHolding
from storage.account_observation_store import insert_snapshot
from storage.database import initialize_database
from storage.market_store import insert_candles
from storage.policy_store import (
    DEFAULT_POLICY,
    canonical_policy_json,
    get_active_policy,
    policy_hash,
)
from tests.test_account_observation_store import _snapshot


def _dynamic_policy():
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
    return build_neutral_policy(policy)


def _seed_dynamic_policy(database):
    dynamic = _dynamic_policy()
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE ips_policy_versions SET policy_json = ?, policy_hash = ? "
            "WHERE account_alias = 'toss-brokerage' AND superseded_at IS NULL",
            (canonical_policy_json(dynamic), policy_hash(dynamic)),
        )
    return dynamic


def _candle(spec, candle_at, *, price=100.0):
    return {
        "source_kind": spec["source_kind"],
        "market_country": spec["market_country"],
        "symbol": spec["symbol"],
        "interval": "1d",
        "candle_at": candle_at,
        "currency": "KRW" if spec["market_country"] == "KR" else "USD",
        "open_price": price,
        "high_price": price + 1.0,
        "low_price": price - 1.0,
        "close_price": price,
        "volume": 1000.0,
        "adjusted": spec["source_kind"] == "stock",
        "adjusted_supported": spec["source_kind"] == "stock",
    }


def _weekday_sessions(start: date, count: int) -> list[date]:
    sessions: list[date] = []
    cursor = start
    while len(sessions) < count:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    return sessions


def _holding(symbol: str, market_country: str) -> NormalizedHolding:
    return NormalizedHolding(
        symbol=symbol,
        name=symbol,
        market_country=market_country,
        currency="KRW" if market_country == "KR" else "USD",
        quantity=1.0,
        last_price=100.0,
        average_purchase_price=100.0,
        market_value_native=100.0,
        market_value_krw=100000.0,
        cost_native=100.0,
        cost_krw=100000.0,
        profit_loss_native=0.0,
        profit_loss_krw=0.0,
        daily_profit_loss_native=0.0,
        daily_profit_loss_krw=0.0,
    )


def _seed_factor_history(policy, *, sessions: list[date]) -> set[str]:
    specs = [
        *allocation_benchmarks(policy),
        *[
            {
                "source_kind": "stock",
                "market_country": item["market_country"],
                "symbol": item["symbol"],
            }
            for item in policy["instruments"]
        ],
    ]
    unique = {
        (item["source_kind"], item["market_country"], item["symbol"]): item
        for item in specs
    }
    for spec in unique.values():
        timezone = ZoneInfo(
            "Asia/Seoul" if spec["market_country"] == "KR" else "America/New_York"
        )
        market_time = time(9) if spec["market_country"] == "KR" else time(9, 30)
        insert_candles(
            [
                _candle(
                    spec,
                    datetime.combine(session, market_time, timezone).isoformat(),
                    price=100.0 + index,
                )
                for index, session in enumerate(sessions)
            ]
        )
    return {f"{item['market_country']}/{item['symbol']}" for item in unique.values()}


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
        assert conn.in_transaction is True
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE forbidden_write (id INTEGER)")


def test_snapshot_derives_trailing_factor_without_future_candles(tmp_path, monkeypatch):
    from research.qlib_validation.source import load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    policy = _seed_dynamic_policy(database)
    sessions = _weekday_sessions(date(2026, 1, 2), 21)
    keys = _seed_factor_history(policy, sessions=sessions)

    complete = load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))

    assert {item.key for item in complete.candles} == keys
    assert len(complete.candles) == len(keys)
    assert all(item.factor == pytest.approx(0.2) for item in complete.candles)

    before_final_candle = load_snapshot(
        database, as_of=datetime(2026, 1, 29, 23, tzinfo=UTC)
    )

    assert before_final_candle.candles == ()


def test_current_holdings_universe_uses_only_latest_complete_toss_holdings(
    tmp_path, monkeypatch
):
    from research.qlib_validation.source import load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    policy = _seed_dynamic_policy(database)
    sessions = _weekday_sessions(date(2026, 1, 2), 21)
    _seed_factor_history(policy, sessions=sessions)
    account_snapshot = insert_snapshot(
        replace(
            _snapshot(
                fingerprint="current-holdings", synced_at="2026-02-01T00:00:00+00:00"
            ),
            holdings=(_holding("SPY", "US"), _holding("QQQ", "US")),
        )
    )
    before = database.read_bytes()

    snapshot = load_snapshot(
        database,
        as_of=datetime(2026, 2, 2, tzinfo=UTC),
        universe="current-holdings",
    )

    assert {item.key for item in snapshot.policy_specs} == {"US/SPY", "US/QQQ"}
    assert snapshot.research_universe == {
        "mode": "current-holdings",
        "account_snapshot_id": account_snapshot["id"],
        "account_snapshot_synced_at": "2026-02-01T00:00:00+00:00",
        "selected_policy_instruments": ["US/QQQ", "US/SPY"],
        "excluded_policy_instruments": ["US/GLD", "US/TQQQ"],
    }
    assert {item.key for item in snapshot.candles} == {
        "KR/KOSDAQ",
        "KR/KOSPI",
        "US/QQQ",
        "US/SPY",
    }
    assert database.read_bytes() == before


def test_current_holdings_universe_rejects_a_snapshot_after_research_as_of(
    tmp_path, monkeypatch
):
    from research.qlib_validation.source import SourceError, load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    policy = _seed_dynamic_policy(database)
    _seed_factor_history(policy, sessions=_weekday_sessions(date(2026, 1, 2), 21))
    insert_snapshot(
        replace(
            _snapshot(
                fingerprint="future-holdings", synced_at="2026-02-02T00:00:00+00:00"
            ),
            holdings=(_holding("SPY", "US"),),
        )
    )

    with pytest.raises(SourceError, match="unavailable at the research as_of"):
        load_snapshot(
            database,
            as_of=datetime(2026, 2, 1, tzinfo=UTC),
            universe="current-holdings",
        )


def test_current_holdings_universe_replays_the_latest_snapshot_available_at_as_of(
    tmp_path, monkeypatch
):
    from research.qlib_validation.source import load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    policy = _seed_dynamic_policy(database)
    _seed_factor_history(policy, sessions=_weekday_sessions(date(2026, 1, 2), 21))
    earlier = insert_snapshot(
        replace(
            _snapshot(fingerprint="earlier", synced_at="2026-02-01T00:00:00+00:00"),
            holdings=(_holding("SPY", "US"),),
        )
    )
    insert_snapshot(
        replace(
            _snapshot(fingerprint="later", synced_at="2026-02-02T00:00:00+00:00"),
            holdings=(_holding("QQQ", "US"),),
        )
    )

    snapshot = load_snapshot(
        database,
        as_of=datetime(2026, 2, 1, 12, tzinfo=UTC),
        universe="current-holdings",
    )

    assert {item.key for item in snapshot.policy_specs} == {"US/SPY"}
    assert snapshot.research_universe["account_snapshot_id"] == earlier["id"]


def test_snapshot_uses_latest_candle_revision_once(tmp_path, monkeypatch):
    from research.qlib_validation.source import load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    policy = _seed_dynamic_policy(database)
    sessions = _weekday_sessions(date(2026, 1, 2), 21)
    _seed_factor_history(policy, sessions=sessions)
    spy = next(
        item
        for item in allocation_benchmarks(policy)
        if item["market_country"] == "US" and item["symbol"] == "SPY"
    )
    revised_at = datetime.combine(
        sessions[-1], time(9, 30), ZoneInfo("America/New_York")
    ).isoformat()
    insert_candles([_candle(spy, revised_at, price=130.0)])

    snapshot = load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))
    spy_candles = [item for item in snapshot.candles if item.key == "US/SPY"]

    assert len(spy_candles) == 1
    assert spy_candles[0].close_price == 130.0
    assert spy_candles[0].factor == pytest.approx(0.3)


def test_readonly_transaction_keeps_one_snapshot_during_concurrent_commit(
    tmp_path, monkeypatch
):
    from research.qlib_validation.source import open_readonly

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE qlib_transaction_probe (value INTEGER NOT NULL)")

    with open_readonly(database) as reader:
        assert (
            reader.execute("SELECT COUNT(*) FROM qlib_transaction_probe").fetchone()[0]
            == 0
        )
        with sqlite3.connect(database) as writer:
            writer.execute("INSERT INTO qlib_transaction_probe VALUES (1)")
        assert (
            reader.execute("SELECT COUNT(*) FROM qlib_transaction_probe").fetchone()[0]
            == 0
        )

    with sqlite3.connect(database) as observer:
        assert (
            observer.execute("SELECT COUNT(*) FROM qlib_transaction_probe").fetchone()[
                0
            ]
            == 1
        )


def test_default_policy_without_dynamic_allocation_fails_closed(tmp_path, monkeypatch):
    from research.qlib_validation.source import SourceError, load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()

    with pytest.raises(SourceError, match="dynamic allocation policy configuration"):
        load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))


def test_malformed_dynamic_policy_duplicates_fail_closed():
    from research.qlib_validation.source import SourceError, _specs

    dynamic = _dynamic_policy()
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


def test_malformed_dynamic_policy_requires_exact_benchmark_identities():
    from research.qlib_validation.source import SourceError, _specs

    dynamic = _dynamic_policy()
    benchmarks = dynamic["allocation_review"]["benchmarks"]
    dynamic["allocation_review"]["benchmarks"] = [
        {**benchmarks[0], "weight": 0.5},
        {**benchmarks[2], "weight": 0.5},
    ]

    with pytest.raises(SourceError, match="exactly the required identities"):
        _specs(dynamic)


def test_snapshot_rejects_policy_hash_mismatch(tmp_path, monkeypatch):
    from research.qlib_validation.source import SourceError, load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    dynamic = _seed_dynamic_policy(database)
    tampered = deepcopy(dynamic)
    tampered["performance"]["annual_return_target"] = 0.99
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE ips_policy_versions SET policy_json = ? "
            "WHERE account_alias = 'toss-brokerage' AND superseded_at IS NULL",
            (canonical_policy_json(tampered),),
        )

    with pytest.raises(SourceError, match="policy hash mismatch"):
        load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))


def test_snapshot_rejects_invalid_or_gapped_candles(tmp_path, monkeypatch):
    from research.qlib_validation.source import SourceError, load_snapshot

    database = tmp_path / "portfolio.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    dynamic = _seed_dynamic_policy(database)
    spec = allocation_benchmarks(dynamic)[0]
    insert_candles(
        [
            _candle(spec, "2026-01-02T09:30:00-05:00", price=-100.0),
            _candle(spec, "2026-01-30T09:30:00-05:00", price=-101.0),
        ]
    )

    with pytest.raises(SourceError, match="invalid positive OHLC"):
        load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))

    with sqlite3.connect(database) as conn:
        conn.execute("DELETE FROM toss_market_candles")
    insert_candles(
        [
            _candle(spec, "2026-01-02T09:30:00-05:00"),
            _candle(spec, "2026-01-30T09:30:00-05:00"),
        ]
    )

    with pytest.raises(SourceError, match="history gap"):
        load_snapshot(database, as_of=datetime(2026, 2, 1, tzinfo=UTC))


def test_availability_rules_fail_closed_for_naive_time_and_unknown_market():
    from research.qlib_validation.source import SourceError, _available_at

    with pytest.raises(SourceError, match="timezone-aware"):
        _available_at(datetime(2026, 1, 30, 9, 0), "KR")
    with pytest.raises(SourceError, match="unsupported market"):
        _available_at(datetime(2026, 1, 30, 9, 0, tzinfo=UTC), "JP")
