import sqlite3

import pytest

from storage.database import initialize_database
from storage.instrument_profile_store import (
    InstrumentProfileError,
    get_profile,
    list_profiles,
    upsert_profile,
)


@pytest.fixture
def snapshot_fixture(monkeypatch, tmp_path):
    path = tmp_path / "profiles.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))
    initialize_database()
    with sqlite3.connect(path) as conn:
        snapshot_id = int(
            conn.execute(
                """
                INSERT INTO broker_account_snapshots (
                    account_alias, sync_started_at, synced_at, state,
                    is_current_evaluable, source_fingerprint,
                    source_timestamps_json, data_quality_json,
                    reconciliation_json, total_value_krw, invested_value_krw,
                    cash_value_krw
                )
                VALUES (
                    'toss-brokerage', '2026-07-23T00:00:00Z',
                    '2026-07-23T00:01:00Z', 'complete', 1, 'profile-fixture',
                    '{}', '{}', '{}', 100000.0, 90000.0, 10000.0
                )
                """
            ).lastrowid
        )
        conn.executemany(
            """
            INSERT INTO broker_holdings (
                snapshot_id, symbol, name, market_country, currency, quantity,
                last_price, average_purchase_price, market_value_native,
                market_value_krw, cost_native, cost_krw, profit_loss_native,
                profit_loss_krw, daily_profit_loss_native,
                daily_profit_loss_krw
            )
            VALUES (?, ?, ?, ?, ?, 1, 100, 90, ?, ?, 90, 90, 10, 10, 1, 1)
            """,
            [
                (snapshot_id, "AAPL", "Apple", "US", "USD", 60000.0, 60000.0),
                (snapshot_id, "005930", "Samsung", "KR", "KRW", 30000.0, 30000.0),
            ],
        )
    return {"path": path, "snapshot_id": snapshot_id}


def test_upsert_profile_requires_a_toss_observed_identity(snapshot_fixture):
    profile = upsert_profile(
        symbol="AAPL",
        market_country="US",
        layer="core",
        thesis_status="valid",
        thesis_note="Long-term core",
    )
    assert profile["symbol"] == "AAPL"
    assert profile["market_country"] == "US"

    with pytest.raises(InstrumentProfileError, match="not observed"):
        upsert_profile(
            symbol="UNSEEN",
            market_country="US",
            layer="core",
            thesis_status="valid",
        )


@pytest.mark.parametrize("layer", ["cash", "other", ""])
def test_upsert_profile_rejects_invalid_layer(snapshot_fixture, layer):
    with pytest.raises(InstrumentProfileError, match="invalid layer"):
        upsert_profile("AAPL", "US", layer, "valid")


@pytest.mark.parametrize("status", ["intact", "sold", ""])
def test_upsert_profile_rejects_invalid_thesis(snapshot_fixture, status):
    with pytest.raises(InstrumentProfileError, match="invalid thesis_status"):
        upsert_profile("AAPL", "US", "core", status)


def test_profile_update_does_not_mutate_broker_holding(snapshot_fixture):
    first = upsert_profile("AAPL", "US", "core", "valid")
    with sqlite3.connect(snapshot_fixture["path"]) as conn:
        before = conn.execute(
            "SELECT symbol, market_value_krw, cost_krw FROM broker_holdings "
            "WHERE symbol = 'AAPL'"
        ).fetchone()

    second = upsert_profile("AAPL", "US", "satellite", "watch", "Review overlap")

    with sqlite3.connect(snapshot_fixture["path"]) as conn:
        after = conn.execute(
            "SELECT symbol, market_value_krw, cost_krw FROM broker_holdings "
            "WHERE symbol = 'AAPL'"
        ).fetchone()
    assert first["layer"] == "core"
    assert second["layer"] == "satellite"
    assert second["thesis_status"] == "watch"
    assert before == after


def test_profile_listing_is_deterministic(snapshot_fixture):
    upsert_profile("005930", "KR", "core", "valid")
    upsert_profile("AAPL", "US", "satellite", "unknown")

    rows = list_profiles()

    assert [(row["market_country"], row["symbol"]) for row in rows] == [
        ("KR", "005930"),
        ("US", "AAPL"),
    ]
    assert get_profile("aapl", "us")["symbol"] == "AAPL"
