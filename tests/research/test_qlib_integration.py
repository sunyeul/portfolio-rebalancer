from copy import deepcopy
from hashlib import sha256
from importlib.util import find_spec
import json
import sqlite3

import pytest

from research.qlib_validation.cli import main
from services.dynamic_allocation import allocation_benchmarks, build_neutral_policy
from storage.database import initialize_database
from storage.market_store import insert_candles
from storage.policy_store import (
    DEFAULT_POLICY,
    canonical_policy_json,
    get_active_policy,
    policy_hash,
)


pytestmark = pytest.mark.skipif(
    find_spec("qlib") is None,
    reason="Qlib research environment only",
)


def _seed_dynamic_policy(database):
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
    dynamic = build_neutral_policy(policy)
    with sqlite3.connect(database) as conn:
        conn.execute(
            "UPDATE ips_policy_versions SET policy_json = ?, policy_hash = ? "
            "WHERE account_alias = 'toss-brokerage' AND superseded_at IS NULL",
            (canonical_policy_json(dynamic), policy_hash(dynamic)),
        )
    return dynamic


def test_real_cli_keeps_fixture_database_bytes_unchanged(monkeypatch, capsys, tmp_path):
    database = tmp_path / "fixture.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(database))
    initialize_database()
    _seed_dynamic_policy(database)
    active = get_active_policy()
    assert active is not None
    policy_specs = [
        {
            "source_kind": "stock",
            "market_country": item["market_country"],
            "symbol": item["symbol"],
        }
        for item in active["policy"]["instruments"]
    ]
    unique_specs = {
        (item["source_kind"], item["market_country"], item["symbol"]): item
        for item in [*allocation_benchmarks(active["policy"]), *policy_specs]
    }
    for spec in unique_specs.values():
        is_stock = spec["source_kind"] == "stock"
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
                    "adjusted": is_stock,
                    "adjusted_supported": is_stock,
                }
            ]
        )
    before = sha256(database.read_bytes()).hexdigest()
    code = main(
        [
            "stage1",
            "--db",
            str(database),
            "--as-of",
            "2026-02-01T00:00:00+00:00",
            "--output",
            str(tmp_path / "artifacts"),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["regime_signal_verdict"] == "inconclusive"
    assert result["target_policy_verdict"] == "inconclusive"
    assert sha256(database.read_bytes()).hexdigest() == before
