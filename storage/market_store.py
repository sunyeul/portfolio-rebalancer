"""Immutable local storage for normalized Toss market observations and candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from storage.database import connect


class MarketStoreError(RuntimeError):
    """Raised when market evidence cannot be persisted safely."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def candle_fingerprint(candle: Mapping[str, Any]) -> str:
    payload = _json(dict(candle)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def insert_candles(candles: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with connect() as conn:
        for candle in candles:
            required = (
                "source_kind",
                "market_country",
                "symbol",
                "interval",
                "candle_at",
                "currency",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
                "adjusted",
            )
            missing = [key for key in required if key not in candle]
            if missing:
                raise MarketStoreError(f"candle missing fields: {', '.join(missing)}")
            fingerprint = str(
                candle.get("source_fingerprint") or candle_fingerprint(candle)
            )
            identity = (
                candle["source_kind"],
                candle["market_country"],
                candle["symbol"],
                candle["interval"],
                candle["candle_at"],
                int(bool(candle["adjusted"])),
            )
            existing = conn.execute(
                """
                SELECT * FROM toss_market_candles
                WHERE source_kind = ? AND market_country = ? AND symbol = ?
                  AND interval = ? AND candle_at = ? AND adjusted = ?
                ORDER BY id LIMIT 1
                """,
                identity,
            ).fetchone()
            if existing is not None:
                if existing["source_fingerprint"] != fingerprint:
                    raise MarketStoreError(
                        "conflicting candle observation for the same timestamp"
                    )
                rows.append(dict(existing))
                continue
            conn.execute(
                """
                INSERT INTO toss_market_candles (
                    source_kind, market_country, symbol, interval, candle_at,
                    currency, open_price, high_price, low_price, close_price,
                    volume, adjusted, adjusted_supported, source_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candle["source_kind"],
                    candle["market_country"],
                    candle["symbol"],
                    candle["interval"],
                    candle["candle_at"],
                    candle["currency"],
                    candle["open_price"],
                    candle["high_price"],
                    candle["low_price"],
                    candle["close_price"],
                    candle["volume"],
                    int(bool(candle["adjusted"])),
                    int(
                        bool(
                            candle.get(
                                "adjusted_supported",
                                candle["source_kind"] == "stock",
                            )
                        )
                    ),
                    fingerprint,
                ),
            )
            row = conn.execute(
                "SELECT * FROM toss_market_candles WHERE id = last_insert_rowid()"
            ).fetchone()
            if row is not None:
                rows.append(dict(row))
    return rows


def list_candles(
    *,
    symbol: str,
    source_kind: str = "stock",
    market_country: str = "",
    interval: str = "1d",
    limit: int = 500,
) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM toss_market_candles
            WHERE source_kind = ? AND market_country = ? AND symbol = ? AND interval = ?
            ORDER BY candle_at DESC LIMIT ?
            """,
            (
                source_kind,
                market_country.upper(),
                symbol.upper(),
                interval,
                max(1, limit),
            ),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def insert_policy_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    required = ("account_alias", "base_policy_version_id", "candidate_json")
    missing = [key for key in required if key not in candidate]
    if missing:
        raise MarketStoreError(f"policy candidate missing fields: {', '.join(missing)}")
    encoded = _json(candidate["candidate_json"])
    fingerprint = str(candidate.get("candidate_hash") or "")
    if not fingerprint:
        identity = "|".join(
            (
                str(candidate["account_alias"]),
                str(candidate["base_policy_version_id"]),
                encoded,
            )
        )
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM ips_policy_candidates WHERE candidate_hash = ?",
            (fingerprint,),
        ).fetchone()
        if existing is None:
            cursor = conn.execute(
                """
                INSERT INTO ips_policy_candidates (
                    account_alias, base_policy_version_id, state, candidate_json, candidate_hash
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    candidate["account_alias"],
                    candidate["base_policy_version_id"],
                    candidate.get("state", "candidate"),
                    encoded,
                    fingerprint,
                ),
            )
            candidate_id = int(cursor.lastrowid)
        else:
            candidate_id = int(existing["id"])
        row = conn.execute(
            "SELECT * FROM ips_policy_candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
    if row is None:  # pragma: no cover
        raise MarketStoreError("policy candidate could not be read back")
    result = dict(row)
    result["candidate_json"] = json.loads(result["candidate_json"])
    return result


def latest_policy_candidate(
    account_alias: str = "toss-brokerage",
    base_policy_version_id: int | None = None,
) -> dict[str, Any] | None:
    with connect() as conn:
        if base_policy_version_id is None:
            row = conn.execute(
                """
                SELECT * FROM ips_policy_candidates
                WHERE account_alias = ? ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (account_alias,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT * FROM ips_policy_candidates
                WHERE account_alias = ? AND base_policy_version_id = ?
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (account_alias, base_policy_version_id),
            ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["candidate_json"] = json.loads(result["candidate_json"])
    return result
