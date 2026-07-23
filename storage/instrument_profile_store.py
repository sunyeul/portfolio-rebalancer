"""Persistent IPS annotations keyed by Toss-observed instrument identity."""

from __future__ import annotations

import sqlite3
from typing import Any


LAYERS = frozenset({"core", "satellite", "experiment"})
THESIS_STATUSES = frozenset({"unknown", "valid", "watch", "broken"})
REVIEW_FACTOR_STATUSES = frozenset({"unknown", "clear", "review"})
ETF_SUBSTITUTION_STATUSES = frozenset(
    {"unknown", "not_applicable", "clear", "review"}
)


class InstrumentProfileError(ValueError):
    """Raised when a Toss instrument cannot receive the requested IPS profile."""


def instrument_key(symbol: str, market_country: str) -> tuple[str, str]:
    """Normalize the composite Toss instrument identity."""
    normalized_symbol = str(symbol).strip().upper()
    normalized_country = str(market_country).strip().upper()
    if not normalized_symbol or not normalized_country:
        raise InstrumentProfileError("symbol and market_country are required")
    return normalized_symbol, normalized_country


def _validate_profile(
    symbol: str,
    market_country: str,
    layer: str,
    thesis_status: str,
    thesis_note: str,
    overlap_status: str,
    management_burden_status: str,
    holdability_status: str,
    etf_substitution_status: str,
    review_factors_note: str,
) -> tuple[str, str, str, str, str, str, str, str, str, str]:
    normalized_symbol, normalized_country = instrument_key(symbol, market_country)
    normalized_layer = str(layer).strip().lower()
    if normalized_layer not in LAYERS:
        raise InstrumentProfileError(f"invalid layer: {layer}")
    normalized_status = str(thesis_status).strip().lower()
    if normalized_status not in THESIS_STATUSES:
        raise InstrumentProfileError(f"invalid thesis_status: {thesis_status}")
    normalized_factors: list[str] = []
    for field, value in (
        ("overlap_status", overlap_status),
        ("management_burden_status", management_burden_status),
        ("holdability_status", holdability_status),
    ):
        normalized = str(value).strip().lower()
        if normalized not in REVIEW_FACTOR_STATUSES:
            raise InstrumentProfileError(f"invalid {field}: {value}")
        normalized_factors.append(normalized)
    normalized_etf = str(etf_substitution_status).strip().lower()
    if normalized_etf not in ETF_SUBSTITUTION_STATUSES:
        raise InstrumentProfileError(
            f"invalid etf_substitution_status: {etf_substitution_status}"
        )
    return (
        normalized_symbol,
        normalized_country,
        normalized_layer,
        normalized_status,
        str(thesis_note or "").strip(),
        *normalized_factors,
        normalized_etf,
        str(review_factors_note or "").strip(),
    )


def _row_to_profile(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "account_alias": row["account_alias"],
        "market_country": row["market_country"],
        "symbol": row["symbol"],
        "layer": row["layer"],
        "thesis_status": row["thesis_status"],
        "thesis_note": row["thesis_note"],
        "overlap_status": row["overlap_status"],
        "management_burden_status": row["management_burden_status"],
        "holdability_status": row["holdability_status"],
        "etf_substitution_status": row["etf_substitution_status"],
        "review_factors_note": row["review_factors_note"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def upsert_profile(
    symbol: str,
    market_country: str,
    layer: str,
    thesis_status: str,
    thesis_note: str = "",
    account_alias: str = "toss-brokerage",
    *,
    overlap_status: str = "unknown",
    management_burden_status: str = "unknown",
    holdability_status: str = "unknown",
    etf_substitution_status: str = "unknown",
    review_factors_note: str = "",
) -> dict[str, Any]:
    """Create or update IPS metadata for a previously observed Toss holding."""
    (
        normalized_symbol,
        normalized_country,
        normalized_layer,
        normalized_status,
        normalized_note,
        normalized_overlap,
        normalized_burden,
        normalized_holdability,
        normalized_etf,
        normalized_factor_note,
    ) = _validate_profile(
        symbol,
        market_country,
        layer,
        thesis_status,
        thesis_note,
        overlap_status,
        management_burden_status,
        holdability_status,
        etf_substitution_status,
        review_factors_note,
    )
    from storage.database import connect

    with connect() as conn:
        observed = conn.execute(
            """
            SELECT 1
            FROM broker_holdings AS h
            JOIN broker_account_snapshots AS s ON s.id = h.snapshot_id
            WHERE s.account_alias = ?
              AND h.market_country = ?
              AND h.symbol = ?
            LIMIT 1
            """,
            (account_alias, normalized_country, normalized_symbol),
        ).fetchone()
        if observed is None:
            raise InstrumentProfileError(
                f"instrument not observed by Toss: "
                f"{normalized_country}/{normalized_symbol}"
            )

        conn.execute(
            """
            INSERT INTO ips_instrument_profiles (
                account_alias, market_country, symbol, layer, thesis_status,
                thesis_note, overlap_status, management_burden_status,
                holdability_status, etf_substitution_status, review_factors_note
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(account_alias, market_country, symbol) DO UPDATE SET
                layer = excluded.layer,
                thesis_status = excluded.thesis_status,
                thesis_note = excluded.thesis_note,
                overlap_status = excluded.overlap_status,
                management_burden_status = excluded.management_burden_status,
                holdability_status = excluded.holdability_status,
                etf_substitution_status = excluded.etf_substitution_status,
                review_factors_note = excluded.review_factors_note,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                account_alias,
                normalized_country,
                normalized_symbol,
                normalized_layer,
                normalized_status,
                normalized_note,
                normalized_overlap,
                normalized_burden,
                normalized_holdability,
                normalized_etf,
                normalized_factor_note,
            ),
        )
        row = conn.execute(
            """
            SELECT account_alias, market_country, symbol, layer, thesis_status,
                   thesis_note, overlap_status, management_burden_status,
                   holdability_status, etf_substitution_status,
                   review_factors_note, created_at, updated_at
            FROM ips_instrument_profiles
            WHERE account_alias = ? AND market_country = ? AND symbol = ?
            """,
            (account_alias, normalized_country, normalized_symbol),
        ).fetchone()
    if row is None:
        raise InstrumentProfileError("profile was not persisted")
    return _row_to_profile(row)


def get_profile(
    symbol: str,
    market_country: str,
    account_alias: str = "toss-brokerage",
) -> dict[str, Any] | None:
    """Return one profile by its Toss identity."""
    normalized_symbol, normalized_country = instrument_key(symbol, market_country)
    from storage.database import connect

    with connect() as conn:
        row = conn.execute(
            """
            SELECT account_alias, market_country, symbol, layer, thesis_status,
                   thesis_note, overlap_status, management_burden_status,
                   holdability_status, etf_substitution_status,
                   review_factors_note, created_at, updated_at
            FROM ips_instrument_profiles
            WHERE account_alias = ? AND market_country = ? AND symbol = ?
            """,
            (account_alias, normalized_country, normalized_symbol),
        ).fetchone()
    return _row_to_profile(row) if row is not None else None


def list_profiles(account_alias: str = "toss-brokerage") -> list[dict[str, Any]]:
    """Return profiles in deterministic Toss identity order."""
    from storage.database import connect

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT account_alias, market_country, symbol, layer, thesis_status,
                   thesis_note, overlap_status, management_burden_status,
                   holdability_status, etf_substitution_status,
                   review_factors_note, created_at, updated_at
            FROM ips_instrument_profiles
            WHERE account_alias = ?
            ORDER BY market_country, symbol
            """,
            (account_alias,),
        ).fetchall()
    return [_row_to_profile(row) for row in rows]


def profile_map(
    account_alias: str = "toss-brokerage",
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return profiles keyed by `(market_country, symbol)`."""
    return {
        (item["market_country"], item["symbol"]): item
        for item in list_profiles(account_alias)
    }
