"""Versioned Toss-only IPS policy persistence."""

from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from typing import Any


DEFAULT_POLICY: dict[str, Any] = {
    "cash_reserve": {"minimum": 0.10, "target": 0.15, "maximum": 0.20},
    "performance": {
        "annual_return_target": 0.10,
        "measurement": "ytd_twr",
        "minimum_history_days": 365,
    },
    "risk_review": {
        "lookback_sessions": 252,
        "minimum_history_points": 200,
        "max_data_age_days": 7,
        "max_gap_days": 7,
        "account_drawdown_review": -0.15,
        "instrument_drawdown_review": {
            "core": -0.25,
            "satellite": -0.20,
            "experiment": -0.15,
        },
    },
    "layers": {
        "core": {"minimum": 0.70, "target": 0.80, "maximum": 0.90},
        "satellite": {"minimum": 0.10, "target": 0.20, "maximum": 0.30},
        "experiment": {"minimum": 0.00, "target": 0.00, "maximum": 0.05},
    },
}


def canonical_policy_json(policy: dict[str, Any]) -> str:
    """Encode policy deterministically for persistence and replay."""
    return json.dumps(
        policy,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def policy_hash(policy: dict[str, Any]) -> str:
    """Return the stable fingerprint for one canonical policy."""
    return sha256(canonical_policy_json(policy).encode("utf-8")).hexdigest()


def ensure_default_policy(
    conn: sqlite3.Connection,
    account_alias: str = "toss-brokerage",
) -> None:
    """Seed the first policy exactly once for an account alias."""
    encoded = canonical_policy_json(DEFAULT_POLICY)
    fingerprint = policy_hash(DEFAULT_POLICY)
    conn.execute(
        """
        INSERT INTO ips_policy_versions (
            account_alias, version, policy_json, policy_hash
        )
        SELECT ?, 1, ?, ?
        WHERE NOT EXISTS (
            SELECT 1 FROM ips_policy_versions WHERE account_alias = ?
        )
        """,
        (account_alias, encoded, fingerprint, account_alias),
    )


def get_active_policy(
    account_alias: str = "toss-brokerage",
    *,
    conn: sqlite3.Connection | None = None,
) -> dict[str, Any] | None:
    """Return the active policy and its replay metadata."""
    from storage.database import connect

    if conn is None:
        with connect() as owned_conn:
            owned_conn.execute("BEGIN IMMEDIATE")
            return get_active_policy(account_alias, conn=owned_conn)
    row = conn.execute(
        """
            SELECT id, account_alias, version, policy_json, policy_hash,
                   superseded_at, created_at
            FROM ips_policy_versions
            WHERE account_alias = ? AND superseded_at IS NULL
            ORDER BY version DESC, id DESC
            LIMIT 1
        """,
        (account_alias,),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "account_alias": row["account_alias"],
        "version": int(row["version"]),
        "policy": json.loads(row["policy_json"]),
        "policy_hash": row["policy_hash"],
        "superseded_at": row["superseded_at"],
        "created_at": row["created_at"],
    }


def get_policy_version(
    version_id: int, account_alias: str = "toss-brokerage"
) -> dict[str, Any] | None:
    """Return one immutable policy version for an anchored evaluation."""
    from storage.database import connect

    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, account_alias, version, policy_json, policy_hash,
                   superseded_at, created_at
            FROM ips_policy_versions
            WHERE id = ? AND account_alias = ?
            """,
            (version_id, account_alias),
        ).fetchone()
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "account_alias": row["account_alias"],
        "version": int(row["version"]),
        "policy": json.loads(row["policy_json"]),
        "policy_hash": row["policy_hash"],
        "superseded_at": row["superseded_at"],
        "created_at": row["created_at"],
    }


def list_observed_identities(
    account_alias: str = "toss-brokerage",
    *,
    conn: sqlite3.Connection | None = None,
) -> list[tuple[str, str]]:
    """Return every Toss-observed instrument identity in stable order."""
    from storage.database import connect

    if conn is not None:
        rows = conn.execute(
            """
            SELECT DISTINCT h.market_country, h.symbol
            FROM broker_holdings AS h
            JOIN broker_account_snapshots AS s ON s.id = h.snapshot_id
            WHERE s.account_alias = ?
            ORDER BY h.market_country, h.symbol
            """,
            (account_alias,),
        ).fetchall()
        return [(str(row["market_country"]), str(row["symbol"])) for row in rows]
    with connect() as owned_conn:
        return list_observed_identities(account_alias, conn=owned_conn)


def activate_policy(
    policy: dict[str, Any],
    expected_current_version: int,
    account_alias: str = "toss-brokerage",
) -> dict[str, Any]:
    """Atomically validate and activate one immutable policy version."""
    from services.policy_validation import policy_metadata, validate_policy

    normalized = validate_policy(policy, list_observed_identities(account_alias))
    metadata = policy_metadata(normalized)
    from storage.database import connect

    with connect() as conn:
        row = conn.execute(
            """
            SELECT id, version FROM ips_policy_versions
            WHERE account_alias = ? AND superseded_at IS NULL
            ORDER BY version DESC, id DESC LIMIT 1
            """,
            (account_alias,),
        ).fetchone()
        current_version = int(row["version"]) if row is not None else 0
        if current_version != int(expected_current_version):
            raise PolicyStoreError(
                f"active policy version conflict: expected {expected_current_version}, current {current_version}"
            )
        if row is not None:
            conn.execute(
                "UPDATE ips_policy_versions SET superseded_at = CURRENT_TIMESTAMP WHERE id = ?",
                (row["id"],),
            )
        cursor = conn.execute(
            """
            INSERT INTO ips_policy_versions (account_alias, version, policy_json, policy_hash)
            VALUES (?, ?, ?, ?)
            """,
            (
                account_alias,
                current_version + 1,
                metadata["policy_json"],
                metadata["policy_hash"],
            ),
        )
        new_id = int(cursor.lastrowid)
    active = get_active_policy(account_alias)
    if active is None or active["id"] != new_id:
        raise PolicyStoreError("activated policy could not be read back")
    return active


class PolicyStoreError(RuntimeError):
    """Raised when a policy persistence operation cannot be completed."""


def policy_template(
    snapshot_id: int | None = None,
    account_alias: str = "toss-brokerage",
) -> dict[str, Any]:
    """Build an app-owned target template from Toss identities only."""
    from services.account_projection import (
        build_account_projection,
        layer_map_from_policy,
    )

    active = get_active_policy(account_alias)
    if active is None:
        raise PolicyStoreError("active policy is required")
    active_policy = active["policy"]
    from services.policy_validation import PolicyValidationError, validate_policy

    try:
        normalized_active = validate_policy(
            active_policy, list_observed_identities(account_alias)
        )
    except PolicyValidationError as error:
        raise PolicyStoreError(f"active policy is invalid: {error}") from error
    projection = build_account_projection(
        snapshot_id=snapshot_id,
        account_alias=account_alias,
        layer_map=layer_map_from_policy(normalized_active),
    )
    policy = dict(normalized_active)
    policy["instruments"] = [
        {
            "market_country": position["market_country"],
            "symbol": position["symbol"],
            "layer": None,
            "minimum": None,
            "target": None,
            "maximum": None,
        }
        for position in projection["positions"]
    ]
    return {
        "account_alias": account_alias,
        "snapshot_id": projection["snapshot_id"],
        "active_policy_version": active["version"],
        "policy": policy,
    }
