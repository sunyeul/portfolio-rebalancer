"""Versioned Toss-only IPS policy persistence."""

from __future__ import annotations

import json
import sqlite3
from hashlib import sha256
from typing import Any


DEFAULT_POLICY: dict[str, Any] = {
    "cash_reserve": {"minimum": 0.10, "target": 0.15, "maximum": 0.20},
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
) -> dict[str, Any] | None:
    """Return the active policy and its replay metadata."""
    from storage.database import connect

    with connect() as conn:
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
