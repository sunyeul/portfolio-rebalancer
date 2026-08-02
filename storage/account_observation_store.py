"""Immutable persistence for normalized Toss account observations."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from integrations.toss.observation import NormalizedSnapshot, SyncState


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def insert_snapshot(snapshot: NormalizedSnapshot) -> dict[str, Any]:
    """Insert one immutable snapshot or return the existing fingerprint match."""
    from storage.database import connect

    with connect() as conn:
        existing = conn.execute(
            """
            SELECT id
            FROM broker_account_snapshots
            WHERE account_alias = ? AND source_fingerprint = ?
            """,
            (snapshot.account_alias, snapshot.fingerprint),
        ).fetchone()
        if existing is not None:
            if snapshot.state != SyncState.COMPLETE:
                conn.execute(
                    "UPDATE broker_account_snapshots SET is_current_evaluable = 0 WHERE account_alias = ?",
                    (snapshot.account_alias,),
                )
            return get_snapshot(int(existing["id"]), conn=conn)

        conn.execute(
            "UPDATE broker_account_snapshots SET is_current_evaluable = 0 WHERE account_alias = ?",
            (snapshot.account_alias,),
        )

        cursor = conn.execute(
            """
            INSERT INTO broker_account_snapshots (
                account_alias, sync_started_at, synced_at, state,
                is_current_evaluable, source_fingerprint,
                source_timestamps_json, data_quality_json, reconciliation_json,
                total_value_krw, invested_value_krw, cash_value_krw
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.account_alias,
                snapshot.sync_started_at,
                snapshot.synced_at,
                snapshot.state.value,
                int(snapshot.state == SyncState.COMPLETE),
                snapshot.fingerprint,
                _json(snapshot.source_timestamps),
                _json(snapshot.data_quality),
                _json(snapshot.reconciliation),
                snapshot.total_value_krw,
                snapshot.invested_value_krw,
                snapshot.cash_value_krw,
            ),
        )
        snapshot_id = int(cursor.lastrowid)
        conn.executemany(
            """
            INSERT INTO broker_holdings (
                snapshot_id, symbol, name, market_country, currency, quantity,
                last_price, average_purchase_price, market_value_native,
                market_value_krw, cost_native, cost_krw, profit_loss_native,
                profit_loss_krw, daily_profit_loss_native, daily_profit_loss_krw
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    item.symbol,
                    item.name,
                    item.market_country,
                    item.currency,
                    item.quantity,
                    item.last_price,
                    item.average_purchase_price,
                    item.market_value_native,
                    item.market_value_krw,
                    item.cost_native,
                    item.cost_krw,
                    item.profit_loss_native,
                    item.profit_loss_krw,
                    item.daily_profit_loss_native,
                    item.daily_profit_loss_krw,
                )
                for item in snapshot.holdings
            ],
        )
        conn.executemany(
            """
            INSERT INTO broker_cash_observations (
                snapshot_id, currency, buying_power_native, buying_power_krw
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    item.currency,
                    item.buying_power_native,
                    item.buying_power_krw,
                )
                for item in snapshot.cash
            ],
        )
        if snapshot.fx_rate is not None:
            conn.execute(
                """
                INSERT INTO broker_exchange_rates (
                    snapshot_id, base_currency, quote_currency, rate, mid_rate,
                    valid_from, valid_until
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    snapshot.fx_rate.base_currency,
                    snapshot.fx_rate.quote_currency,
                    snapshot.fx_rate.rate,
                    snapshot.fx_rate.mid_rate,
                    snapshot.fx_rate.valid_from,
                    snapshot.fx_rate.valid_until,
                ),
            )
        conn.executemany(
            """
            INSERT INTO broker_orders (
                snapshot_id, order_id, symbol, currency, side, order_type, status,
                ordered_at, canceled_at, quantity, order_price_native,
                order_amount_native, filled_quantity, average_filled_price_native,
                filled_amount_native, commission_native, tax_native, filled_at,
                settlement_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    item.order_id,
                    item.symbol,
                    item.currency,
                    item.side,
                    item.order_type,
                    item.status,
                    item.ordered_at,
                    item.canceled_at,
                    item.quantity,
                    item.order_price_native,
                    item.order_amount_native,
                    item.filled_quantity,
                    item.average_filled_price_native,
                    item.filled_amount_native,
                    item.commission_native,
                    item.tax_native,
                    item.filled_at,
                    item.settlement_date,
                )
                for item in snapshot.orders
            ],
        )
        return get_snapshot(snapshot_id, conn=conn)


def _row_to_snapshot(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    snapshot_id = int(row["id"])
    holdings = [
        dict(item)
        for item in conn.execute(
            """
            SELECT symbol, name, market_country, currency, quantity, last_price,
                   average_purchase_price, market_value_native, market_value_krw,
                   cost_native, cost_krw, profit_loss_native, profit_loss_krw,
                   daily_profit_loss_native, daily_profit_loss_krw
            FROM broker_holdings WHERE snapshot_id = ? ORDER BY id
            """,
            (snapshot_id,),
        ).fetchall()
    ]
    cash = [
        dict(item)
        for item in conn.execute(
            """
            SELECT currency, buying_power_native, buying_power_krw
            FROM broker_cash_observations WHERE snapshot_id = ? ORDER BY id
            """,
            (snapshot_id,),
        ).fetchall()
    ]
    fx = conn.execute(
        """
        SELECT base_currency, quote_currency, rate, mid_rate, valid_from, valid_until
        FROM broker_exchange_rates WHERE snapshot_id = ? ORDER BY id LIMIT 1
        """,
        (snapshot_id,),
    ).fetchone()
    orders = [
        dict(item)
        for item in conn.execute(
            """
            SELECT order_id, symbol, currency, side, order_type, status,
                   ordered_at, canceled_at, quantity, order_price_native,
                   order_amount_native, filled_quantity, average_filled_price_native,
                   filled_amount_native, commission_native, tax_native, filled_at,
                   settlement_date
            FROM broker_orders WHERE snapshot_id = ? ORDER BY id
            """,
            (snapshot_id,),
        ).fetchall()
    ]
    return {
        "id": snapshot_id,
        "account_alias": row["account_alias"],
        "sync_started_at": row["sync_started_at"],
        "synced_at": row["synced_at"],
        "state": row["state"],
        "is_current_evaluable": bool(row["is_current_evaluable"]),
        "source_fingerprint": row["source_fingerprint"],
        "source_timestamps": json.loads(row["source_timestamps_json"]),
        "data_quality": json.loads(row["data_quality_json"]),
        "reconciliation": json.loads(row["reconciliation_json"]),
        "total_value_krw": row["total_value_krw"],
        "invested_value_krw": row["invested_value_krw"],
        "cash_value_krw": row["cash_value_krw"],
        "holdings": holdings,
        "cash": cash,
        "fx_rate": dict(fx) if fx is not None else None,
        "orders": orders,
    }


def get_snapshot(
    snapshot_id: int, *, conn: sqlite3.Connection | None = None
) -> dict[str, Any] | None:
    """Return one normalized snapshot without raw broker identifiers."""
    from storage.database import connect

    if conn is not None:
        row = conn.execute(
            "SELECT * FROM broker_account_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        return _row_to_snapshot(conn, row) if row is not None else None
    with connect() as owned_conn:
        row = owned_conn.execute(
            "SELECT * FROM broker_account_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        return _row_to_snapshot(owned_conn, row) if row is not None else None


def latest_complete(account_alias: str = "toss-brokerage") -> dict[str, Any] | None:
    """Return the sole current evaluable complete snapshot."""
    from storage.database import connect

    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM broker_account_snapshots
            WHERE account_alias = ? AND state = 'complete'
              AND is_current_evaluable = 1
            ORDER BY synced_at DESC, id DESC LIMIT 1
            """,
            (account_alias,),
        ).fetchone()
        return _row_to_snapshot(conn, row) if row is not None else None


def latest_verified_complete(
    account_alias: str = "toss-brokerage",
) -> dict[str, Any] | None:
    """Return the newest complete snapshot, even when a newer attempt failed."""
    from storage.database import connect

    with connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM broker_account_snapshots
            WHERE account_alias = ? AND state = 'complete'
            ORDER BY synced_at DESC, id DESC LIMIT 1
            """,
            (account_alias,),
        ).fetchone()
        return _row_to_snapshot(conn, row) if row is not None else None


def list_complete_snapshots(
    account_alias: str = "toss-brokerage",
) -> list[dict[str, Any]]:
    """Return complete account snapshots in deterministic projection order."""
    from storage.database import connect

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM broker_account_snapshots
            WHERE account_alias = ? AND state = 'complete'
            ORDER BY synced_at ASC, id ASC
            """,
            (account_alias,),
        ).fetchall()
        return [_row_to_snapshot(conn, row) for row in rows]


def list_snapshots(
    limit: int = 20, account_alias: str | None = None
) -> list[dict[str, Any]]:
    """Return recent normalized snapshots, newest first."""
    from storage.database import connect

    bounded_limit = max(1, min(int(limit), 100))
    with connect() as conn:
        if account_alias is None:
            rows = conn.execute(
                """
                SELECT * FROM broker_account_snapshots
                ORDER BY synced_at DESC, id DESC LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM broker_account_snapshots WHERE account_alias = ?
                ORDER BY synced_at DESC, id DESC LIMIT ?
                """,
                (account_alias, bounded_limit),
            ).fetchall()
        return [_row_to_snapshot(conn, row) for row in rows]
