"""DB-backed application and IPS configuration helpers."""

from __future__ import annotations

from typing import Any

from core.asset import LAYER_TYPES
from storage.database import connect, initialize_database


TARGET_LAYERS = LAYER_TYPES
OPTION_TABLES = {
    "thesis_statuses": {
        "default": "valid",
    },
}


class ConfigError(Exception):
    """Raised when config persistence cannot complete."""


def normalize_code(value: Any) -> str:
    return str(value or "").strip().lower()


def _row_to_option(row) -> dict[str, Any]:
    return {
        "value": row["code"],
        "label": row["label"],
        "is_active": bool(row["is_active"]),
        "sort_order": row["sort_order"],
    }


def list_options(include_inactive: bool = True) -> dict[str, list[dict[str, Any]]]:
    initialize_database()
    result: dict[str, list[dict[str, Any]]] = {}
    inactive_clause = "" if include_inactive else "WHERE is_active = 1"
    with connect() as conn:
        for table in OPTION_TABLES:
            rows = conn.execute(
                f"""
                SELECT *
                FROM {table}
                {inactive_clause}
                ORDER BY sort_order ASC, code ASC
                """
            ).fetchall()
            result[table] = [_row_to_option(row) for row in rows]
    return result


def active_codes(table: str) -> set[str]:
    if table not in OPTION_TABLES:
        raise ConfigError("지원하지 않는 옵션 테이블입니다.")
    initialize_database()
    with connect() as conn:
        rows = conn.execute(
            f"SELECT code FROM {table} WHERE is_active = 1"
        ).fetchall()
    return {row["code"] for row in rows}


def get_ips_config() -> dict[str, Any]:
    initialize_database()
    with connect() as conn:
        target_rows = conn.execute(
            "SELECT * FROM ips_target_allocations ORDER BY layer ASC"
        ).fetchall()

    return {
        "target_allocation": {
            row["layer"]: {
                "min": row["min"],
                "target": row["target"],
                "max": row["max"],
            }
            for row in target_rows
        },
    }


def get_ips_management_config() -> dict[str, Any]:
    initialize_database()
    with connect() as conn:
        targets = conn.execute(
            "SELECT * FROM ips_target_allocations ORDER BY layer ASC"
        ).fetchall()
    return {
        "target_allocations": [
            {
                "layer": row["layer"],
                "min": row["min"],
                "target": row["target"],
                "max": row["max"],
            }
            for row in targets
        ],
        "ips_config": get_ips_config(),
    }


def replace_target_allocations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initialize_database()
    with connect() as conn:
        conn.execute("DELETE FROM ips_target_allocations")
        for row in rows:
            layer = normalize_code(row.get("layer"))
            if layer not in TARGET_LAYERS:
                raise ConfigError("지원하지 않는 layer입니다.")
            min_value = float(row.get("min"))
            target_value = float(row.get("target"))
            max_value = float(row.get("max"))
            if not (0 <= min_value <= target_value <= max_value <= 1):
                raise ConfigError("목표 비중은 0~1 범위에서 min <= target <= max 여야 합니다.")
            conn.execute(
                """
                INSERT INTO ips_target_allocations (layer, min, target, max)
                VALUES (?, ?, ?, ?)
                """,
                (layer, min_value, target_value, max_value),
            )
    return get_ips_management_config()["target_allocations"]
