"""DB-backed application and IPS configuration helpers."""

from __future__ import annotations

import json
from typing import Any

from core.asset import VALID_GROUPS
from storage.database import connect, initialize_database


TARGET_GROUPS = VALID_GROUPS
OPTION_TABLES = {
    "thesis_statuses": {
        "default": "unknown",
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
            'SELECT * FROM ips_target_allocations ORDER BY "group" ASC'
        ).fetchall()
        priority_rows = conn.execute(
            """
            SELECT action_code, priority
            FROM ips_action_priorities
            WHERE is_active = 1
            ORDER BY priority ASC, action_code ASC
            """
        ).fetchall()
        rule_rows = conn.execute("SELECT key, value_json FROM ips_rules").fetchall()

    return {
        "target_allocation": {
            row["group"]: {
                "min": row["min"],
                "target": row["target"],
                "max": row["max"],
            }
            for row in target_rows
        },
        "action_priority": {
            row["action_code"]: row["priority"]
            for row in priority_rows
        },
        "rules": {
            row["key"]: json.loads(row["value_json"])
            for row in rule_rows
        },
    }


def get_ips_management_config() -> dict[str, Any]:
    initialize_database()
    with connect() as conn:
        targets = conn.execute(
            'SELECT * FROM ips_target_allocations ORDER BY "group" ASC'
        ).fetchall()
        priorities = conn.execute(
            """
            SELECT *
            FROM ips_action_priorities
            ORDER BY priority ASC, action_code ASC
            """
        ).fetchall()
        rules = conn.execute("SELECT * FROM ips_rules ORDER BY key ASC").fetchall()
    return {
        "target_allocations": [
            {
                "group": row["group"],
                "min": row["min"],
                "target": row["target"],
                "max": row["max"],
            }
            for row in targets
        ],
        "action_priorities": [
            {
                "action_code": row["action_code"],
                "label": row["label"],
                "priority": row["priority"],
                "is_active": bool(row["is_active"]),
            }
            for row in priorities
        ],
        "rules": [
            {
                "key": row["key"],
                "value": json.loads(row["value_json"]),
            }
            for row in rules
        ],
        "ips_config": get_ips_config(),
    }


def replace_target_allocations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initialize_database()
    with connect() as conn:
        conn.execute("DELETE FROM ips_target_allocations")
        for row in rows:
            group = normalize_code(row.get("group"))
            if group not in TARGET_GROUPS:
                raise ConfigError("지원하지 않는 group입니다.")
            min_value = float(row.get("min"))
            target_value = float(row.get("target"))
            max_value = float(row.get("max"))
            if not (0 <= min_value <= target_value <= max_value <= 1):
                raise ConfigError("목표 비중은 0~1 범위에서 min <= target <= max 여야 합니다.")
            conn.execute(
                """
                INSERT INTO ips_target_allocations ("group", min, target, max)
                VALUES (?, ?, ?, ?)
                """,
                (group, min_value, target_value, max_value),
            )
    return get_ips_management_config()["target_allocations"]


def replace_action_priorities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initialize_database()
    with connect() as conn:
        conn.execute("DELETE FROM ips_action_priorities")
        for row in rows:
            action_code = normalize_code(row.get("action_code"))
            label = str(row.get("label") or "").strip()
            if not action_code or not label:
                raise ConfigError("action_code와 label을 입력해주세요.")
            conn.execute(
                """
                INSERT INTO ips_action_priorities (
                    action_code,
                    label,
                    priority,
                    is_active
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    action_code,
                    label,
                    int(row.get("priority", 99)),
                    1 if row.get("is_active", True) else 0,
                ),
            )
    return get_ips_management_config()["action_priorities"]


def replace_rules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    initialize_database()
    with connect() as conn:
        conn.execute("DELETE FROM ips_rules")
        for row in rows:
            key = normalize_code(row.get("key"))
            if not key:
                raise ConfigError("rule key를 입력해주세요.")
            conn.execute(
                "INSERT INTO ips_rules (key, value_json) VALUES (?, ?)",
                (key, json.dumps(row.get("value"), ensure_ascii=False)),
            )
    return get_ips_management_config()["rules"]
