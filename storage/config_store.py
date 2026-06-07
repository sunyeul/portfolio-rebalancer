"""DB-backed application and IPS configuration helpers."""

from __future__ import annotations

import json
from typing import Any

from storage.database import connect, initialize_database


GROUP_TYPES = {"core", "satellite", "defensive", "cash", "unknown"}
OPTION_TABLES = {
    "groups": {
        "default": "ungrouped",
        "extra_columns": ("group_type",),
    },
    "thesis_statuses": {
        "default": "unknown",
        "extra_columns": (),
    },
}


class ConfigError(Exception):
    """Raised when config persistence cannot complete."""


def normalize_code(value: Any) -> str:
    return str(value or "").strip().lower()


def _row_to_option(row) -> dict[str, Any]:
    option = {
        "value": row["code"],
        "label": row["label"],
        "is_active": bool(row["is_active"]),
        "sort_order": row["sort_order"],
    }
    if "group_type" in row.keys():
        option["group_type"] = row["group_type"]
    return option


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


def upsert_option(
    table: str,
    code: str,
    label: str,
    sort_order: int = 999,
    is_active: bool = True,
    group_type: str | None = None,
) -> dict[str, Any]:
    if table not in OPTION_TABLES:
        raise ConfigError("지원하지 않는 옵션 테이블입니다.")
    code = normalize_code(code)
    label = label.strip()
    if not code:
        raise ConfigError("code를 입력해주세요.")
    if not label:
        raise ConfigError("label을 입력해주세요.")

    initialize_database()
    with connect() as conn:
        if table == "groups":
            next_group_type = normalize_code(group_type) or "unknown"
            if next_group_type not in GROUP_TYPES:
                raise ConfigError("지원하지 않는 group_type입니다.")
            conn.execute(
                """
                INSERT INTO groups (code, label, group_type, sort_order, is_active)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    label = excluded.label,
                    group_type = excluded.group_type,
                    sort_order = excluded.sort_order,
                    is_active = excluded.is_active
                """,
                (code, label, next_group_type, sort_order, 1 if is_active else 0),
            )
        else:
            conn.execute(
                f"""
                INSERT INTO {table} (code, label, sort_order, is_active)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    label = excluded.label,
                    sort_order = excluded.sort_order,
                    is_active = excluded.is_active
                """,
                (code, label, sort_order, 1 if is_active else 0),
            )
        row = conn.execute(
            f"SELECT * FROM {table} WHERE code = ?",
            (code,),
        ).fetchone()
    return _row_to_option(row)


def set_option_active(table: str, code: str, is_active: bool) -> dict[str, Any]:
    if table not in OPTION_TABLES:
        raise ConfigError("지원하지 않는 옵션 테이블입니다.")
    code = normalize_code(code)
    initialize_database()
    with connect() as conn:
        conn.execute(
            f"UPDATE {table} SET is_active = ? WHERE code = ?",
            (1 if is_active else 0, code),
        )
        row = conn.execute(f"SELECT * FROM {table} WHERE code = ?", (code,)).fetchone()
    if row is None:
        raise ConfigError("옵션을 찾을 수 없습니다.")
    return _row_to_option(row)


def get_ips_config() -> dict[str, Any]:
    initialize_database()
    with connect() as conn:
        target_rows = conn.execute(
            "SELECT * FROM ips_target_allocations ORDER BY group_type ASC"
        ).fetchall()
        group_rows = conn.execute("SELECT code, group_type FROM groups").fetchall()
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
            row["group_type"]: {
                "min": row["min"],
                "target": row["target"],
                "max": row["max"],
            }
            for row in target_rows
        },
        "groups": {
            row["code"]: {"type": row["group_type"]}
            for row in group_rows
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
            "SELECT * FROM ips_target_allocations ORDER BY group_type ASC"
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
                "group_type": row["group_type"],
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
            group_type = normalize_code(row.get("group_type"))
            if group_type not in GROUP_TYPES:
                raise ConfigError("지원하지 않는 group_type입니다.")
            min_value = float(row.get("min"))
            target_value = float(row.get("target"))
            max_value = float(row.get("max"))
            if not (0 <= min_value <= target_value <= max_value <= 1):
                raise ConfigError("목표 비중은 0~1 범위에서 min <= target <= max 여야 합니다.")
            conn.execute(
                """
                INSERT INTO ips_target_allocations (group_type, min, target, max)
                VALUES (?, ?, ?, ?)
                """,
                (group_type, min_value, target_value, max_value),
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
