"""Repository helpers for saved portfolios and snapshots."""

from __future__ import annotations

import json
import re
from typing import Any

from api.v1.serialization import json_safe
from storage.database import connect, initialize_database


TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,15}$")


class StorageError(Exception):
    """Raised when a portfolio persistence operation cannot be completed."""


def _json_dump(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False)


def _json_load(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return json_safe(value)


def _normalize_code(value: Any, default: str) -> str:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower()


def _ensure_lookup(conn, table: str, code: str, default_code: str) -> int:
    code = _normalize_code(code, default_code)
    row = conn.execute(f"SELECT id FROM {table} WHERE code = ?", (code,)).fetchone()
    if row:
        return int(row["id"])
    label = code.replace("_", " ")
    cursor = conn.execute(
        f"""
        INSERT INTO {table} (code, label, sort_order, is_active)
        VALUES (?, ?, 999, 1)
        """,
        (code, label),
    )
    return int(cursor.lastrowid)


def _ensure_asset(conn, ticker: str) -> int:
    ticker = str(ticker).strip().upper()
    if not TICKER_RE.match(ticker):
        raise StorageError(f"유효하지 않은 티커입니다: {ticker}")
    row = conn.execute("SELECT id FROM assets WHERE ticker = ?", (ticker,)).fetchone()
    if row:
        return int(row["id"])
    cursor = conn.execute(
        "INSERT INTO assets (ticker, display_name) VALUES (?, ?)",
        (ticker, ticker),
    )
    return int(cursor.lastrowid)


def _state_payload(session_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    analysis_payload = None
    if session_state.get("metrics_df"):
        analysis_payload = {
            "metrics_df": session_state.get("metrics_df") or [],
            "portfolio_metrics": session_state.get("portfolio_metrics"),
            "benchmark_metrics": session_state.get("benchmark_metrics"),
            "missing_tickers": session_state.get("missing_tickers") or [],
        }

    evaluation_payload = None
    if session_state.get("proposal_df"):
        evaluation_payload = {
            "proposal_df": session_state.get("proposal_df") or [],
            "ips_action_df": session_state.get("ips_action_df") or [],
            "group_summary_df": session_state.get("group_summary_df") or [],
            "rc_violations": session_state.get("rc_violations") or [],
            "ips_config_snapshot": session_state.get("ips_config_snapshot"),
        }

    return session_state, analysis_payload, evaluation_payload


def create_portfolio(name: str, description: str = "") -> dict[str, Any]:
    initialize_database()
    name = name.strip()
    if not name:
        raise StorageError("포트폴리오 이름을 입력해주세요.")
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO portfolios (name, description)
            VALUES (?, ?)
            """,
            (name, description.strip()),
        )
        portfolio_id = int(cursor.lastrowid)
    portfolio = get_portfolio(portfolio_id)
    if portfolio is None:
        raise StorageError("포트폴리오 생성 결과를 찾을 수 없습니다.")
    return portfolio


def update_portfolio(
    portfolio_id: int,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    initialize_database()
    current = get_portfolio(portfolio_id)
    if current is None:
        raise StorageError("포트폴리오를 찾을 수 없습니다.")
    next_name = current["name"] if name is None else name.strip()
    next_description = (
        current["description"] if description is None else description.strip()
    )
    if not next_name:
        raise StorageError("포트폴리오 이름을 입력해주세요.")
    with connect() as conn:
        conn.execute(
            """
            UPDATE portfolios
            SET name = ?, description = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND is_active = 1
            """,
            (next_name, next_description, portfolio_id),
        )
    updated = get_portfolio(portfolio_id)
    if updated is None:
        raise StorageError("포트폴리오 수정 결과를 찾을 수 없습니다.")
    return updated


def list_portfolios() -> list[dict[str, Any]]:
    initialize_database()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                p.id,
                p.name,
                p.description,
                p.created_at,
                p.updated_at,
                s.id AS latest_snapshot_id,
                s.name AS latest_snapshot_name,
                s.created_at AS latest_snapshot_created_at,
                COUNT(pos.id) AS latest_position_count
            FROM portfolios p
            LEFT JOIN portfolio_snapshots s
                ON s.id = (
                    SELECT id
                    FROM portfolio_snapshots
                    WHERE portfolio_id = p.id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
            LEFT JOIN snapshot_positions pos ON pos.snapshot_id = s.id
            WHERE p.is_active = 1
            GROUP BY p.id, s.id
            ORDER BY p.updated_at DESC, p.id DESC
            """
        ).fetchall()
    return [_portfolio_row(row) for row in rows]


def get_portfolio(portfolio_id: int) -> dict[str, Any] | None:
    initialize_database()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT
                p.id,
                p.name,
                p.description,
                p.created_at,
                p.updated_at,
                s.id AS latest_snapshot_id,
                s.name AS latest_snapshot_name,
                s.created_at AS latest_snapshot_created_at,
                COUNT(pos.id) AS latest_position_count
            FROM portfolios p
            LEFT JOIN portfolio_snapshots s
                ON s.id = (
                    SELECT id
                    FROM portfolio_snapshots
                    WHERE portfolio_id = p.id
                    ORDER BY created_at DESC, id DESC
                    LIMIT 1
                )
            LEFT JOIN snapshot_positions pos ON pos.snapshot_id = s.id
            WHERE p.id = ? AND p.is_active = 1
            GROUP BY p.id, s.id
            """,
            (portfolio_id,),
        ).fetchone()
    return _portfolio_row(row) if row else None


def save_current_state(
    portfolio_id: int,
    session_data: dict[str, Any],
) -> dict[str, Any]:
    initialize_database()
    if get_portfolio(portfolio_id) is None:
        raise StorageError("포트폴리오를 찾을 수 없습니다.")
    asset_rows = session_data.get("asset_df")
    if not asset_rows:
        raise StorageError("저장할 포트폴리오 입력이 없습니다.")

    state_json = _json_dump(session_data)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO portfolio_current_states (portfolio_id, state_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(portfolio_id) DO UPDATE SET
                state_json = excluded.state_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (portfolio_id, state_json),
        )
        conn.execute(
            "UPDATE portfolios SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (portfolio_id,),
        )

    current_state = get_current_state(portfolio_id)
    if current_state is None:
        raise StorageError("현재 상태 저장 결과를 찾을 수 없습니다.")
    return current_state


def get_current_state(portfolio_id: int) -> dict[str, Any] | None:
    initialize_database()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT portfolio_id, state_json, updated_at
            FROM portfolio_current_states
            WHERE portfolio_id = ?
            """,
            (portfolio_id,),
        ).fetchone()
    if row is None:
        return None

    session_state, analysis_payload, evaluation_payload = _state_payload(
        _json_load(row["state_json"], {})
    )
    return {
        "portfolio_id": row["portfolio_id"],
        "updated_at": row["updated_at"],
        "session_state": session_state,
        "analysis": analysis_payload,
        "evaluation": evaluation_payload,
    }


def _portfolio_row(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "latest_snapshot": (
            {
                "id": row["latest_snapshot_id"],
                "name": row["latest_snapshot_name"],
                "created_at": row["latest_snapshot_created_at"],
                "position_count": row["latest_position_count"],
            }
            if row["latest_snapshot_id"] is not None
            else None
        ),
    }


def list_snapshots(portfolio_id: int) -> list[dict[str, Any]]:
    initialize_database()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.portfolio_id,
                s.name,
                s.note,
                s.created_at,
                COUNT(pos.id) AS position_count,
                CASE WHEN ar.id IS NULL THEN 0 ELSE 1 END AS has_analysis,
                CASE WHEN er.id IS NULL THEN 0 ELSE 1 END AS has_evaluation
            FROM portfolio_snapshots s
            LEFT JOIN snapshot_positions pos ON pos.snapshot_id = s.id
            LEFT JOIN analysis_runs ar ON ar.snapshot_id = s.id
            LEFT JOIN evaluation_runs er ON er.snapshot_id = s.id
            WHERE s.portfolio_id = ?
            GROUP BY s.id
            ORDER BY s.created_at DESC, s.id DESC
            """,
            (portfolio_id,),
        ).fetchall()
    return [_snapshot_summary(row) for row in rows]


def _snapshot_summary(row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "portfolio_id": row["portfolio_id"],
        "name": row["name"],
        "note": row["note"],
        "created_at": row["created_at"],
        "position_count": row["position_count"],
        "has_analysis": bool(row["has_analysis"]),
        "has_evaluation": bool(row["has_evaluation"]),
    }


def create_snapshot(
    portfolio_id: int,
    name: str,
    note: str,
    session_data: dict[str, Any],
) -> dict[str, Any]:
    initialize_database()
    if get_portfolio(portfolio_id) is None:
        raise StorageError("포트폴리오를 찾을 수 없습니다.")
    asset_rows = session_data.get("asset_df")
    if not asset_rows:
        raise StorageError("저장할 포트폴리오 입력이 없습니다.")

    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO portfolio_snapshots (portfolio_id, name, note)
            VALUES (?, ?, ?)
            """,
            (portfolio_id, name.strip() or "저장된 스냅샷", note.strip()),
        )
        snapshot_id = int(cursor.lastrowid)
        _insert_positions(conn, snapshot_id, asset_rows)
        _insert_analysis(conn, snapshot_id, session_data)
        _insert_evaluation(conn, snapshot_id, session_data)
        conn.execute(
            "UPDATE portfolios SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (portfolio_id,),
        )
    snapshot = get_snapshot(snapshot_id)
    if snapshot is None:
        raise StorageError("스냅샷 저장 결과를 찾을 수 없습니다.")
    return snapshot["summary"]


def update_snapshot(
    snapshot_id: int,
    name: str | None = None,
    note: str | None = None,
    asset_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    initialize_database()
    current = get_snapshot(snapshot_id)
    if current is None:
        raise StorageError("스냅샷을 찾을 수 없습니다.")
    current_summary = current["summary"]
    next_name = current_summary["name"] if name is None else name.strip() or "저장된 스냅샷"
    next_note = current_summary["note"] if note is None else note.strip()
    with connect() as conn:
        conn.execute(
            """
            UPDATE portfolio_snapshots
            SET name = ?, note = ?
            WHERE id = ?
            """,
            (next_name, next_note, snapshot_id),
        )
        if asset_rows is not None:
            conn.execute(
                "DELETE FROM analysis_runs WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            conn.execute(
                "DELETE FROM evaluation_runs WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            conn.execute(
                "DELETE FROM snapshot_positions WHERE snapshot_id = ?",
                (snapshot_id,),
            )
            _insert_positions(conn, snapshot_id, asset_rows)
        conn.execute(
            "UPDATE portfolios SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (current_summary["portfolio_id"],),
        )
    updated = get_snapshot(snapshot_id)
    if updated is None:
        raise StorageError("스냅샷 수정 결과를 찾을 수 없습니다.")
    return updated["summary"]


def delete_snapshot(snapshot_id: int) -> None:
    initialize_database()
    with connect() as conn:
        row = conn.execute(
            "SELECT portfolio_id FROM portfolio_snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise StorageError("스냅샷을 찾을 수 없습니다.")
        portfolio_id = int(row["portfolio_id"])
        conn.execute("DELETE FROM portfolio_snapshots WHERE id = ?", (snapshot_id,))
        conn.execute(
            "UPDATE portfolios SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (portfolio_id,),
        )


def _insert_positions(conn, snapshot_id: int, asset_rows: list[dict[str, Any]]) -> None:
    for position_order, row in enumerate(asset_rows):
        asset_id = _ensure_asset(conn, row["ticker"])
        group_id = _ensure_lookup(conn, "groups", row.get("group"), "ungrouped")
        thesis_id = _ensure_lookup(
            conn, "thesis_statuses", row.get("thesis_status"), "unknown"
        )
        conn.execute(
            """
            INSERT INTO snapshot_positions (
                snapshot_id,
                asset_id,
                allocation,
                weight,
                return_total,
                group_id,
                dca_enabled,
                thesis_status_id,
                position_order
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                asset_id,
                row.get("allocation", 0),
                row.get("weight", 0),
                row.get("return_total"),
                group_id,
                1 if row.get("dca_enabled", True) else 0,
                thesis_id,
                position_order,
            ),
        )


def _insert_analysis(
    conn,
    snapshot_id: int,
    session_data: dict[str, Any],
) -> int | None:
    metrics_rows = session_data.get("metrics_df")
    if not metrics_rows:
        return None

    settings = session_data.get("analysis_settings") or {}
    cursor = conn.execute(
        """
        INSERT INTO analysis_runs (
            snapshot_id,
            period,
            rf,
            bench,
            portfolio_metrics_json,
            benchmark_metrics_json,
            missing_tickers_json,
            returns_smooth_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot_id,
            str(settings.get("period")) if settings.get("period") is not None else None,
            settings.get("rf"),
            settings.get("bench"),
            _json_dump(session_data.get("portfolio_metrics")),
            _json_dump(session_data.get("benchmark_metrics")),
            _json_dump(session_data.get("missing_tickers", [])),
            _json_dump(session_data.get("returns_smooth")),
        ),
    )
    analysis_run_id = int(cursor.lastrowid)
    for row in metrics_rows:
        ticker = row.get("ticker")
        asset_id = _ensure_asset(conn, ticker) if ticker else None
        conn.execute(
            """
            INSERT INTO analysis_metrics (
                analysis_run_id,
                asset_id,
                ticker,
                cagr,
                volatility,
                sharpe,
                max_drawdown,
                information_ratio,
                beta,
                alpha,
                risk_contribution,
                return_contribution,
                weight,
                efficiency_score,
                return_total,
                record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_run_id,
                asset_id,
                ticker,
                row.get("CAGR"),
                row.get("변동성"),
                row.get("샤프"),
                row.get("최대낙폭"),
                row.get("IR"),
                row.get("베타"),
                row.get("알파"),
                row.get("위험기여도"),
                row.get("수익기여도"),
                row.get("가중치"),
                row.get("E"),
                row.get("return_total"),
                _json_dump(row),
            ),
        )
    return analysis_run_id


def _insert_evaluation(
    conn,
    snapshot_id: int,
    session_data: dict[str, Any],
) -> None:
    proposal_rows = session_data.get("proposal_df")
    if not proposal_rows:
        return

    settings = session_data.get("evaluation_settings") or {}
    cursor = conn.execute(
        """
        INSERT INTO evaluation_runs (
            snapshot_id,
            rc_over_thresh_pct,
            e_thresh,
            target_weights_json,
            ips_config_snapshot_json
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            snapshot_id,
            settings.get("rc_over_thresh_pct"),
            settings.get("e_thresh"),
            _json_dump(settings.get("target_weights")),
            _json_dump(session_data.get("ips_config_snapshot")),
        ),
    )
    evaluation_run_id = int(cursor.lastrowid)
    for row in proposal_rows:
        ticker = row.get("ticker")
        asset_id = _ensure_asset(conn, ticker) if ticker else None
        conn.execute(
            """
            INSERT INTO evaluation_rows (
                evaluation_run_id,
                asset_id,
                ticker,
                current_weight_pct,
                target_weight_pct,
                gap_pct,
                adjusted_gap_pct,
                rc_over_pct,
                rc_target_pct,
                should_execute,
                record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evaluation_run_id,
                asset_id,
                ticker,
                row.get("현재%"),
                row.get("목표%"),
                row.get("갭%"),
                row.get("제안조정%"),
                row.get("RC_Over%"),
                row.get("RC_Target%"),
                1 if row.get("실행") else 0,
                _json_dump(row),
            ),
        )
    _insert_json_rows(
        conn,
        "ips_action_rows",
        evaluation_run_id,
        session_data.get("ips_action_df") or [],
        "ticker",
    )
    _insert_json_rows(
        conn,
        "group_summary_rows",
        evaluation_run_id,
        session_data.get("group_summary_df") or [],
        "group",
    )
    _insert_json_rows(
        conn,
        "rc_violation_rows",
        evaluation_run_id,
        session_data.get("rc_violations") or [],
        "ticker",
    )


def _insert_json_rows(
    conn,
    table: str,
    evaluation_run_id: int,
    rows: list[dict[str, Any]],
    key_name: str,
) -> None:
    for row in rows:
        if table == "group_summary_rows":
            conn.execute(
                """
                INSERT INTO group_summary_rows (
                    evaluation_run_id,
                    group_type,
                    group_code,
                    record_json
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    evaluation_run_id,
                    row.get("group_type"),
                    row.get(key_name),
                    _json_dump(row),
                ),
            )
        else:
            conn.execute(
                f"""
                INSERT INTO {table} (evaluation_run_id, ticker, record_json)
                VALUES (?, ?, ?)
                """,
                (evaluation_run_id, row.get(key_name), _json_dump(row)),
            )


def get_snapshot(snapshot_id: int) -> dict[str, Any] | None:
    initialize_database()
    with connect() as conn:
        summary_row = conn.execute(
            """
            SELECT
                s.id,
                s.portfolio_id,
                s.name,
                s.note,
                s.created_at,
                COUNT(pos.id) AS position_count,
                CASE WHEN ar.id IS NULL THEN 0 ELSE 1 END AS has_analysis,
                CASE WHEN er.id IS NULL THEN 0 ELSE 1 END AS has_evaluation
            FROM portfolio_snapshots s
            LEFT JOIN snapshot_positions pos ON pos.snapshot_id = s.id
            LEFT JOIN analysis_runs ar ON ar.snapshot_id = s.id
            LEFT JOIN evaluation_runs er ON er.snapshot_id = s.id
            WHERE s.id = ?
            GROUP BY s.id
            """,
            (snapshot_id,),
        ).fetchone()
        if summary_row is None:
            return None

        positions = conn.execute(
            """
            SELECT
                a.ticker,
                pos.allocation,
                pos.weight,
                pos.return_total,
                g.code AS group_code,
                pos.dca_enabled,
                ts.code AS thesis_status_code
            FROM snapshot_positions pos
            JOIN assets a ON a.id = pos.asset_id
            JOIN groups g ON g.id = pos.group_id
            JOIN thesis_statuses ts ON ts.id = pos.thesis_status_id
            WHERE pos.snapshot_id = ?
            ORDER BY pos.position_order ASC, a.ticker ASC
            """,
            (snapshot_id,),
        ).fetchall()

        analysis_run = conn.execute(
            "SELECT * FROM analysis_runs WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        analysis_metrics = []
        if analysis_run:
            analysis_metrics = conn.execute(
                """
                SELECT record_json
                FROM analysis_metrics
                WHERE analysis_run_id = ?
                ORDER BY id ASC
                """,
                (analysis_run["id"],),
            ).fetchall()

        evaluation_run = conn.execute(
            "SELECT * FROM evaluation_runs WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        evaluation_rows = []
        ips_action_rows = []
        group_summary_rows = []
        rc_violation_rows = []
        if evaluation_run:
            evaluation_rows = conn.execute(
                """
                SELECT record_json
                FROM evaluation_rows
                WHERE evaluation_run_id = ?
                ORDER BY id ASC
                """,
                (evaluation_run["id"],),
            ).fetchall()
            ips_action_rows = conn.execute(
                """
                SELECT record_json
                FROM ips_action_rows
                WHERE evaluation_run_id = ?
                ORDER BY id ASC
                """,
                (evaluation_run["id"],),
            ).fetchall()
            group_summary_rows = conn.execute(
                """
                SELECT record_json
                FROM group_summary_rows
                WHERE evaluation_run_id = ?
                ORDER BY id ASC
                """,
                (evaluation_run["id"],),
            ).fetchall()
            rc_violation_rows = conn.execute(
                """
                SELECT record_json
                FROM rc_violation_rows
                WHERE evaluation_run_id = ?
                ORDER BY id ASC
                """,
                (evaluation_run["id"],),
            ).fetchall()

    session_state = {
        "asset_df": [
            {
                "ticker": row["ticker"],
                "allocation": row["allocation"],
                "return_total": row["return_total"],
                "group": row["group_code"],
                "dca_enabled": bool(row["dca_enabled"]),
                "thesis_status": row["thesis_status_code"],
                "weight": row["weight"],
            }
            for row in positions
        ]
    }
    analysis_payload = None
    if analysis_run:
        session_state.update(
            {
                "metrics_df": [_json_load(row["record_json"]) for row in analysis_metrics],
                "portfolio_metrics": _json_load(
                    analysis_run["portfolio_metrics_json"], None
                ),
                "benchmark_metrics": _json_load(
                    analysis_run["benchmark_metrics_json"], None
                ),
                "missing_tickers": _json_load(
                    analysis_run["missing_tickers_json"], []
                ),
                "returns_smooth": _json_load(analysis_run["returns_smooth_json"], None),
                "analysis_settings": {
                    "period": analysis_run["period"],
                    "rf": analysis_run["rf"],
                    "bench": analysis_run["bench"],
                },
            }
        )
        analysis_payload = {
            "metrics_df": session_state["metrics_df"],
            "portfolio_metrics": session_state["portfolio_metrics"],
            "benchmark_metrics": session_state["benchmark_metrics"],
            "missing_tickers": session_state["missing_tickers"],
        }

    evaluation_payload = None
    if evaluation_run:
        session_state.update(
            {
                "proposal_df": [_json_load(row["record_json"]) for row in evaluation_rows],
                "ips_action_df": [
                    _json_load(row["record_json"]) for row in ips_action_rows
                ],
                "group_summary_df": [
                    _json_load(row["record_json"]) for row in group_summary_rows
                ],
                "rc_violations": [
                    _json_load(row["record_json"]) for row in rc_violation_rows
                ],
                "evaluation_settings": {
                    "rc_over_thresh_pct": evaluation_run["rc_over_thresh_pct"],
                    "e_thresh": evaluation_run["e_thresh"],
                    "target_weights": _json_load(
                        evaluation_run["target_weights_json"], None
                    ),
                },
                "ips_config_snapshot": _json_load(
                    evaluation_run["ips_config_snapshot_json"], None
                ),
            }
        )
        evaluation_payload = {
            "proposal_df": session_state["proposal_df"],
            "ips_action_df": session_state["ips_action_df"],
            "group_summary_df": session_state["group_summary_df"],
            "rc_violations": session_state["rc_violations"],
            "ips_config_snapshot": session_state["ips_config_snapshot"],
        }

    return {
        "summary": _snapshot_summary(summary_row),
        "session_state": session_state,
        "analysis": analysis_payload,
        "evaluation": evaluation_payload,
    }
