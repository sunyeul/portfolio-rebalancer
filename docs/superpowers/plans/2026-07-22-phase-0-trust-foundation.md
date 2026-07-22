# Phase 0 Trust Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lossless SQLite migration/backup infrastructure and a credential-safe Toss Securities transport boundary that permits observations plus OAuth token issuance while making broker order mutation impossible from IPS Pilot.

**Architecture:** Keep schema migration and backup concerns inside `storage`, and add a new `integrations.toss` package with separate configuration, redaction, transport, and token-provider units. The transport uses an explicit method-and-path allowlist: approved GET observations and `POST /oauth2/token` only. No account synchronization, persistence, valuation, evaluation, API route, frontend surface, or live credential use belongs to Phase 0.

**Tech Stack:** Python 3.12, SQLite, FastAPI project conventions, httpx, Pydantic-free dataclasses, pytest, ruff, uv, RTK-prefixed shell commands.

---

## Execution Preconditions

- Execute this plan in a dedicated worktree on branch `codex/phase-0-trust-foundation`.
- Start from commit `3ec88cd` or a descendant containing the approved roadmap design.
- Do not load real Toss credentials into the test process.
- Do not call the live Toss API; use `httpx.MockTransport` fixtures only.
- Do not edit portfolio snapshots, IPS configuration values, journal entries, or the user's production SQLite file.

## Delivery Boundary

Phase 0 produces infrastructure that is independently testable but intentionally has no user-facing broker sync feature. Phase 1 will add account/holdings/purchasing-power reads and immutable account snapshots after Phase 0 passes its adversarial gate.

## File Responsibility Map

### Create

- `storage/schema.py` — ordered schema migrations, schema-version checks, and the baseline schema SQL without destructive drops.
- `tests/test_database_migrations.py` — fresh-schema, legacy-adoption, future-version, backup, and idempotency tests.
- `integrations/__init__.py` — integration package marker.
- `integrations/toss/__init__.py` — Toss integration package marker and safe public exports.
- `integrations/toss/config.py` — lazy environment-backed credentials with secret-safe representations.
- `integrations/toss/redaction.py` — header, account identifier, and known-secret redaction helpers.
- `integrations/toss/transport.py` — method/path allowlist and sanitized httpx error boundary.
- `integrations/toss/auth.py` — in-memory OAuth token provider and authorized read session.
- `tests/test_toss_config.py` — configuration validation and redaction tests.
- `tests/test_toss_transport.py` — allowlist, mutation-blocking, sanitized-error, token-cache, and authorization-header tests.

### Modify

- `storage/database.py` — call ordered migrations, create pre-migration SQLite backups, and retain seed behavior.
- `tests/test_portfolio_persistence.py` — replace the destructive legacy-table expectation with preservation behavior.
- `pyproject.toml` — make httpx a runtime dependency and package `integrations*`.
- `uv.lock` — regenerate after dependency-group movement.
- `.gitignore` — ignore pre-migration SQLite backup files.
- `.env.example` — document Toss credential variable names without values.
- `README.md` — document Phase 0's read-only boundary and the absence of live sync.

## Task 1: Replace Destructive Initialization With Versioned Schema Migration

**Files:**
- Create: `storage/schema.py`
- Create: `tests/test_database_migrations.py`
- Modify: `storage/database.py:1-178`
- Modify: `tests/test_portfolio_persistence.py:154-171`

- [ ] **Step 1: Write failing schema migration tests**

Create `tests/test_database_migrations.py` with these tests:

```python
import sqlite3

import pytest

from storage.database import connect, initialize_database
from storage.schema import LATEST_SCHEMA_VERSION, SchemaVersionError


def _set_database_path(monkeypatch, path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))


def _schema_version(path):
    with sqlite3.connect(path) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def test_fresh_database_uses_latest_schema_without_legacy_tables(monkeypatch, tmp_path):
    path = tmp_path / "fresh.sqlite3"
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION
    with sqlite3.connect(path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "portfolios" in names
    assert "snapshot_evaluation_runs" in names
    assert "analysis_runs" not in names


def test_existing_version_zero_database_is_adopted_without_data_loss(
    monkeypatch, tmp_path
):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE analysis_runs (id INTEGER PRIMARY KEY, payload TEXT);
            INSERT INTO portfolios (name) VALUES ('Existing account');
            INSERT INTO analysis_runs (id, payload) VALUES (7, 'preserve me');
            """
        )
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION
    with connect() as conn:
        assert conn.execute("SELECT name FROM portfolios WHERE id = 1").fetchone()[0] == "Existing account"
        assert conn.execute("SELECT payload FROM analysis_runs WHERE id = 7").fetchone()[0] == "preserve me"


def test_migration_preserves_snapshots_evaluations_ips_config_and_journal(
    monkeypatch, tmp_path
):
    path = tmp_path / "real-prior-schema.sqlite3"
    _set_database_path(monkeypatch, path)
    initialize_database()

    with connect() as conn:
        portfolio_id = int(
            conn.execute(
                "INSERT INTO portfolios (name) VALUES ('Existing portfolio')"
            ).lastrowid
        )
        asset_id = int(
            conn.execute(
                "INSERT INTO assets (ticker, display_name) VALUES ('VOO', 'S&P 500')"
            ).lastrowid
        )
        thesis_status_id = int(
            conn.execute(
                "SELECT id FROM thesis_statuses WHERE code = 'valid'"
            ).fetchone()["id"]
        )
        snapshot_id = int(
            conn.execute(
                """
                INSERT INTO portfolio_snapshots (portfolio_id, name, note)
                VALUES (?, 'Existing snapshot', 'keep snapshot')
                """,
                (portfolio_id,),
            ).lastrowid
        )
        conn.execute(
            """
            INSERT INTO portfolio_current_states (portfolio_id, state_json)
            VALUES (?, '{"source":"existing"}')
            """,
            (portfolio_id,),
        )
        conn.execute(
            """
            INSERT INTO snapshot_positions (
                snapshot_id, asset_id, allocation, weight, return_total,
                layer, thesis_status_id, position_order
            )
            VALUES (?, ?, 100.0, 1.0, 0.12, 'core', ?, 0)
            """,
            (snapshot_id, asset_id, thesis_status_id),
        )
        conn.execute(
            """
            INSERT INTO snapshot_evaluation_runs (
                snapshot_id, settings_json, result_json, schema_version,
                engine_version, ips_config_hash, status
            )
            VALUES (?, '{}', '{"status":"OK"}', 2, 'existing-engine',
                    'existing-hash', 'active')
            """,
            (snapshot_id,),
        )
        conn.execute(
            "INSERT INTO ips_rules (key, value_json) VALUES ('existing-rule', 'true')"
        )
        conn.execute(
            """
            INSERT INTO ips_action_priorities (action_code, label, priority)
            VALUES ('existing-action', 'Existing', 77)
            """
        )
        conn.execute(
            "UPDATE ips_target_allocations SET target = 0.77 WHERE layer = 'core'"
        )
        conn.execute(
            """
            INSERT INTO journal_entries (
                snapshot_id, date, decision_context, review_items_json,
                decision_note
            )
            VALUES (?, '2026-07-22', 'existing context', '[]', 'keep journal')
            """,
            (snapshot_id,),
        )
        conn.execute("PRAGMA user_version = 0")

    initialize_database()

    assert _schema_version(path) == LATEST_SCHEMA_VERSION
    with connect() as conn:
        assert conn.execute(
            "SELECT note FROM portfolio_snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()["note"] == "keep snapshot"
        assert conn.execute(
            "SELECT state_json FROM portfolio_current_states WHERE portfolio_id = ?",
            (portfolio_id,),
        ).fetchone()["state_json"] == '{"source":"existing"}'
        assert conn.execute(
            "SELECT return_total FROM snapshot_positions WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()["return_total"] == pytest.approx(0.12)
        assert conn.execute(
            "SELECT engine_version FROM snapshot_evaluation_runs WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()["engine_version"] == "existing-engine"
        assert conn.execute(
            "SELECT value_json FROM ips_rules WHERE key = 'existing-rule'"
        ).fetchone()["value_json"] == "true"
        assert conn.execute(
            "SELECT priority FROM ips_action_priorities WHERE action_code = 'existing-action'"
        ).fetchone()["priority"] == 77
        assert conn.execute(
            "SELECT target FROM ips_target_allocations WHERE layer = 'core'"
        ).fetchone()["target"] == pytest.approx(0.77)
        assert conn.execute(
            "SELECT decision_note FROM journal_entries WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()["decision_note"] == "keep journal"


def test_database_from_newer_schema_is_rejected_without_mutation(
    monkeypatch, tmp_path
):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel (value) VALUES ('unchanged')")
        conn.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION + 1}")
    _set_database_path(monkeypatch, path)

    with pytest.raises(SchemaVersionError, match="newer than supported"):
        initialize_database()

    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM sentinel").fetchone()[0] == "unchanged"
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == LATEST_SCHEMA_VERSION + 1
```

In `tests/test_portfolio_persistence.py`, replace `test_initialize_database_drops_legacy_run_tables` with:

```python
def test_initialize_database_preserves_unversioned_legacy_run_tables(portfolio_db):
    with connect() as conn:
        conn.execute("CREATE TABLE analysis_runs (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE analysis_metrics (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE evaluation_runs (id INTEGER PRIMARY KEY)")

    initialize_database()

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('analysis_runs', 'analysis_metrics', 'evaluation_runs')
            ORDER BY name
            """
        ).fetchall()

    assert [row["name"] for row in rows] == [
        "analysis_metrics",
        "analysis_runs",
        "evaluation_runs",
    ]
```

- [ ] **Step 2: Run the tests and verify the current destructive behavior fails**

Run:

```bash
rtk uv run pytest tests/test_database_migrations.py tests/test_portfolio_persistence.py::test_initialize_database_preserves_unversioned_legacy_run_tables -v
```

Expected: FAIL because `storage.schema` does not exist and the current initializer drops the three legacy tables.

- [ ] **Step 3: Add the ordered migration module**

Create `storage/schema.py`:

```python
"""Forward-only SQLite schema migrations."""

from __future__ import annotations

import sqlite3


LATEST_SCHEMA_VERSION = 1


class SchemaVersionError(RuntimeError):
    """Raised when the database schema cannot be migrated safely."""


MIGRATION_1_SQL = """
CREATE TABLE IF NOT EXISTS portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    display_name TEXT,
    asset_type TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS thesis_statuses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 999,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id),
    name TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_current_states (
    portfolio_id INTEGER PRIMARY KEY REFERENCES portfolios(id) ON DELETE CASCADE,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS snapshot_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
    asset_id INTEGER NOT NULL REFERENCES assets(id),
    allocation REAL NOT NULL,
    weight REAL NOT NULL,
    return_total REAL,
    layer TEXT NOT NULL DEFAULT 'core',
    thesis_status_id INTEGER NOT NULL REFERENCES thesis_statuses(id),
    position_order INTEGER NOT NULL DEFAULT 0,
    UNIQUE(snapshot_id, asset_id)
);

CREATE TABLE IF NOT EXISTS snapshot_evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
    settings_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    engine_version TEXT NOT NULL,
    ips_config_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    superseded_by_run_id INTEGER REFERENCES snapshot_evaluation_runs(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshot_evaluation_runs_snapshot_status
    ON snapshot_evaluation_runs(snapshot_id, status, id);

CREATE TABLE IF NOT EXISTS ips_target_allocations (
    layer TEXT PRIMARY KEY,
    min REAL NOT NULL,
    target REAL NOT NULL,
    max REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ips_action_priorities (
    action_code TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    priority INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS ips_rules (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS journal_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL UNIQUE REFERENCES portfolio_snapshots(id) ON DELETE CASCADE,
    date TEXT NOT NULL,
    decision_context TEXT NOT NULL,
    playbook_code TEXT,
    review_items_json TEXT NOT NULL DEFAULT '[]',
    decision_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


MIGRATIONS = {1: MIGRATION_1_SQL}


def schema_version(conn: sqlite3.Connection) -> int:
    """Return SQLite's application schema version."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply every pending migration in order without destructive cleanup."""
    current = schema_version(conn)
    if current > LATEST_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"Database schema {current} is newer than supported {LATEST_SCHEMA_VERSION}."
        )

    for target in range(current + 1, LATEST_SCHEMA_VERSION + 1):
        script = MIGRATIONS[target]
        try:
            conn.executescript(
                f"BEGIN IMMEDIATE;\n{script}\nPRAGMA user_version = {target};\nCOMMIT;"
            )
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
    return schema_version(conn)
```

- [ ] **Step 4: Route database initialization through migrations**

In `storage/database.py`, import the migration function:

```python
from storage.schema import migrate
```

Replace the body of `initialize_database` with:

```python
def initialize_database() -> None:
    """Migrate the persistence schema and seed stable lookup values."""
    with connect() as conn:
        migrate(conn)
        _seed_lookup(conn, "thesis_statuses", THESIS_STATUS_SEEDS)
        _seed_target_allocations(conn)
```

Delete the embedded schema SQL and all three `DROP TABLE` statements from `storage/database.py`. Do not change `connect`, `_seed_lookup`, or `_seed_target_allocations` in this task.

- [ ] **Step 5: Run focused migration and persistence tests**

Run:

```bash
rtk uv run pytest tests/test_database_migrations.py tests/test_portfolio_persistence.py -v
```

Expected: PASS. The legacy tables and rows remain present; portfolios, current state, snapshots, positions, evaluation runs, IPS configuration, and journal data survive adoption; fresh databases use schema version 1; and newer schemas are rejected.

- [ ] **Step 6: Commit the migration boundary**

```bash
rtk git add storage/schema.py storage/database.py tests/test_database_migrations.py tests/test_portfolio_persistence.py
rtk git commit -m "refactor: add forward-only database migrations"
```

## Task 2: Back Up Existing Databases Before Migration

**Files:**
- Modify: `storage/database.py`
- Modify: `tests/test_database_migrations.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add failing backup and idempotency tests**

Append to `tests/test_database_migrations.py`:

```python
def test_existing_database_is_backed_up_once_before_first_migration(
    monkeypatch, tmp_path
):
    path = tmp_path / "portfolio.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sentinel (value TEXT NOT NULL)")
        conn.execute("INSERT INTO sentinel (value) VALUES ('original')")
    _set_database_path(monkeypatch, path)

    initialize_database()

    backups = list(tmp_path.glob("portfolio.sqlite3.pre-v0-to-v1-*.bak"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as conn:
        assert conn.execute("SELECT value FROM sentinel").fetchone()[0] == "original"
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 0

    initialize_database()

    assert list(tmp_path.glob("portfolio.sqlite3.pre-v0-to-v1-*.bak")) == backups


def test_fresh_database_does_not_create_a_backup(monkeypatch, tmp_path):
    path = tmp_path / "fresh.sqlite3"
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert list(tmp_path.glob("*.bak")) == []
```

- [ ] **Step 2: Run the backup tests and verify they fail**

Run:

```bash
rtk uv run pytest tests/test_database_migrations.py -k "backup or backed_up" -v
```

Expected: FAIL because initialization does not create a pre-migration backup.

- [ ] **Step 3: Implement consistent SQLite backup creation**

Update the imports in `storage/database.py`:

```python
from datetime import datetime, timezone

from storage.schema import LATEST_SCHEMA_VERSION, migrate, schema_version
```

Add these helpers before `initialize_database`:

```python
def _migration_source_version(path: Path) -> int | None:
    """Return a migratable source version only when a real schema exists."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with sqlite3.connect(path) as conn:
        version = schema_version(conn)
        object_count = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM sqlite_master
                WHERE type IN ('table', 'index', 'trigger', 'view')
                  AND name NOT LIKE 'sqlite_%'
                """
            ).fetchone()[0]
        )
    if object_count == 0 or version >= LATEST_SCHEMA_VERSION:
        return None
    return version


def _create_migration_backup(path: Path, source_version: int) -> Path:
    """Create a SQLite-consistent backup before a forward migration."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = path.with_name(
        f"{path.name}.pre-v{source_version}-to-v{LATEST_SCHEMA_VERSION}-{stamp}.bak"
    )
    with sqlite3.connect(path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    return backup_path
```

Replace `initialize_database` with:

```python
def initialize_database() -> None:
    """Back up, migrate, and seed the local persistence database."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    source_version = _migration_source_version(path)
    if source_version is not None:
        _create_migration_backup(path, source_version)

    with connect() as conn:
        migrate(conn)
        _seed_lookup(conn, "thesis_statuses", THESIS_STATUS_SEEDS)
        _seed_target_allocations(conn)
```

- [ ] **Step 4: Ignore migration backups in the project data directory**

Append to the local SQLite section of `.gitignore`:

```gitignore
data/*.sqlite3.pre-v*-to-v*-*.bak
```

- [ ] **Step 5: Run migration tests twice to verify idempotency**

Run twice:

```bash
rtk uv run pytest tests/test_database_migrations.py -v
```

Expected on both runs: PASS. Each test database creates at most one backup before its first migration.

- [ ] **Step 6: Commit backup safety**

```bash
rtk git add storage/database.py tests/test_database_migrations.py .gitignore
rtk git commit -m "feat: back up sqlite before schema migration"
```

## Task 3: Add Secret-Safe Toss Configuration and Redaction

**Files:**
- Create: `integrations/__init__.py`
- Create: `integrations/toss/__init__.py`
- Create: `integrations/toss/config.py`
- Create: `integrations/toss/redaction.py`
- Create: `tests/test_toss_config.py`

- [ ] **Step 1: Write failing configuration and redaction tests**

Create `tests/test_toss_config.py`:

```python
import pytest

from integrations.toss.config import TossApiConfig, TossConfigError
from integrations.toss.redaction import REDACTED, redact_headers, redact_known_values


def test_toss_config_loads_required_environment_without_secret_repr(monkeypatch):
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_ID", "client-visible-only-to-server")
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_SECRET", "super-secret")
    monkeypatch.setenv("TOSS_OPEN_API_ACCOUNT_SEQ", "42")

    config = TossApiConfig.from_env()

    assert config.client_id == "client-visible-only-to-server"
    assert config.client_secret == "super-secret"
    assert config.account_seq == 42
    assert config.base_url == "https://openapi.tossinvest.com"
    assert "client-visible-only-to-server" not in repr(config)
    assert "super-secret" not in repr(config)
    assert "42" not in repr(config)


@pytest.mark.parametrize(
    "missing_name",
    [
        "TOSS_OPEN_API_CLIENT_ID",
        "TOSS_OPEN_API_CLIENT_SECRET",
        "TOSS_OPEN_API_ACCOUNT_SEQ",
    ],
)
def test_toss_config_reports_missing_variable_name_only(monkeypatch, missing_name):
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_ID", "client-id")
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("TOSS_OPEN_API_ACCOUNT_SEQ", "7")
    monkeypatch.delenv(missing_name)

    with pytest.raises(TossConfigError) as exc_info:
        TossApiConfig.from_env()

    assert missing_name in str(exc_info.value)
    assert "client-id" not in str(exc_info.value)
    assert "client-secret" not in str(exc_info.value)


def test_toss_config_rejects_non_positive_account_sequence(monkeypatch):
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_ID", "client-id")
    monkeypatch.setenv("TOSS_OPEN_API_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("TOSS_OPEN_API_ACCOUNT_SEQ", "0")

    with pytest.raises(TossConfigError, match="positive integer"):
        TossApiConfig.from_env()


def test_sensitive_headers_and_known_values_are_redacted():
    headers = redact_headers(
        {
            "Authorization": "Bearer access-token",
            "X-Tossinvest-Account": "42",
            "Accept": "application/json",
        }
    )

    assert headers == {
        "Authorization": REDACTED,
        "X-Tossinvest-Account": REDACTED,
        "Accept": "application/json",
    }
    assert redact_known_values(
        "client-id client-secret access-token account-1234",
        ["client-id", "client-secret", "access-token", "account-1234"],
    ) == f"{REDACTED} {REDACTED} {REDACTED} {REDACTED}"
```

- [ ] **Step 2: Run the tests and verify imports fail**

Run:

```bash
rtk uv run pytest tests/test_toss_config.py -v
```

Expected: FAIL because the `integrations.toss` package does not exist.

- [ ] **Step 3: Create the package markers**

Create `integrations/__init__.py`:

```python
"""External read-only integration boundaries."""
```

Create `integrations/toss/__init__.py`:

```python
"""Read-only Toss Securities integration boundary."""

from integrations.toss.config import TossApiConfig, TossConfigError

__all__ = ["TossApiConfig", "TossConfigError"]
```

- [ ] **Step 4: Implement lazy environment configuration**

Create `integrations/toss/config.py`:

```python
"""Environment-backed Toss Open API configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_TOSS_BASE_URL = "https://openapi.tossinvest.com"


class TossConfigError(RuntimeError):
    """Raised when required Toss configuration is absent or invalid."""


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise TossConfigError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class TossApiConfig:
    """Server-only Toss credentials and endpoint configuration."""

    client_id: str = field(repr=False)
    client_secret: str = field(repr=False)
    account_seq: int = field(repr=False)
    base_url: str = DEFAULT_TOSS_BASE_URL

    @classmethod
    def from_env(cls) -> "TossApiConfig":
        account_seq_text = _required_env("TOSS_OPEN_API_ACCOUNT_SEQ")
        try:
            account_seq = int(account_seq_text)
        except ValueError as exc:
            raise TossConfigError(
                "TOSS_OPEN_API_ACCOUNT_SEQ must be a positive integer."
            ) from exc
        if account_seq <= 0:
            raise TossConfigError(
                "TOSS_OPEN_API_ACCOUNT_SEQ must be a positive integer."
            )
        return cls(
            client_id=_required_env("TOSS_OPEN_API_CLIENT_ID"),
            client_secret=_required_env("TOSS_OPEN_API_CLIENT_SECRET"),
            account_seq=account_seq,
        )
```

- [ ] **Step 5: Implement deterministic redaction helpers**

Create `integrations/toss/redaction.py`:

```python
"""Redaction helpers for Toss credentials and account identifiers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


REDACTED = "<redacted>"
SENSITIVE_HEADERS = frozenset({"authorization", "x-tossinvest-account"})


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy safe for diagnostics."""
    return {
        name: REDACTED if name.lower() in SENSITIVE_HEADERS else value
        for name, value in headers.items()
    }


def redact_known_values(text: str, values: Iterable[str]) -> str:
    """Replace exact known secret/account values in diagnostic text."""
    redacted = text
    for value in values:
        if value:
            redacted = redacted.replace(value, REDACTED)
    return redacted
```

- [ ] **Step 6: Run configuration tests**

Run:

```bash
rtk uv run pytest tests/test_toss_config.py -v
```

Expected: PASS. No credential or account sequence appears in `repr(config)` or validation errors.

- [ ] **Step 7: Commit configuration safety**

```bash
rtk git add integrations/__init__.py integrations/toss/__init__.py integrations/toss/config.py integrations/toss/redaction.py tests/test_toss_config.py
rtk git commit -m "feat: add secret-safe toss configuration"
```

## Task 4: Enforce the Broker Method-and-Path Allowlist

**Files:**
- Create: `integrations/toss/transport.py`
- Create: `tests/test_toss_transport.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Write failing transport allowlist tests**

Create `tests/test_toss_transport.py` with the initial transport tests:

```python
import httpx
import pytest

from integrations.toss.config import TossApiConfig
from integrations.toss.transport import (
    TossRequestBlocked,
    TossTransport,
    TossTransportError,
)


@pytest.fixture()
def config():
    return TossApiConfig(
        client_id="client-id",
        client_secret="client-secret",
        account_seq=42,
    )


def _transport(config, handler):
    client = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )
    return TossTransport(config=config, client=client)


def test_transport_permits_oauth_token_post(config):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})

    transport = _transport(config, handler)

    payload = transport.request_json(
        "POST",
        "/oauth2/token",
        data={"grant_type": "client_credentials"},
    )

    assert payload["access_token"] == "token"
    assert seen == [("POST", "/oauth2/token")]


def test_transport_permits_allowlisted_observation_get(config):
    def handler(request):
        return httpx.Response(200, json={"result": []})

    transport = _transport(config, handler)

    assert transport.request_json("GET", "/api/v1/holdings") == {"result": []}


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/v1/orders"),
        ("POST", "/api/v1/orders/order-1/cancel"),
        ("POST", "/api/v1/orders/order-1/modify"),
        ("DELETE", "/api/v1/orders/order-1"),
        ("PATCH", "/api/v1/orders/order-1"),
        ("GET", "/api/v1/unknown"),
    ],
)
def test_transport_blocks_every_non_allowlisted_request_before_network(
    config, method, path
):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    transport = _transport(config, handler)

    with pytest.raises(TossRequestBlocked, match="blocked by read-only policy"):
        transport.request_json(method, path)

    assert calls == 0


def test_transport_error_does_not_echo_response_body_or_credentials(config):
    def handler(request):
        return httpx.Response(
            401,
            json={
                "echo_client_secret": "client-secret",
                "echo_authorization": "Bearer access-token",
            },
        )

    transport = _transport(config, handler)

    with pytest.raises(TossTransportError) as exc_info:
        transport.request_json("GET", "/api/v1/accounts")

    message = str(exc_info.value)
    assert "status=401" in message
    assert "client-secret" not in message
    assert "access-token" not in message
```

- [ ] **Step 2: Make httpx a runtime dependency and package integrations**

In `pyproject.toml`, add httpx to `[project].dependencies`:

```toml
    "httpx>=0.28.0",
```

Remove the same entry from `[dependency-groups].dev` so it is declared exactly once. Change the setuptools package include list to:

```toml
include = ["api*", "core*", "integrations*", "middleware*", "services*", "storage*", "utils*"]
```

Regenerate the lockfile:

```bash
rtk uv lock
```

Expected: `uv.lock` remains internally consistent and httpx is available to the runtime package.

- [ ] **Step 3: Run the transport tests and verify implementation is missing**

Run:

```bash
rtk uv run pytest tests/test_toss_transport.py -v
```

Expected: FAIL because `integrations.toss.transport` does not exist.

- [ ] **Step 4: Implement the allowlisted transport**

Create `integrations/toss/transport.py`:

```python
"""HTTP transport that makes Toss order mutation unreachable."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from integrations.toss.config import TossApiConfig


TOKEN_PATH = "/oauth2/token"
ALLOWED_GET_PATHS = frozenset(
    {
        "/api/v1/accounts",
        "/api/v1/holdings",
        "/api/v1/buying-power",
        "/api/v1/exchange-rate",
        "/api/v1/orders",
    }
)


class TossRequestBlocked(RuntimeError):
    """Raised before a non-read-only request can reach the network."""


class TossTransportError(RuntimeError):
    """Sanitized Toss transport failure."""


def _normalized_path(path: str) -> str:
    if not path.startswith("/") or "?" in path or "#" in path:
        raise TossRequestBlocked("Toss request blocked by read-only policy.")
    return path.rstrip("/") or "/"


def _assert_allowed(method: str, path: str) -> tuple[str, str]:
    normalized_method = method.upper()
    normalized_path = _normalized_path(path)
    if normalized_method == "GET" and normalized_path in ALLOWED_GET_PATHS:
        return normalized_method, normalized_path
    if normalized_method == "POST" and normalized_path == TOKEN_PATH:
        return normalized_method, normalized_path
    raise TossRequestBlocked("Toss request blocked by read-only policy.")


class TossTransport:
    """Send only explicitly allowlisted Toss observation/auth requests."""

    def __init__(self, config: TossApiConfig, client: httpx.Client | None = None):
        self._client = client or httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(10.0),
        )
        self._owns_client = client is None

    def request_json(
        self,
        method: str,
        path: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str | int] | None = None,
        data: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        allowed_method, allowed_path = _assert_allowed(method, path)
        try:
            response = self._client.request(
                allowed_method,
                allowed_path,
                headers=dict(headers or {}),
                params=dict(params or {}),
                data=dict(data or {}),
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise TossTransportError(
                f"Toss API request failed: status={exc.response.status_code} path={allowed_path}"
            ) from None
        except (httpx.HTTPError, ValueError, TypeError):
            raise TossTransportError(
                f"Toss API request failed: status=unavailable path={allowed_path}"
            ) from None
        if not isinstance(payload, dict):
            raise TossTransportError(
                f"Toss API request failed: status=invalid-json path={allowed_path}"
            )
        return payload

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
```

- [ ] **Step 5: Run the allowlist tests**

Run:

```bash
rtk uv run pytest tests/test_toss_transport.py -v
```

Expected: PASS. The mock handler sees the OAuth POST and observation GET, but sees zero blocked order-mutation calls.

- [ ] **Step 6: Commit the transport boundary**

```bash
rtk git add integrations/toss/transport.py tests/test_toss_transport.py pyproject.toml uv.lock
rtk git commit -m "feat: enforce read-only toss transport"
```

## Task 5: Add In-Memory OAuth Tokens and Authorized Reads

**Files:**
- Create: `integrations/toss/auth.py`
- Modify: `integrations/toss/__init__.py`
- Modify: `tests/test_toss_transport.py`

- [ ] **Step 1: Add failing token-cache and authorized-read tests**

Append to `tests/test_toss_transport.py`:

```python
from integrations.toss.auth import TossAuthorizedReader, TossTokenProvider


def test_token_provider_caches_token_in_memory_until_refresh_window(config):
    token_calls = 0
    now = [100.0]

    def handler(request):
        nonlocal token_calls
        token_calls += 1
        form = dict(httpx.QueryParams(request.content.decode()))
        assert form == {
            "grant_type": "client_credentials",
            "client_id": "client-id",
            "client_secret": "client-secret",
        }
        return httpx.Response(
            200,
            json={
                "access_token": f"token-{token_calls}",
                "token_type": "Bearer",
                "expires_in": 3600,
            },
        )

    transport = _transport(config, handler)
    provider = TossTokenProvider(config, transport, clock=lambda: now[0])

    assert provider.access_token() == "token-1"
    assert provider.access_token() == "token-1"
    assert token_calls == 1

    now[0] = 3701.0
    assert provider.access_token() == "token-2"
    assert token_calls == 2
    assert "token-2" not in repr(provider)


def test_authorized_reader_sends_required_headers_without_exposing_them(config):
    seen_headers = {}

    def handler(request):
        if request.url.path == "/oauth2/token":
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )
        seen_headers.update(request.headers)
        return httpx.Response(200, json={"result": []})

    transport = _transport(config, handler)
    provider = TossTokenProvider(config, transport, clock=lambda: 100.0)
    reader = TossAuthorizedReader(config, transport, provider)

    assert reader.get_json("/api/v1/holdings") == {"result": []}
    assert seen_headers["authorization"] == "Bearer access-token"
    assert seen_headers["x-tossinvest-account"] == "42"
    assert "access-token" not in repr(reader)
    assert "42" not in repr(reader)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"access_token": "", "expires_in": 3600},
        {"access_token": "token", "expires_in": 0},
        {"access_token": "token", "expires_in": "invalid"},
    ],
)
def test_token_provider_rejects_malformed_token_response(config, payload):
    def handler(request):
        return httpx.Response(200, json=payload)

    provider = TossTokenProvider(config, _transport(config, handler))

    with pytest.raises(TossTransportError, match="invalid token response"):
        provider.access_token()
```

- [ ] **Step 2: Run the new tests and verify auth imports fail**

Run:

```bash
rtk uv run pytest tests/test_toss_transport.py -k "token or authorized" -v
```

Expected: FAIL because `integrations.toss.auth` does not exist.

- [ ] **Step 3: Implement the in-memory token provider and reader**

Create `integrations/toss/auth.py`:

```python
"""In-memory OAuth token handling for read-only Toss observations."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from integrations.toss.config import TossApiConfig
from integrations.toss.transport import TossTransport, TossTransportError


TOKEN_REFRESH_SKEW_SECONDS = 60.0


@dataclass(frozen=True)
class _CachedToken:
    value: str = field(repr=False)
    refresh_at: float


class TossTokenProvider:
    """Issue and cache an OAuth token in process memory only."""

    def __init__(
        self,
        config: TossApiConfig,
        transport: TossTransport,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._config = config
        self._transport = transport
        self._clock = clock
        self._cached: _CachedToken | None = None

    def access_token(self) -> str:
        now = self._clock()
        if self._cached is not None and now < self._cached.refresh_at:
            return self._cached.value

        payload = self._transport.request_json(
            "POST",
            "/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._config.client_id,
                "client_secret": self._config.client_secret,
            },
        )
        token = payload.get("access_token")
        expires_in = payload.get("expires_in")
        try:
            lifetime = float(expires_in)
        except (TypeError, ValueError):
            lifetime = 0.0
        if not isinstance(token, str) or not token or lifetime <= 0:
            raise TossTransportError("Toss API request failed: invalid token response")

        refresh_at = now + max(1.0, lifetime - TOKEN_REFRESH_SKEW_SECONDS)
        self._cached = _CachedToken(value=token, refresh_at=refresh_at)
        return token

    def invalidate(self) -> None:
        self._cached = None


class TossAuthorizedReader:
    """Add OAuth and account headers to allowlisted Toss GET observations."""

    def __init__(
        self,
        config: TossApiConfig,
        transport: TossTransport,
        tokens: TossTokenProvider,
    ):
        self._config = config
        self._transport = transport
        self._tokens = tokens

    def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> dict[str, Any]:
        return self._transport.request_json(
            "GET",
            path,
            headers={
                "Authorization": f"Bearer {self._tokens.access_token()}",
                "X-Tossinvest-Account": str(self._config.account_seq),
            },
            params=params,
        )
```

- [ ] **Step 4: Export only the safe Phase 0 public surface**

Replace `integrations/toss/__init__.py` with:

```python
"""Read-only Toss Securities integration boundary."""

from integrations.toss.auth import TossAuthorizedReader, TossTokenProvider
from integrations.toss.config import TossApiConfig, TossConfigError
from integrations.toss.transport import (
    TossRequestBlocked,
    TossTransport,
    TossTransportError,
)

__all__ = [
    "TossApiConfig",
    "TossConfigError",
    "TossAuthorizedReader",
    "TossTokenProvider",
    "TossRequestBlocked",
    "TossTransport",
    "TossTransportError",
]
```

- [ ] **Step 5: Run all Toss boundary tests**

Run:

```bash
rtk uv run pytest tests/test_toss_config.py tests/test_toss_transport.py -v
```

Expected: PASS. Tokens are reused only inside the valid window, malformed responses fail safely, and account reads receive the required headers.

- [ ] **Step 6: Commit OAuth memory handling**

```bash
rtk git add integrations/toss/__init__.py integrations/toss/auth.py tests/test_toss_transport.py
rtk git commit -m "feat: add in-memory toss oauth tokens"
```

## Task 6: Document and Package the Phase 0 Boundary

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_toss_config.py`

- [ ] **Step 1: Add a failing packaging declaration test**

Append to `tests/test_toss_config.py`:

```python
import tomllib
from pathlib import Path


def test_toss_runtime_dependencies_and_package_are_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert "httpx>=0.28.0" in pyproject["project"]["dependencies"]
    assert "httpx>=0.28.0" not in pyproject["dependency-groups"]["dev"]
    assert "integrations*" in pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
```

- [ ] **Step 2: Run the declaration test**

Run:

```bash
rtk uv run pytest tests/test_toss_config.py::test_toss_runtime_dependencies_and_package_are_declared -v
```

Expected: PASS if Task 4 completed correctly. Treat failure as a Task 4 packaging regression and fix `pyproject.toml` before continuing.

- [ ] **Step 3: Document environment variable names without values**

Append to `.env.example`:

```dotenv

# Toss Securities Open API (server-only; never commit actual values)
# TOSS_OPEN_API_CLIENT_ID=
# TOSS_OPEN_API_CLIENT_SECRET=
# TOSS_OPEN_API_ACCOUNT_SEQ=
```

- [ ] **Step 4: Add the Phase 0 security boundary to README**

Add this subsection after the development environment settings in `README.md`:

```markdown
### Toss Securities integration boundary

The Toss Securities integration is server-only and read-only. Phase 0 permits the OAuth token request and allowlisted account observations, but it exposes no sync API or user-facing account data yet. Order creation, modification, cancellation, sizing, and execution are outside IPS Pilot's product boundary.

Configure credentials only through the local environment variables listed in `.env.example`. Do not place real credentials in source files, SQLite, logs, browser storage, screenshots, or test fixtures.
```

- [ ] **Step 5: Verify the built wheel contains the integration package**

Run:

```bash
rtk uv build
```

Expected: PASS and create a wheel under `dist/`.

Inspect the wheel:

```bash
rtk unzip -l dist/ips_pilot-0.1.0-py3-none-any.whl
```

Expected: output includes `integrations/toss/config.py`, `integrations/toss/transport.py`, and `integrations/toss/auth.py`.

- [ ] **Step 6: Run documentation-adjacent tests**

Run:

```bash
rtk uv run pytest tests/test_toss_config.py tests/test_toss_transport.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit packaging and documentation**

```bash
rtk git add .env.example README.md tests/test_toss_config.py
rtk git commit -m "docs: define toss integration security boundary"
```

`pyproject.toml` and `uv.lock` should already be clean from Task 4. If the declaration test required a correction, include those two files in this commit.

## Task 7: Run the Phase 0 Adversarial Completion Gate

**Files:**
- Verify only; modify a file only when a failing check identifies a concrete Phase 0 defect.

- [ ] **Step 1: Run focused trust-foundation tests**

```bash
rtk uv run pytest tests/test_database_migrations.py tests/test_portfolio_persistence.py tests/test_toss_config.py tests/test_toss_transport.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the complete backend regression suite**

```bash
rtk uv run pytest
```

Expected: PASS with no live market or broker calls.

- [ ] **Step 3: Run formatting and lint checks without mutation**

```bash
rtk uv run ruff format --check storage integrations tests
rtk uv run ruff check storage integrations tests
```

Expected: PASS. If formatting fails, run `rtk uv run ruff format` only on the listed Phase 0 files, then rerun both checks.

- [ ] **Step 4: Prove no broker mutation route is allowlisted**

```bash
rtk rg -n "api/v1/(orders|conditional-orders).*(cancel|modify)|place_order|create_order|cancel_order|modify_order" integrations tests/test_toss_transport.py
```

Expected: matches occur only in negative tests that assert requests are blocked. `ALLOWED_GET_PATHS` contains `/api/v1/orders` only for observation; no order mutation path or method is present.

- [ ] **Step 5: Prove secrets are not persisted or exposed**

```bash
rtk rg -n "TOSS_OPEN_API_CLIENT_SECRET|access_token|X-Tossinvest-Account" api core middleware services storage frontend integrations
```

Expected: credential names and token handling occur only in `integrations/toss/config.py`, `integrations/toss/auth.py`, `integrations/toss/redaction.py`, and `integrations/toss/transport.py`. There are no API response models, SQLite columns, session keys, or frontend fields for these values.

- [ ] **Step 6: Inspect migration behavior and staged scope**

```bash
rtk git status --short
rtk git diff --check
```

Expected: no unrelated user files, snapshots, SQLite databases, IPS config, or journal data are modified. `git diff --check` reports no whitespace errors.

- [ ] **Step 7: Commit only concrete gate fixes, if any**

If Steps 1–6 required a correction:

```bash
rtk git add .env.example .gitignore README.md pyproject.toml uv.lock storage/database.py storage/schema.py integrations/__init__.py integrations/toss/__init__.py integrations/toss/auth.py integrations/toss/config.py integrations/toss/redaction.py integrations/toss/transport.py tests/test_database_migrations.py tests/test_portfolio_persistence.py tests/test_toss_auth.py tests/test_toss_config.py tests/test_toss_transport.py
rtk git commit -m "fix: close phase zero trust gate gaps"
```

The `git add` command intentionally enumerates the complete Phase 0 responsibility map; already-clean paths are harmless. Before committing, inspect `rtk git diff --cached --name-only` and unstage any path that was not changed for the gate. If no correction was required, do not create an empty commit.

## Phase 0 Completion Report

Before requesting Phase 1 design approval, report:

- the final schema version and backup filename pattern;
- the migration fixture results and confirmation that legacy data survives;
- the exact Toss method/path allowlist;
- confirmation that only `POST /oauth2/token` is permitted outside GET;
- confirmation that order mutation tests observe zero outbound requests;
- confirmation that secrets and raw account identifiers do not enter persistence or frontend state;
- focused and full test results;
- any remaining Pydantic or unrelated warnings separately from Phase 0 status.

Stop after the completion report. Phase 1 account synchronization requires its own approved design and implementation plan.
