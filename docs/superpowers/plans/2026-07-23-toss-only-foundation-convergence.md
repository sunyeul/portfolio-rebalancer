# Toss-only Foundation Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the generic portfolio foundation with a Toss-only schema,
instrument-profile model, deterministic account projection, and CLI while
preserving all existing Toss observation and performance evidence.

**Architecture:** Apply one destructive schema-v4 migration that drops only
named generic tables and adds Toss-keyed IPS state. Keep immutable Phase 1/2
tables unchanged, derive account views through a pure projection service, and
remove the generic API/frontend/analysis dependency graph rather than keeping a
compatibility layer.

**Tech Stack:** Python 3.12, SQLite, Typer, httpx, pytest, Ruff, uv, Taskfile

---

## Execution Constraints

- Work on the current `codex/phase-1-toss-account-observation` worktree, which
  already contains completed Phases 0–2 and the approved design commit.
- Read
  `docs/superpowers/specs/2026-07-23-toss-only-foundation-convergence-design.md`
  before editing.
- Do not call the live Toss API. All verification before the final local cutover
  uses persisted data or mocked responses.
- Use `apply_patch` for every tracked-file edit or deletion.
- Begin every shell command with `rtk`.
- Preserve one JSON object on stdout for every retained CLI command.
- Do not add direct buy/sell recommendations, order sizing, or broker mutation.
- Do not delete the three named local backup files until Task 6 proves the v4
  database and retained Toss rows are healthy.

## Target File Map

### Create

- `storage/policy_store.py` — canonical default policy, immutable policy lookup,
  and idempotent seed.
- `storage/instrument_profile_store.py` — validate and persist Toss-keyed IPS
  annotations.
- `services/account_projection.py` — pure complete-snapshot projection with
  gross and invested denominators.
- `tests/test_policy_store.py`
- `tests/test_instrument_profile_store.py`
- `tests/test_account_projection.py`

### Modify

- `storage/schema.py` — schema v4 destructive convergence migration.
- `storage/database.py` — remove durable migration backups, enable secure
  deletion for v4, seed policy, validate, and vacuum.
- `cli.py` — retain Toss/performance commands and add profile/account-view
  commands; remove generic commands.
- `tests/test_database_migrations.py`
- `tests/test_cli.py`
- `pyproject.toml`
- `uv.lock`
- `Taskfile.yml`
- `README.md`
- `AGENTS.md`

### Delete

- `api/`
- `frontend/`
- `middleware/`
- `core/`
- `main.py`
- `config/ips.yaml`
- `services/analysis_service.py`
- `services/evaluation_engine.py`
- `services/evaluation_period.py`
- `services/evaluation_status.py`
- `services/evaluation_units.py`
- `services/portfolio_service.py`
- `storage/config_store.py`
- `storage/journal_store.py`
- `storage/portfolio_store.py`
- every existing file under `utils/`
- generic-only tests listed in Task 5

## Task 0: Record the local cutover preconditions

**Files:** none

- [ ] **Step 1: Confirm the worktree starts clean**

Run:

```bash
rtk git status --short
```

Expected: no output.

- [ ] **Step 2: Verify the retained local Toss evidence before code changes**

Run:

```bash
rtk uv run ips-pilot toss-snapshots --snapshot-id 4
rtk uv run ips-pilot performance history --run-id 1
```

Expected: two JSON objects with `ok: true`; the first reports
`snapshot_id: 4`, and the second reports `run_id: 1` and
`baseline_snapshot_id: 4`.

- [ ] **Step 3: Resolve the exact legacy backup targets**

Run:

```bash
rtk rg --files -g '*.bak' data
```

Expected exact targets:

```text
data/portfolio_rebalancer.sqlite3.pre-v0-to-v1-20260722T151550329704Z.bak
data/portfolio_rebalancer.sqlite3.pre-v1-to-v2-20260722T155401188399Z.bak
data/portfolio_rebalancer.sqlite3.pre-v2-to-v3-20260722T164302140860Z.bak
```

If any additional file appears, stop and inspect it instead of widening the
delete target.

## Task 1: Add the schema-v4 Toss-only migration

**Files:**

- Modify: `storage/schema.py`
- Modify: `storage/database.py`
- Create: `storage/policy_store.py`
- Modify: `tests/test_database_migrations.py`
- Create: `tests/test_policy_store.py`

- [ ] **Step 1: Replace legacy-preservation migration expectations with a
  failing v3-to-v4 contract test**

Build the v3 fixture directly from `MIGRATION_1_SQL`, `MIGRATION_2_SQL`, and
`MIGRATION_3_SQL`, then insert one generic portfolio row and linked Toss rows.
The core assertion must be:

```python
GENERIC_TABLES = {
    "portfolios",
    "assets",
    "thesis_statuses",
    "portfolio_snapshots",
    "portfolio_current_states",
    "snapshot_positions",
    "snapshot_evaluation_runs",
    "ips_target_allocations",
    "ips_action_priorities",
    "ips_rules",
    "journal_entries",
    "analysis_runs",
}


def test_v3_to_v4_drops_generic_tables_and_preserves_toss_evidence(
    monkeypatch, tmp_path
):
    path = tmp_path / "v3.sqlite3"
    _create_v3_fixture(path)
    expected = _insert_v3_generic_and_toss_rows(path)
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert _schema_version(path) == 4
    with sqlite3.connect(path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert GENERIC_TABLES.isdisjoint(names)
        assert conn.execute(
            "SELECT source_fingerprint FROM broker_account_snapshots WHERE id = ?",
            (expected["snapshot_id"],),
        ).fetchone()[0] == expected["snapshot_fingerprint"]
        assert conn.execute(
            "SELECT confirmation_fingerprint FROM account_tracking_baselines "
            "WHERE id = ?",
            (expected["baseline_id"],),
        ).fetchone()[0] == expected["baseline_fingerprint"]
        assert conn.execute(
            "SELECT input_fingerprint FROM account_performance_runs WHERE id = ?",
            (expected["run_id"],),
        ).fetchone()[0] == expected["run_fingerprint"]
```

The fixture helper must insert every non-null column with fixed values, create
the account snapshot before the baseline, and create the baseline before the
performance run. Do not call current application stores to build the old
schema.

- [ ] **Step 2: Add failing rollback, no-backup, and fresh-schema tests**

Add these explicit contracts:

```python
def test_failed_v4_migration_rolls_back_all_drops(monkeypatch, tmp_path):
    path = tmp_path / "rollback.sqlite3"
    _create_v3_fixture(path)
    _set_database_path(monkeypatch, path)
    broken = schema_module.MIGRATION_4_SQL + "\nINSERT INTO missing_table VALUES (1);"
    monkeypatch.setitem(schema_module.MIGRATIONS, 4, broken)

    with connect() as conn, pytest.raises(sqlite3.OperationalError):
        schema_module.migrate(conn)

    assert _schema_version(path) == 3
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='portfolios'"
        ).fetchone() is not None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='ips_instrument_profiles'"
        ).fetchone() is None


def test_v4_migration_creates_no_adjacent_backup(monkeypatch, tmp_path):
    path = tmp_path / "portfolio.sqlite3"
    _create_v3_fixture(path)
    _set_database_path(monkeypatch, path)

    initialize_database()

    assert list(tmp_path.glob("*.bak")) == []


def test_fresh_database_contains_only_toss_and_ips_tables(monkeypatch, tmp_path):
    path = tmp_path / "fresh.sqlite3"
    _set_database_path(monkeypatch, path)

    initialize_database()

    with sqlite3.connect(path) as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert GENERIC_TABLES.isdisjoint(names)
        assert {"ips_instrument_profiles", "ips_policy_versions"} <= names
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
```

- [ ] **Step 3: Run the migration tests and confirm they fail**

Run:

```bash
rtk uv run pytest tests/test_database_migrations.py -q
```

Expected: failures because the latest schema is still v3, generic tables remain,
and the new IPS tables do not exist.

- [ ] **Step 4: Implement `MIGRATION_4_SQL`**

Set `LATEST_SCHEMA_VERSION = 4`, add migration 4 to `MIGRATIONS`, and use this
schema shape:

```python
MIGRATION_4_SQL = """
PRAGMA secure_delete = ON;

DROP TABLE IF EXISTS journal_entries;
DROP TABLE IF EXISTS snapshot_evaluation_runs;
DROP TABLE IF EXISTS snapshot_positions;
DROP TABLE IF EXISTS portfolio_current_states;
DROP TABLE IF EXISTS portfolio_snapshots;
DROP TABLE IF EXISTS assets;
DROP TABLE IF EXISTS portfolios;
DROP TABLE IF EXISTS ips_action_priorities;
DROP TABLE IF EXISTS ips_rules;
DROP TABLE IF EXISTS ips_target_allocations;
DROP TABLE IF EXISTS thesis_statuses;
DROP TABLE IF EXISTS analysis_runs;

CREATE TABLE ips_instrument_profiles (
    account_alias TEXT NOT NULL,
    market_country TEXT NOT NULL,
    symbol TEXT NOT NULL,
    layer TEXT NOT NULL CHECK(layer IN ('core', 'satellite', 'experiment')),
    thesis_status TEXT NOT NULL
        CHECK(thesis_status IN ('unknown', 'valid', 'watch', 'broken')),
    thesis_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(account_alias, market_country, symbol)
);

CREATE TABLE ips_policy_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_alias TEXT NOT NULL,
    version INTEGER NOT NULL,
    policy_json TEXT NOT NULL,
    policy_hash TEXT NOT NULL UNIQUE,
    superseded_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_alias, version)
);

CREATE UNIQUE INDEX idx_ips_policy_one_active
    ON ips_policy_versions(account_alias)
    WHERE superseded_at IS NULL;
"""
```

The old tables must be dropped child-first exactly as shown. Do not add a
generic JSON state table.

- [ ] **Step 5: Add canonical policy seeding**

Create `storage/policy_store.py` with this public contract:

```python
DEFAULT_POLICY = {
    "cash_reserve": {"minimum": 0.10, "target": 0.15, "maximum": 0.20},
    "layers": {
        "core": {"minimum": 0.70, "target": 0.80, "maximum": 0.90},
        "satellite": {"minimum": 0.10, "target": 0.20, "maximum": 0.30},
        "experiment": {"minimum": 0.00, "target": 0.00, "maximum": 0.05},
    },
}


def canonical_policy_json(policy: dict[str, object]) -> str:
    return json.dumps(
        policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def policy_hash(policy: dict[str, object]) -> str:
    return sha256(canonical_policy_json(policy).encode("utf-8")).hexdigest()


def ensure_default_policy(conn: sqlite3.Connection) -> None:
    encoded = canonical_policy_json(DEFAULT_POLICY)
    fingerprint = policy_hash(DEFAULT_POLICY)
    conn.execute(
        """
        INSERT INTO ips_policy_versions (
            account_alias, version, policy_json, policy_hash
        )
        SELECT 'toss-brokerage', 1, ?, ?
        WHERE NOT EXISTS (
            SELECT 1 FROM ips_policy_versions
            WHERE account_alias = 'toss-brokerage'
        )
        """,
        (encoded, fingerprint),
    )
```

Also implement `get_active_policy(account_alias="toss-brokerage")` to return the
decoded policy plus `id`, `version`, `policy_hash`, and timestamps, or `None`.
Use this test contract to prove repeated initialization retains one row and
never changes its hash:

```python
def test_default_policy_is_seeded_once_and_is_replayable(monkeypatch, tmp_path):
    path = tmp_path / "policy.sqlite3"
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(path))

    initialize_database()
    first = get_active_policy()
    initialize_database()
    second = get_active_policy()

    assert first is not None
    assert second == first
    assert first["version"] == 1
    assert first["policy"] == DEFAULT_POLICY
    assert first["policy_hash"] == policy_hash(DEFAULT_POLICY)
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM ips_policy_versions"
        ).fetchone()[0] == 1
```

- [ ] **Step 6: Replace durable backups with validated secure cutover**

In `storage/database.py`:

- remove `_create_migration_backup`, the timestamp import, and backup naming;
- retain a read-only source-version probe;
- enable `PRAGMA secure_delete = ON` before migrating an existing schema below
  v4;
- call `ensure_default_policy(conn)` after `migrate(conn)`;
- require `PRAGMA integrity_check == 'ok'` and an empty
  `PRAGMA foreign_key_check` result;
- after committing a real pre-v4 database, open a new connection, run `VACUUM`,
  and repeat both checks.

Use one explicit error type:

```python
class DatabaseIntegrityError(RuntimeError):
    """Raised when a migrated database fails SQLite integrity checks."""


def _assert_integrity(conn: sqlite3.Connection) -> None:
    result = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    foreign_keys = conn.execute("PRAGMA foreign_key_check").fetchall()
    if result != "ok" or foreign_keys:
        raise DatabaseIntegrityError(
            f"database integrity failed: integrity={result}, "
            f"foreign_key_errors={len(foreign_keys)}"
        )
```

- [ ] **Step 7: Run focused schema and policy tests**

Run:

```bash
rtk uv run pytest tests/test_database_migrations.py tests/test_policy_store.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit the schema convergence**

```bash
rtk git add storage/schema.py storage/database.py storage/policy_store.py tests/test_database_migrations.py tests/test_policy_store.py
rtk git commit -m "feat: converge persistence on Toss schema"
```

## Task 2: Add Toss-keyed instrument profiles

**Files:**

- Create: `storage/instrument_profile_store.py`
- Create: `tests/test_instrument_profile_store.py`

- [ ] **Step 1: Write failing profile-store tests**

Use a complete broker snapshot fixture containing `005930/KR` and `AAPL/US`.
Cover the following exact behaviors:

```python
def test_upsert_profile_requires_a_toss_observed_identity(snapshot_fixture):
    profile = upsert_profile(
        symbol="AAPL",
        market_country="US",
        layer="core",
        thesis_status="valid",
        thesis_note="Long-term core",
    )
    assert profile["symbol"] == "AAPL"
    assert profile["market_country"] == "US"

    with pytest.raises(InstrumentProfileError, match="not observed"):
        upsert_profile(
            symbol="UNSEEN",
            market_country="US",
            layer="core",
            thesis_status="valid",
        )


@pytest.mark.parametrize("layer", ["cash", "other", ""])
def test_upsert_profile_rejects_invalid_layer(snapshot_fixture, layer):
    with pytest.raises(InstrumentProfileError, match="invalid layer"):
        upsert_profile("AAPL", "US", layer, "valid")


@pytest.mark.parametrize("status", ["intact", "sold", ""])
def test_upsert_profile_rejects_invalid_thesis(snapshot_fixture, status):
    with pytest.raises(InstrumentProfileError, match="invalid thesis_status"):
        upsert_profile("AAPL", "US", "core", status)
```

Also assert that updating a profile changes only the profile row and leaves the
matching `broker_holdings` row byte-for-byte unchanged.

- [ ] **Step 2: Run tests and confirm the store is missing**

```bash
rtk uv run pytest tests/test_instrument_profile_store.py -q
```

Expected: collection fails because `storage.instrument_profile_store` does not
exist.

- [ ] **Step 3: Implement the profile store**

Use the exact accepted sets and normalize only surrounding whitespace and case:

```python
LAYERS = frozenset({"core", "satellite", "experiment"})
THESIS_STATUSES = frozenset({"unknown", "valid", "watch", "broken"})


class InstrumentProfileError(ValueError):
    """Raised when a Toss instrument cannot receive the requested IPS profile."""


def instrument_key(symbol: str, market_country: str) -> tuple[str, str]:
    normalized_symbol = str(symbol).strip().upper()
    normalized_country = str(market_country).strip().upper()
    if not normalized_symbol or not normalized_country:
        raise InstrumentProfileError("symbol and market_country are required")
    return normalized_symbol, normalized_country
```

Implement:

- `upsert_profile(symbol, market_country, layer, thesis_status,
  thesis_note="", account_alias="toss-brokerage")`;
- `get_profile(symbol, market_country, account_alias="toss-brokerage")`;
- `list_profiles(account_alias="toss-brokerage")`;
- `profile_map(account_alias="toss-brokerage")`, keyed by
  `(market_country, symbol)`.

Before an upsert, verify existence with a join from `broker_holdings` to
`broker_account_snapshots` on `snapshot_id` and the exact account alias,
country, and symbol. Use `INSERT ... ON CONFLICT ... DO UPDATE` only on the IPS
fields and `updated_at`; never write a broker table.

- [ ] **Step 4: Run the focused tests**

```bash
rtk uv run pytest tests/test_instrument_profile_store.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit instrument profiles**

```bash
rtk git add storage/instrument_profile_store.py tests/test_instrument_profile_store.py
rtk git commit -m "feat: persist Toss instrument profiles"
```

## Task 3: Build the deterministic Toss account projection

**Files:**

- Create: `services/account_projection.py`
- Create: `tests/test_account_projection.py`

- [ ] **Step 1: Write failing projection tests**

Create complete, partial, stale, and failed local snapshots without live calls.
The complete fixture must contain two holdings, cash, and one unclassified
instrument. Assert exact denominators:

```python
def test_projection_uses_explicit_gross_and_invested_denominators(
    complete_snapshot, profiled_aapl
):
    result = build_account_projection(snapshot_id=complete_snapshot["id"])

    assert result["snapshot_id"] == complete_snapshot["id"]
    assert result["cash_weight_gross"] == pytest.approx(0.15)
    aapl = next(item for item in result["positions"] if item["symbol"] == "AAPL")
    assert aapl["gross_weight"] == pytest.approx(
        aapl["market_value_krw"] / result["total_value_krw"]
    )
    assert aapl["invested_weight"] == pytest.approx(
        aapl["market_value_krw"] / result["invested_value_krw"]
    )
    assert result["unclassified"] == [
        {"market_country": "KR", "symbol": "005930"}
    ]
```

Also cover:

- latest selection reports the actual selected snapshot ID;
- partial, stale, and failed IDs raise `AccountProjectionError`;
- missing KRW values raise instead of becoming zero;
- `total != invested + cash` outside a 1 KRW absolute tolerance raises;
- holding sum mismatch outside 1 KRW raises;
- an all-cash account keeps cash weight evaluable but reports
  `invested_weights_evaluable: false` and emits no invented invested weights;
- layer weights aggregate only classified holdings and report classification
  coverage explicitly.

- [ ] **Step 2: Run tests and confirm the service is missing**

```bash
rtk uv run pytest tests/test_account_projection.py -q
```

Expected: collection fails because `services.account_projection` does not
exist.

- [ ] **Step 3: Implement the pure projection**

Expose one public function:

```python
def build_account_projection(
    snapshot_id: int | None = None,
    account_alias: str = "toss-brokerage",
) -> dict[str, object]:
    snapshot = (
        get_snapshot(snapshot_id)
        if snapshot_id is not None
        else latest_complete(account_alias)
    )
    if snapshot is None:
        raise AccountProjectionError("complete Toss snapshot not found")
    if snapshot["account_alias"] != account_alias:
        raise AccountProjectionError("snapshot account alias mismatch")
    if snapshot["state"] != "complete":
        raise AccountProjectionError(
            f"snapshot {snapshot['id']} is not complete: {snapshot['state']}"
        )
    return _project_complete_snapshot(snapshot, profile_map(account_alias))
```

Use `math.isfinite` on every numeric input. Require `total_value_krw > 0` and
nonnegative invested and cash values. Treat `invested_value_krw == 0` as a valid
all-cash account only when there are no positive holding values. Sort positions
by `(market_country, symbol)` and unclassified identities by the same key so the
JSON is deterministic.

Return these top-level keys and no evaluation status:

```python
{
    "snapshot_id": int,
    "account_alias": str,
    "synced_at": str,
    "source_timestamps": dict,
    "total_value_krw": float,
    "invested_value_krw": float,
    "cash_value_krw": float,
    "cash_weight_gross": float,
    "invested_weights_evaluable": bool,
    "classification_coverage_invested": float | None,
    "positions": list[dict],
    "layer_weights_invested": dict[str, float],
    "unclassified": list[dict[str, str]],
    "data_quality": dict,
    "reconciliation": dict,
}
```

- [ ] **Step 4: Run projection and retained store tests**

```bash
rtk uv run pytest tests/test_account_projection.py tests/test_account_observation_store.py tests/test_performance_store.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the projection**

```bash
rtk git add services/account_projection.py tests/test_account_projection.py
rtk git commit -m "feat: project Toss account state for IPS"
```

## Task 4: Cut the CLI over to Toss-only commands

**Files:**

- Modify: `cli.py`
- Modify: `tests/test_cli.py`
- Verify: `tests/test_toss_cli.py`
- Verify: `tests/test_performance_cli.py`

- [ ] **Step 1: Replace generic CLI tests with failing Toss-only surface tests**

Keep one help contract and add profile/account-view command tests:

```python
def test_help_exposes_only_toss_product_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "toss-health",
        "toss-sync",
        "toss-snapshots",
        "performance",
        "profiles",
        "account-view",
    ):
        assert command in result.stdout
    for removed in (
        "evaluate",
        "agent-brief",
        "review-queue",
        "risk",
        "portfolios",
        "snapshots",
    ):
        assert removed not in result.stdout


def test_account_view_emits_one_projection_json(monkeypatch):
    monkeypatch.setattr(
        cli,
        "build_account_projection",
        lambda snapshot_id=None: {"snapshot_id": snapshot_id or 7},
    )
    result = runner.invoke(app, ["account-view", "--snapshot-id", "7"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "ok": True,
        "command": "account-view",
        "projection": {"snapshot_id": 7},
        "error": None,
    }
```

Add `profiles set` tests for success and a sanitized
`InstrumentProfileError`, and `profiles list` tests that emit profiles plus the
projection's unclassified identities.

- [ ] **Step 2: Run CLI tests and confirm the new contract fails**

```bash
rtk uv run pytest tests/test_cli.py -q
```

Expected: failures because generic commands still exist and the new commands do
not.

- [ ] **Step 3: Remove generic CLI code and pandas serialization**

Delete from `cli.py`:

- pandas and generic analysis/evaluation imports;
- `portfolios_app` and `snapshots_app`;
- `_empty_v2_payload`, `_exit_with_error`, `_selected_sources`,
  `_parse_layer_benchmarks`, `_load_asset_df`, `_save_run`, `_run_v2`,
  `_session_data`, and `_status_summary`;
- `evaluate`, `agent-brief`, `review-queue`, and `risk` commands;
- all generic portfolio/snapshot subcommands.

Replace API/pandas serialization with:

```python
def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
```

Retain `_exit_with_command_error`, `_build_toss_service`, Toss commands, and
every performance command unchanged except for import cleanup.

- [ ] **Step 4: Add profiles and account-view commands**

Register `profiles_app = typer.Typer(help="Manage Toss instrument IPS profiles.")`
and implement:

```python
@app.command("account-view")
def account_view(
    snapshot_id: Annotated[int | None, typer.Option("--snapshot-id")] = None,
) -> None:
    try:
        initialize_database()
        projection = build_account_projection(snapshot_id=snapshot_id)
        _emit_json(
            {
                "ok": True,
                "command": "account-view",
                "projection": projection,
                "error": None,
            }
        )
    except Exception as exc:
        _exit_with_command_error("account-view", exc)
```

`profiles set` must call `upsert_profile` with the five explicit options and
emit the returned row. `profiles list` must initialize the database, build the
requested/latest projection, list profiles, and emit both `profiles` and
`unclassified`. Do not add quantity, price, or value mutation options.

- [ ] **Step 5: Run all retained CLI tests**

```bash
rtk uv run pytest tests/test_cli.py tests/test_toss_cli.py tests/test_performance_cli.py -q
```

Expected: all tests pass and each success/failure capture parses as one JSON
object.

- [ ] **Step 6: Commit the Toss-only CLI**

```bash
rtk git add cli.py tests/test_cli.py
rtk git commit -m "feat: expose Toss-only inspection CLI"
```

## Task 5: Delete the generic runtime and dependency graph

**Files:** all exact paths in the Target File Map

- [ ] **Step 1: Delete generic runtime files with `apply_patch`**

Delete all tracked files under `api/`, `frontend/`, `middleware/`, `core/`, and
`utils/`, then delete:

```text
main.py
config/ips.yaml
services/analysis_service.py
services/evaluation_engine.py
services/evaluation_period.py
services/evaluation_status.py
services/evaluation_units.py
services/portfolio_service.py
storage/config_store.py
storage/journal_store.py
storage/portfolio_store.py
```

Do not delete `services/account_performance.py`, the new
`services/account_projection.py`, any `integrations/toss/` file, or any retained
Toss/performance store.

- [ ] **Step 2: Delete generic-only tests with `apply_patch`**

Delete exactly:

```text
tests/test_analysis_service.py
tests/test_api_v1.py
tests/test_config_management.py
tests/test_data_fetcher.py
tests/test_efficiency_metrics.py
tests/test_evaluation_period.py
tests/test_evaluation_status.py
tests/test_evaluation_units.py
tests/test_journal_store.py
tests/test_performance_metrics.py
tests/test_portfolio_input.py
tests/test_portfolio_persistence.py
tests/test_risk_metrics.py
```

Retain and adapt `tests/test_cli.py`; do not delete Toss, schema, account
observation, account performance, or performance CLI/store tests.

- [ ] **Step 3: Reduce the Python package and dependencies**

Make `pyproject.toml` runtime dependencies exactly:

```toml
dependencies = [
    "httpx>=0.28.0",
    "typer>=0.12.0",
]

[tool.setuptools]
py-modules = ["cli"]

[tool.setuptools.packages.find]
include = ["integrations*", "services*", "storage*"]
```

Keep the project script, build system, Python requirement, pytest, and Ruff.
Regenerate the lockfile:

```bash
rtk uv lock
```

Expected: a successful lock resolution without yfinance, pandas, NumPy, SciPy,
FastAPI, Pydantic, PyYAML, python-multipart, itsdangerous, or uvicorn.

- [ ] **Step 4: Reduce Taskfile commands**

Delete `run`, `frontend-dev`, `dev`, and `build-frontend`. Retain formatting,
linting, Toss health/sync, and performance tasks. Add:

```yaml
  toss-account-view:
    desc: Show the latest complete Toss account projection
    cmds:
      - uv run ips-pilot account-view

  toss-profiles:
    desc: Show Toss instrument IPS profiles and missing classifications
    cmds:
      - uv run ips-pilot profiles list
```

- [ ] **Step 5: Prove the removed dependency graph is gone**

Run:

```bash
rtk rg -n -i "yfinance|pandas|numpy|scipy|fastapi|pydantic|manual|csv|tsv|japan|일본" --glob '*.py' --glob 'pyproject.toml'
rtk rg --files -g 'api/**' -g 'frontend/**' -g 'middleware/**' -g 'core/**' -g 'utils/**' -g 'config/**' .
```

Expected: both commands report no matching runtime code/files. References in
approved design/history documents are allowed and are intentionally outside the
scan.

- [ ] **Step 6: Run the full remaining test suite**

```bash
rtk uv run pytest -q
```

Expected: all remaining tests pass without live credentials or network access.

- [ ] **Step 7: Commit the legacy removal**

```bash
rtk git add -A api frontend middleware core utils config main.py services/analysis_service.py services/evaluation_engine.py services/evaluation_period.py services/evaluation_status.py services/evaluation_units.py services/portfolio_service.py storage/config_store.py storage/journal_store.py storage/portfolio_store.py tests/test_analysis_service.py tests/test_api_v1.py tests/test_config_management.py tests/test_data_fetcher.py tests/test_efficiency_metrics.py tests/test_evaluation_period.py tests/test_evaluation_status.py tests/test_evaluation_units.py tests/test_journal_store.py tests/test_performance_metrics.py tests/test_portfolio_input.py tests/test_portfolio_persistence.py tests/test_risk_metrics.py pyproject.toml uv.lock Taskfile.yml
rtk git commit -m "refactor: remove generic portfolio runtime"
```

Before committing, inspect `rtk git status --short` and confirm that no Toss or
performance implementation file is accidentally deleted.

## Task 6: Perform and verify the local v4 cutover

**Files:**

- Migrate: `data/portfolio_rebalancer.sqlite3` (ignored local state)
- Delete after verification: the three exact `.bak` files from Task 0

- [ ] **Step 1: Trigger the local migration without network access**

Run:

```bash
rtk uv run ips-pilot account-view --snapshot-id 4
```

Expected: one JSON object with `ok: true`, `snapshot_id: 4` inside
`projection`, correct account totals, and either explicit profiles or an
`unclassified` list. Initialization upgrades the local database to v4,
secure-deletes generic rows, validates it, and vacuums it.

- [ ] **Step 2: Verify retained local Phase 1/2 evidence**

Run:

```bash
rtk uv run ips-pilot toss-snapshots --snapshot-id 4
rtk uv run ips-pilot performance history --run-id 1
```

Expected: both still return `ok: true`; snapshot 4, baseline 1, and run 1 match
the Task 0 IDs and fingerprints/totals.

- [ ] **Step 3: Verify the physical SQLite cutover**

Run:

```bash
rtk sqlite3 data/portfolio_rebalancer.sqlite3 "PRAGMA user_version; PRAGMA integrity_check; PRAGMA foreign_key_check; PRAGMA freelist_count;"
```

Expected output contains schema version `4`, integrity result `ok`, no
foreign-key rows, and freelist count `0` after `VACUUM`.

Then list tables:

```bash
rtk sqlite3 data/portfolio_rebalancer.sqlite3 ".tables"
```

Expected: only broker account, performance, profile, policy, and SQLite internal
tables; none of the `GENERIC_TABLES` names.

- [ ] **Step 4: Re-resolve and delete only the approved backup files**

Run the read-only check again:

```bash
rtk rg --files -g '*.bak' data
```

If and only if it returns the same three Task 0 paths, remove exactly those
files:

```bash
rtk rm -- data/portfolio_rebalancer.sqlite3.pre-v0-to-v1-20260722T151550329704Z.bak data/portfolio_rebalancer.sqlite3.pre-v1-to-v2-20260722T155401188399Z.bak data/portfolio_rebalancer.sqlite3.pre-v2-to-v3-20260722T164302140860Z.bak
```

Run:

```bash
rtk rg --files -g '*.bak' data
```

Expected: no output. Report explicitly that these ignored local backups were
permanently removed and are not recoverable through the application.

- [ ] **Step 5: Confirm no tracked file changed during local migration**

```bash
rtk git status --short
```

Expected: no output.

## Task 7: Document the Toss-only runtime and run the completion gate

**Files:**

- Modify: `README.md`
- Modify: `AGENTS.md`
- Verify: all retained source and tests

- [ ] **Step 1: Rewrite README around the Toss-only product**

Keep these sections and remove all generic examples:

1. Product and inspection guardrails.
2. Required environment variables:
   `TOSS_OPEN_API_CLIENT_ID`, `TOSS_OPEN_API_CLIENT_SECRET`, and
   `TOSS_OPEN_API_ACCOUNT_SEQ`.
3. Read-only trust boundary.
4. `task toss-health`, `task toss-sync`, `task toss-account-view`, profile
   commands, and performance commands.
5. Immutable snapshot states and latest-complete behavior.
6. Tracking baseline and cash-flow classification limits.
7. Current Phase 2.5 boundary and Phases 3–6 roadmap link.

Do not document removed API routes, frontend commands, CSV columns, manual rows,
yfinance benchmarks, or generic portfolios.

- [ ] **Step 2: Add one durable Toss-only rule to AGENTS.md**

Under Product Guardrails, add:

```markdown
- Treat normalized Toss account snapshots as the only source of holdings, cash,
  cost, price, order, and execution facts; never reintroduce manual portfolio,
  yfinance, generic broker, or Japan-account fallback paths.
```

Do not add a directory map or command inventory.

- [ ] **Step 3: Run formatting and lint checks**

```bash
rtk uv run ruff format --check .
rtk uv run ruff check .
```

Expected: both exit successfully with no changes required.

- [ ] **Step 4: Run the full test suite**

```bash
rtk uv run pytest -q
```

Expected: all retained schema, profile, projection, CLI, Toss integration, and
performance tests pass.

- [ ] **Step 5: Run the adversarial repository scans**

```bash
rtk rg -n -i "import yfinance|from yfinance|portfolio/manual|portfolio/csv|UploadFile|parse_csv|parse_text|Japan account|일본 계좌" --glob '*.py' --glob '*.ts' --glob '*.tsx' --glob 'pyproject.toml' .
rtk rg -n "POST /api/v1/orders|conditional-orders|/modify|/cancel" integrations/toss
rtk git diff --check
```

Expected:

- the first scan has no matches;
- the second scan finds no newly allowlisted mutation path outside tests that
  explicitly prove rejection;
- the diff check is clean.

- [ ] **Step 6: Verify machine-readable CLI failures**

With Toss credentials absent in an isolated test environment, verify
`toss-health` emits one sanitized JSON error. Verify `account-view` with a
nonexistent snapshot and `profiles set` with an unseen symbol also emit one JSON
error and never a traceback.

Run the focused automated contract:

```bash
rtk uv run pytest tests/test_cli.py tests/test_toss_cli.py tests/test_performance_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit documentation and completion evidence**

```bash
rtk git add README.md AGENTS.md
rtk git commit -m "docs: document Toss-only inspection runtime"
```

- [ ] **Step 8: Final worktree and history check**

```bash
rtk git status --short
rtk git log --oneline -8
```

Expected: a clean worktree and focused commits for schema, profiles, projection,
CLI, legacy removal, and documentation.

## Completion Gate

Phase 2.5 is complete only when all of the following are true:

- schema v4 preserves Toss snapshot 4, baseline 1, and performance run 1;
- generic tables and the three named legacy backup files are absent;
- no manual input, yfinance, Japan-account, generic API, or frontend runtime
  remains;
- the lockfile contains only the reduced runtime dependency graph;
- profiles can annotate only previously observed Toss instruments;
- `account-view` uses the latest complete or explicit complete snapshot and
  exposes gross/invested denominators without judgment language;
- partial, stale, failed, inconsistent, and unclassified states fail closed;
- every retained CLI command emits one JSON object;
- Toss order mutation remains blocked;
- the full remaining test and lint suite is green;
- the worktree is clean.
