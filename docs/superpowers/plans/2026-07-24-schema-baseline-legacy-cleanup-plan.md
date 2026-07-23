# Schema Baseline and Legacy Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Replace the historical numbered schema migrations and retired manual instrument metadata with one current v10 schema baseline while preserving existing v10 user data.

**Architecture:** `storage/schema.py` will own one complete current-schema definition and accept only empty databases or v10 databases. Older and newer schema versions will fail before writes. The runtime, agent guidance, tests, and obsolete design artifacts will contain no active references to the retired profile model.

**Tech Stack:** Python 3, SQLite, pytest, Typer/JSON CLI contracts, `apply_patch`.

---

### Task 1: Freeze the current v10 schema as the supported baseline

**Files:**
- Modify: `storage/schema.py`
- Test: `tests/test_database_migrations.py`

- [ ] **Step 1: Capture the active v10 schema before editing**

Run:

```bash
sqlite3 -readonly data/portfolio_rebalancer.sqlite3 ".schema"
```

Expected: only the active Toss snapshot, performance, policy, market-candle, and evaluation tables listed in the design document; no instrument-profile table.

- [ ] **Step 2: Replace the historical migration map with one current schema definition**

In `storage/schema.py`, set `LATEST_SCHEMA_VERSION = 10`, remove `MIGRATION_1_SQL` through `MIGRATION_10_SQL` and `MIGRATIONS`, and add `CURRENT_SCHEMA_SQL` containing the exact active v10 tables and indexes from the captured schema. The SQL must include the current `ips_evaluation_runs` columns `market_evidence_fingerprint` and `market_evidence_json`, and must not create retired profile tables or columns.

- [ ] **Step 3: Implement fail-closed baseline initialization**

Replace `migrate(conn)` with this behavior. The `current == 0` branch must
first count non-`sqlite_%` objects and raise if a non-empty unversioned
database is encountered, so the baseline SQL cannot collide with an unknown
schema:

```python
current = schema_version(conn)
if current == 0:
    object_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type IN ('table', 'index', 'trigger', 'view')
          AND name NOT LIKE 'sqlite_%'
        """
    ).fetchone()[0]
    if object_count:
        raise SchemaVersionError(
            "Unsupported unversioned database; only an empty database "
            "can be initialized as schema 10."
        )
    conn.executescript(
        f"BEGIN IMMEDIATE;\n{CURRENT_SCHEMA_SQL}\n"
        f"PRAGMA user_version = {LATEST_SCHEMA_VERSION};\nCOMMIT;"
    )
elif current != LATEST_SCHEMA_VERSION:
    raise SchemaVersionError(
        f"Unsupported database schema {current}; "
        f"only schema {LATEST_SCHEMA_VERSION} is supported."
    )
return schema_version(conn)
```

Keep the existing rollback behavior for failed baseline creation. A nonzero unsupported database must not receive any SQL writes.

- [ ] **Step 4: Replace migration-history tests with baseline tests**

In `tests/test_database_migrations.py`, keep tests for fresh-schema table/column presence and integrity. Add tests that create a v10 database with a sentinel row and assert initialization preserves the row and `PRAGMA user_version`. Add v9, future-version, and non-empty-unversioned fixtures and assert `SchemaVersionError`, unchanged sentinel data, unchanged `user_version`, and no new tables. Remove tests whose only purpose is to exercise deleted historical migration numbers.

- [ ] **Step 5: Run the focused schema tests**

Run:

```bash
uv run pytest tests/test_database_migrations.py -q
```

Expected: all baseline, preservation, rollback, and unsupported-version tests pass.

### Task 2: Remove active legacy guidance and obsolete artifacts

**Files:**
- Modify: `.agents/skills/ips-judgment-filter/SKILL.md`
- Delete: `docs/superpowers/specs/2026-07-24-retire-instrument-metadata-design.md`
- Modify: `storage/schema.py` and `tests/test_database_migrations.py` if the
  final search finds historical identifiers outside the baseline SQL/tests

- [ ] **Step 1: Remove the retired thesis predicate from agent guidance**

Replace the sentence that requires `thesis_status=broken` for `Action` with guidance that the current runtime has no persisted thesis metadata and must not invent a thesis status. Keep the four status vocabulary and the prohibition on automatic execution.

- [ ] **Step 2: Remove obsolete documentation and compatibility references**

Delete the retirement design document after its requirements are represented by the baseline design and tests. Remove any remaining profile API, CLI, storage, fixture, or snapshot references discovered by:

```bash
rg --hidden -n --glob '!node_modules/**' --glob '!data/**' \
  'thesis_status|thesis_note|overlap_status|management_burden_status|holdability_status|etf_substitution_status|review_factors_note|instrument_profile|ips_instrument_profiles' .
```

The only allowed match after cleanup is the retained cleanup design record if it intentionally documents the removed model; no runtime, API, test, or active guidance match is allowed.

- [ ] **Step 3: Run contract and inspection tests**

Run:

```bash
uv run pytest tests/test_inspection_engine.py tests/test_inspection_service.py tests/test_cli.py tests/test_api_contract.py -q
```

Expected: JSON contracts remain machine-readable and inspection outputs contain no retired metadata fields.

### Task 3: Verify current user state and repository cleanliness

**Files:**
- No new production files
- Test-only edits from Tasks 1–2

- [ ] **Step 1: Validate the real v10 database read-only**

Run:

```bash
sqlite3 -readonly data/portfolio_rebalancer.sqlite3 \
  "SELECT user_version FROM pragma_user_version; SELECT name FROM sqlite_master WHERE type='table' AND name='ips_evaluation_runs';"
```

Expected: `10` and `ips_evaluation_runs`; no profile table.

- [ ] **Step 2: Run the complete test suite**

Run:

```bash
uv run pytest -q
```

Expected: pass with no live market-data dependency.

- [ ] **Step 3: Run the final hidden legacy search**

Run the search from Task 2 and confirm that only the retained decision record contains historical terms.

- [ ] **Step 4: Review the diff and commit the implementation**

Run:

```bash
git diff --check
git status --short
git add storage/schema.py tests/test_database_migrations.py .agents/skills/ips-judgment-filter/SKILL.md docs/superpowers/plans/2026-07-24-schema-baseline-legacy-cleanup-plan.md
git commit -m "refactor: baseline schema and remove legacy metadata"
```

Do not stage unrelated user changes already present in the worktree.
