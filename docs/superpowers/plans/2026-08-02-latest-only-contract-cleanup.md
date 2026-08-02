# Latest-only contract cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove legacy and compatibility paths and retain current-contract policy, evaluation, and candle records without losing required price history.

**Architecture:** The policy and persisted-evaluation validators own strict current contracts. API and CLI expose persisted values without hydration. An explicit Typer maintenance command cleans the local database transactionally, keeping the active policy and one latest revision for each candle date.

**Tech Stack:** Python 3.12, SQLite, Typer, FastAPI, pytest, Ruff.

---

### Task 1: Enforce a strict current policy

**Files:**

- Modify: `services/policy_validation.py:54-307,321-532`
- Modify: `storage/policy_store.py:54-105,241-278`
- Modify: `tests/test_policy_store.py:34-149`

- [ ] **Step 1: Write failing policy tests**

```python
def test_policy_validation_rejects_unknown_top_level_fields():
    policy = _current_policy()
    policy["allocation_review"] = {"strategy": "retired"}
    with pytest.raises(PolicyValidationError, match="unknown policy fields"):
        validate_policy(policy, [("US", "SPY")])
```

- [ ] **Step 2: Verify that the test fails**

Run: `uv run pytest tests/test_policy_store.py -q`

Expected: FAIL; unknown fields are silently discarded.

- [ ] **Step 3: Implement the minimal strict contract**

```python
POLICY_FIELDS = frozenset({
    "cash_reserve", "performance", "risk_review", "cadence", "layers", "instruments",
})
unknown = sorted(set(policy) - POLICY_FIELDS)
if unknown:
    errors.append(f"unknown policy fields: {', '.join(unknown)}")
```

Delete `_signed_unit`, `_allocation_benchmarks`, `_allocation_regimes`, and
`_allocation_review`. Delete stored-policy default injection from
`ensure_default_policy`. In `policy_template`, validate the active policy and
raise `PolicyStoreError` rather than inserting a missing `risk_review`.

- [ ] **Step 4: Replace retired fixtures and verify**

Remove `_allocation_review` from `tests/test_policy_store.py`; rename the
fixture `_current_policy`; retain the layer-first inspection test.

Run: `uv run pytest tests/test_policy_store.py -q`

Expected: PASS.

### Task 2: Persist and expose only valid phase5-v2 evaluations

**Files:**

- Modify: `storage/evaluation_store.py:16-74`
- Modify: `api/app.py:24-132,204-220`
- Modify: `cli.py:103-110`
- Modify: `services/inspection_engine.py:330-337`
- Modify: `services/inspection_service.py:149-170`
- Modify: `tests/test_evaluation_store.py`
- Modify: `tests/test_reduced_surface.py`

- [ ] **Step 1: Write failing persistence and API tests**

```python
def test_evaluation_store_rejects_malformed_phase5_v2_result(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "evaluation.sqlite3"))
    initialize_database()
    evaluation = _current_evaluation(snapshot_id=_insert_snapshot())
    evaluation["result"] = {"status": "Review"}
    with pytest.raises(EvaluationStorageError, match="result.account"):
        insert_evaluation_run(evaluation)

def test_api_returns_persisted_account_without_hydration(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "api.sqlite3"))
    # Persist a valid result whose account values are None, then create TestClient.
    assert client.get("/api/inspection").json()["data"]["evaluation"]["result"]["account"] == expected
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_evaluation_store.py tests/test_reduced_surface.py -q`

Expected: FAIL; malformed results persist and API replaces account fields.

- [ ] **Step 3: Implement shared v2-result validation**

```python
REQUIRED_RESULT_FIELDS = frozenset({
    "engine_version", "source", "allocation_state", "account", "layers",
    "instruments", "review_queue",
})
def current_v2_result(result: Any) -> bool:
    return isinstance(result, dict) and result.get("engine_version") == "phase5-v2" and REQUIRED_RESULT_FIELDS <= result.keys()
```

Expand the predicate to check object/list types and allowed allocation states.
Reject invalid `phase5-v2` results in `insert_evaluation_run`; use the same
predicate in API and CLI `contract_supported`. Delete `_evaluation_view`, its
performance import, and `_ACCOUNT_SUMMARY_KEYS`. Index required engine/service
inputs without legacy defaults.

- [ ] **Step 4: Update fixtures and verify**

Make `_evaluation()` contain a full valid `phase5-v2` result; retain
idempotency and history assertions.

Run: `uv run pytest tests/test_evaluation_store.py tests/test_reduced_surface.py tests/test_toss_cli.py -q`

Expected: PASS.

### Task 3: Remove candle value fallback and retain fingerprint idempotency

**Files:**

- Modify: `storage/market_store.py:21-91`
- Create: `tests/test_market_store.py`

- [ ] **Step 1: Write failing replay tests**

```python
def test_candle_replay_uses_full_identity_and_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "market.sqlite3"))
    initialize_database()
    first = insert_candles([_candle("a")])[0]
    revised = insert_candles([_candle("b")])[0]
    replay = insert_candles([_candle("a")])[0]
    assert replay["id"] == first["id"]
    assert revised["id"] != first["id"]
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_market_store.py -q`

Expected: FAIL; same values are treated as equal and `A -> B -> A` is not replay-safe.

- [ ] **Step 3: Implement full uniqueness-key lookup**

Delete `_same_candle_values`. Query the exact candle uniqueness key including
`source_fingerprint` before inserting. Return that row on an exact replay and
insert a new revision for a new fingerprint. Keep the existing greatest-id
selector in `list_adjusted_stock_candles`.

- [ ] **Step 4: Verify market behavior**

Run: `uv run pytest tests/test_market_store.py tests/test_inspection_service.py -q`

Expected: PASS.

### Task 4: Add and test the approved atomic database cleanup

**Files:**

- Modify: `storage/policy_store.py`
- Modify: `cli.py`
- Create: `tests/test_latest_contract_cleanup.py`
- Modify: `tests/test_toss_cli.py`

- [ ] **Step 1: Write failing cleanup tests**

```python
def test_cleanup_keeps_active_policy_and_candle_dates(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "cleanup.sqlite3"))
    initialize_database()
    result = cleanup_latest_contract_data()
    assert result == {"deleted_policy_versions": 1, "deleted_evaluations": 1, "deleted_candle_revisions": 1}
    assert _active_policy_count() == 1
    assert _candle_dates() == ["2026-01-01", "2026-01-02"]

def test_cleanup_rolls_back_on_retained_policy_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("PORTFOLIO_DB_PATH", str(tmp_path / "rollback.sqlite3"))
    initialize_database()
    with pytest.raises(PolicyStoreError, match="retained evaluation"):
        cleanup_latest_contract_data()
    assert _row_counts() == before
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_latest_contract_cleanup.py -q`

Expected: FAIL; cleanup function and command do not exist.

- [ ] **Step 3: Implement transaction and explicit command**

```python
def cleanup_latest_contract_data(account_alias: str = "toss-brokerage") -> dict[str, int]:
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        active = conn.execute("SELECT id, policy_json FROM ips_policy_versions WHERE account_alias = ? AND superseded_at IS NULL", (account_alias,)).fetchone()
        if active is None:
            raise PolicyStoreError("active latest policy is required")
        validate_policy(json.loads(active["policy_json"]), list_observed_identities(account_alias))
        deleted_evaluations = conn.execute("DELETE FROM ips_evaluation_runs WHERE account_alias = ? AND (engine_version != ? OR result_json NOT LIKE ?)", (account_alias, "phase5-v2", '%"engine_version":"phase5-v2"%')).rowcount
        referenced = conn.execute("SELECT 1 FROM ips_evaluation_runs WHERE account_alias = ? AND policy_version_id != ? LIMIT 1", (account_alias, active["id"])).fetchone()
        if referenced is not None:
            raise PolicyStoreError("retained evaluation references a superseded policy")
        deleted_policies = conn.execute("DELETE FROM ips_policy_versions WHERE account_alias = ? AND id != ?", (account_alias, active["id"])).rowcount
        deleted_candles = conn.execute("DELETE FROM toss_market_candles WHERE id NOT IN (SELECT MAX(id) FROM toss_market_candles GROUP BY source_kind, market_country, symbol, interval, candle_at, adjusted)").rowcount
        return {"deleted_policy_versions": deleted_policies, "deleted_evaluations": deleted_evaluations, "deleted_candle_revisions": deleted_candles}
```

Expose `ips-pilot maintenance cleanup-latest-contract`; it emits one JSON
object with deleted-row counts. No HTTP endpoint or automatic cleanup. A
retained evaluation that references a policy selected for deletion raises and
rolls back the transaction.

- [ ] **Step 4: Verify cleanup tests and command contract**

Run: `uv run pytest tests/test_latest_contract_cleanup.py tests/test_toss_cli.py -q`

Expected: PASS.

### Task 5: Execute the approved cleanup and validate

**Files:**

- Modify: no additional source files

- [ ] **Step 1: Run checks before mutation**

Run: `uv run ruff check . && uv run pytest -q`

Expected: both commands exit 0.

- [ ] **Step 2: Record candidate counts read-only**

Run: `sqlite3 -readonly data/portfolio_rebalancer.sqlite3 "SELECT count(*) FROM ips_evaluation_runs; SELECT count(*) FROM toss_market_candles;"`

Expected: counts are recorded before deletion.

- [ ] **Step 3: Run the approved local cleanup once**

Run: `uv run ips-pilot maintenance cleanup-latest-contract`

Expected: one JSON object with deletion counts.

- [ ] **Step 4: Verify runtime and full suite**

Run: `uv run ips-pilot inspection run && uv run pytest -q`

Expected: current v2 inspection JSON and a passing suite.

- [ ] **Step 5: Commit only implementation files**

Run: `git add services storage api cli.py tests && git commit -m "refactor: enforce latest-only inspection contracts"`

Expected: review the staged diff first and exclude pre-existing user changes.

## Self-review

- Spec coverage: Tasks 1–4 remove every identified code compatibility boundary; Task 5 performs the approved destructive cleanup only after tests pass.
- Placeholder scan: no deferred behavior; each contract, command, and failure case is named.
- Type consistency: `phase5-v2`, `current_v2_result`, and the full candle identity plus fingerprint key are the same across every task.
