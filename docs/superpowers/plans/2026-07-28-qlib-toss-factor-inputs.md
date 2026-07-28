# Qlib Toss Factor Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Populate Qlib Stage 1 inputs only from official Toss daily candles and derive a causally safe 20-session factor so the Qlib data adapter can be evaluated on real data.

**Architecture:** Extend the existing market sync command with an opt-in research-only collection mode. It persists immutable normalized candles but bypasses dynamic-allocation evaluation and policy-candidate persistence. During read-only Qlib export, derive a 20-session trailing return per series and omit its warm-up rows; no factor is stored in the operational database.

**Tech Stack:** Python 3.12, Typer, SQLite, Toss authorized market API integration, pytest, Ruff, isolated pyqlib==0.9.7 environment.

---

### Task 1: Add a policy-safe Toss research collection mode

**Files:**
- Modify: cli.py:445-529
- Modify: tests/test_cli.py:280-355

- [ ] **Step 1: Write the failing CLI test**

Add test_market_sync_research_only_collects_policy_universe_without_policy_side_effects. Give it a policy with US/SPY and US/GLD, one observed identity US/SPY, and a fake service. Invoke:

~~~
result = runner.invoke(
    app,
    [
        "market", "sync", "--research-only", "--target-points", "756",
        "--max-pages", "4",
    ],
)
~~~

Assert each collect_history call uses target_points == 756, calls include both policy instruments plus all allocation benchmarks, and JSON has research_only true, target_points 756, context null, candidate null. Monkeypatch evaluate_dynamic_allocation and insert_policy_candidate to raise AssertionError; success proves the mode cannot reach either mutation path.

- [ ] **Step 2: Run the new test to verify it fails**

Run:

~~~
uv run pytest tests/test_cli.py::test_market_sync_research_only_collects_policy_universe_without_policy_side_effects -q
~~~

Expected: FAIL because the command does not yet recognize the research-only options.

- [ ] **Step 3: Implement the minimal command extension**

In market_sync_command add:

~~~
target_points: Annotated[int, typer.Option("--target-points")] = 252,
research_only: Annotated[bool, typer.Option("--research-only")] = False,
~~~

Raise CliError("input", "--target-points는 1 이상이어야 합니다.") below one. In research-only mode add all active policy instrument identities to selected_stocks, pass target_points to every stock and indicator collection call, and skip evaluate_dynamic_allocation plus insert_policy_candidate. Emit context=None, candidate=None, research_only, and target_points. Preserve the existing dynamic-context behavior without the flag.

- [ ] **Step 4: Run focused CLI regression tests**

Run:

~~~
uv run pytest tests/test_cli.py -q
~~~

Expected: PASS; normal market sync behavior remains intact and research-only sync collects candles only.

- [ ] **Step 5: Commit the isolated collection change**

~~~
git add cli.py tests/test_cli.py
git commit -m "feat: add qlib research market sync"
~~~

### Task 2: Derive a no-lookahead Qlib factor from verified candles

**Files:**
- Modify: research/qlib_validation/source.py:1-371
- Modify: tests/research/test_qlib_source.py:1-276
- Modify: tests/research/test_qlib_capability.py:1-31
- Modify: tests/research/test_qlib_integration.py:1-150

- [ ] **Step 1: Write failing factor tests**

Seed 21 weekday candles for every required series with close values 100.0 through 120.0. Assert load_snapshot returns only the final candle per series and factor == (120.0 / 100.0) - 1.0. Seed a 21st candle whose available_at is after as_of and assert no factored candle is returned. This makes future-data use detectable.

Update the capability tests: a snapshot whose every returned candle has a factor must be adapter-suitable; a fixture with any None factor remains fail-closed.

- [ ] **Step 2: Run research tests to verify they fail**

Run:

~~~
research/qlib_validation/.venv/bin/python -m pytest tests/research/test_qlib_source.py tests/research/test_qlib_capability.py -q
~~~

Expected: FAIL because source currently returns factor=None and does not remove warm-up rows.

- [ ] **Step 3: Implement causal factor derivation**

In source.py define FACTOR_LOOKBACK_SESSIONS = 20 and add:

~~~
def _with_trailing_factors(candles: tuple[Candle, ...]) -> tuple[Candle, ...]:
    if len(candles) <= FACTOR_LOOKBACK_SESSIONS:
        return ()
    result = []
    for index in range(FACTOR_LOOKBACK_SESSIONS, len(candles)):
        baseline = candles[index - FACTOR_LOOKBACK_SESSIONS]
        current = candles[index]
        result.append(
            replace(
                current,
                factor=current.close_price / baseline.close_price - 1.0,
            )
        )
    return tuple(result)
~~~

Import replace from dataclasses. Validate the full raw series first, then append _with_trailing_factors(series) in load_snapshot. Do not add a database column or any write path.

- [ ] **Step 4: Make Qlib integration expectations explicit**

In test_real_cli_keeps_fixture_database_bytes_unchanged, after reading successful JSON, add:

~~~
assert result["qlib_capability"]["data_adapter_suitable"] is True
assert result["qlib_capability"]["reasons"] == []
~~~

Keep the database-byte hash assertion to prove factors are derived only in memory.

- [ ] **Step 5: Run the isolated Qlib suite**

Run:

~~~
research/qlib_validation/.venv/bin/python -m pytest tests/research -q
~~~

Expected: PASS, including StaticDataLoader round-trip and source read-only tests.

- [ ] **Step 6: Commit the factor change**

~~~
git add research/qlib_validation/source.py tests/research/test_qlib_source.py tests/research/test_qlib_capability.py tests/research/test_qlib_integration.py
git commit -m "feat: derive causal qlib research factor"
~~~

### Task 3: Document and verify the supported research workflow

**Files:**
- Modify: research/qlib_validation/README.md:1-17
- Modify: tests/research/test_qlib_cli.py only if output contract changes

- [ ] **Step 1: Document exact collection and analysis commands**

Add this collection command before Stage 1:

~~~
ips-pilot market sync --research-only --target-points 756 --max-pages 4
~~~

Explain that it writes immutable Toss candles only, does not create policy candidates, and derives a 20-session trailing-return factor in memory. State that adapter success does not enable orders, IPS statuses, model training, or Stage 2 policy conclusions.

- [ ] **Step 2: Run formatting and regression checks**

Run:

~~~
uv run ruff format --check cli.py research/qlib_validation tests/test_cli.py tests/research
uv run ruff check cli.py research/qlib_validation tests/test_cli.py tests/research
uv run pytest tests/test_cli.py tests/test_toss_market.py -q
research/qlib_validation/.venv/bin/python -m pytest tests/research -q
~~~

Expected: all pass; tests must not make a live Toss call.

- [ ] **Step 3: Commit documentation**

~~~
git add research/qlib_validation/README.md
git commit -m "docs: explain qlib research data refresh"
~~~

### Task 4: Perform the user-authorized live data and Qlib verification

**Files:**
- Create (ignored artifacts only): research/qlib_validation/artifacts/live-qlib-factor-<run-id>/

- [ ] **Step 1: Read aggregate readiness state**

Query active policy version and per-series candle counts from SQLite without exposing holdings. Confirm that historical data is needed.

- [ ] **Step 2: Populate official Toss candles in research-only mode**

Run:

~~~
ips-pilot market sync --research-only --target-points 756 --max-pages 4
~~~

Expected: one JSON object with context and candidate both null. On authentication, network, normalization, or conflicting historical observations, stop and report the machine-readable error rather than starting Stage 1.

- [ ] **Step 3: Run Stage 1 against the resulting database**

Run:

~~~
research/qlib_validation/.venv/bin/python -m research.qlib_validation.cli stage1 --db data/portfolio_rebalancer.sqlite3 --as-of 2026-07-28T12:00:00+00:00 --output research/qlib_validation/artifacts/live-qlib-factor
~~~

Expected: one JSON object and immutable artifacts. Report freshness, adapter suitability, research verdict, and exclusions without creating a trade or IPS instruction.

- [ ] **Step 4: Commit code only**

~~~
git status --short
git add cli.py research/qlib_validation tests/test_cli.py tests/research
git commit -m "feat: complete qlib toss factor inputs"
~~~

Do not stage data/ or research/qlib_validation/artifacts/.
