# Technical Target Candidate Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate role-aware technical target candidates and a non-persisted IPS preview before human activation.

**Architecture:** `services/dynamic_allocation.py` owns target ranges and eligibility. CLI and API consume its proposed policy and call the existing inspection preview service; no adapter changes inspection statuses and no code activates a policy.

**Tech Stack:** Python, Typer, FastAPI, SQLite-backed Toss observations, pytest.

---

## File structure

- `services/dynamic_allocation.py`: role-aware severe target ranges.
- `cli.py`: `market context` candidate preview.
- `api/app.py`: matching API candidate preview.
- `tests/test_dynamic_allocation.py`: target range tests.
- `tests/test_cli.py`: command contract tests.
- `tests/test_api_contract.py`: API contract tests.

### Task 1: Make severe ranges role aware

**Files:**
- Modify: `services/dynamic_allocation.py:424-439`
- Test: `tests/test_dynamic_allocation.py:212-430`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize(
    ("layer", "expected"),
    [
        ("core", {"minimum": 0.08, "maximum": 0.10}),
        ("satellite", {"minimum": 0.0, "maximum": 0.08}),
        ("experiment", {"minimum": 0.0, "maximum": 0.08}),
    ],
)
def test_severe_signal_uses_role_aware_range(layer, expected):
    policy = build_neutral_policy(_active_policy())
    item = _by_symbol(policy, "GLD")
    item.update({"layer": layer, "minimum": 0.08, "target": 0.10, "maximum": 0.12})
    assert _analysis_range("severe", item, item) == expected
```

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run pytest tests/test_dynamic_allocation.py -k severe_signal_uses_role_aware_range -v`

Expected: the core case fails because severe maps every layer to zero through its minimum.

- [ ] **Step 3: Implement the minimum behavior**

```python
if signal == "severe":
    lower, upper = (
        (minimum, target)
        if str(policy_item["layer"]) == "core"
        else (0.0, minimum)
    )
else:
    lower, upper = {
        "supportive": (target, maximum),
        "neutral": (minimum, maximum),
        "adverse": (minimum, target),
    }[signal]
```

- [ ] **Step 4: Run the module**

Run: `uv run pytest tests/test_dynamic_allocation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add services/dynamic_allocation.py tests/test_dynamic_allocation.py && git commit -m "feat: make severe target ranges role aware"`

### Task 2: Add the CLI candidate preview

**Files:**
- Modify: `cli.py:712-754`
- Test: `tests/test_cli.py:328-380`

- [ ] **Step 1: Write failing command contract tests**

```python
def test_market_context_returns_nonpersisted_candidate_preview(monkeypatch):
    monkeypatch.setattr("cli.preview_inspection", lambda policy: {
        "persisted": False,
        "snapshot_id": 10,
        "evaluation": {"allocation_state": "complete"},
    })
    # Arrange a dict proposed_policy from evaluate_dynamic_allocation.
    assert _payload(result)["candidate_evaluation"]["persisted"] is False

def test_market_context_returns_null_preview_without_proposed_policy(monkeypatch):
    # Arrange proposed_policy=None.
    assert _payload(result)["candidate_evaluation"] is None
```

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run pytest tests/test_cli.py -k market_context -v`

Expected: FAIL because `candidate_evaluation` is absent.

- [ ] **Step 3: Implement the nullable preview**

```python
candidate_evaluation = (
    preview_inspection(context["proposed_policy"])
    if isinstance(context.get("proposed_policy"), dict)
    else None
)
```

Add it after dynamic allocation and return it as `candidate_evaluation`.
Preserve candidate persistence, the one-object JSON contract, and the existing
machine-readable exception path.

- [ ] **Step 4: Run focused CLI tests**

Run: `uv run pytest tests/test_cli.py -k market_context -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add cli.py tests/test_cli.py && git commit -m "feat: preview technical target candidates in cli"`

### Task 3: Add the API candidate preview

**Files:**
- Modify: `api/app.py:14-35,438-470`
- Test: `tests/test_api_contract.py:78-145`

- [ ] **Step 1: Write failing API tests**

```python
def test_market_context_returns_nonpersisted_candidate_preview(monkeypatch, tmp_path):
    monkeypatch.setattr("api.app.preview_inspection", lambda policy: {
        "persisted": False,
        "snapshot_id": 10,
        "evaluation": {"allocation_state": "complete"},
    })
    # Arrange a dict proposed_policy and call GET /api/market-context.
    assert response.json()["data"]["candidate_evaluation"]["persisted"] is False

def test_market_context_returns_null_preview_without_candidate(monkeypatch, tmp_path):
    # Arrange proposed_policy=None.
    assert response.json()["data"]["candidate_evaluation"] is None
```

- [ ] **Step 2: Run the test to verify failure**

Run: `uv run pytest tests/test_api_contract.py -k market_context -v`

Expected: FAIL because `candidate_evaluation` is absent.

- [ ] **Step 3: Reuse the preview service**

```python
candidate_evaluation = (
    preview_inspection(context["proposed_policy"])
    if isinstance(context.get("proposed_policy"), dict)
    else None
)
```

Import `preview_inspection` and return the value under API `data`. Do not
activate a policy or adapt an inspection status, priority, queue class, or
suggestion.

- [ ] **Step 4: Run focused API tests**

Run: `uv run pytest tests/test_api_contract.py -k market_context -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add api/app.py tests/test_api_contract.py && git commit -m "feat: expose technical candidate preview in api"`

### Task 4: Verify guardrails and regressions

**Files:**
- Verify: `services/dynamic_allocation.py`, `cli.py`, `api/app.py`
- Verify: `tests/test_dynamic_allocation.py`, `tests/test_cli.py`, `tests/test_api_contract.py`

- [ ] **Step 1: Check format and lint**

Run: `uv run ruff format --check services/dynamic_allocation.py cli.py api/app.py tests/test_dynamic_allocation.py tests/test_cli.py tests/test_api_contract.py && uv run ruff check services/dynamic_allocation.py cli.py api/app.py tests/test_dynamic_allocation.py tests/test_cli.py tests/test_api_contract.py`

Expected: PASS.

- [ ] **Step 2: Run targeted tests**

Run: `uv run pytest tests/test_dynamic_allocation.py tests/test_cli.py tests/test_api_contract.py -q`

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`

Expected: PASS with only declared skips.

- [ ] **Step 4: Review the diff**

Run: `git diff HEAD -- services/dynamic_allocation.py cli.py api/app.py tests/test_dynamic_allocation.py tests/test_cli.py tests/test_api_contract.py`

Expected: no order, quantity, execution, price-target, or timing fields; no activation path.

- [ ] **Step 5: Commit**

Run: `git add services/dynamic_allocation.py cli.py api/app.py tests/test_dynamic_allocation.py tests/test_cli.py tests/test_api_contract.py && git commit -m "feat: adopt reviewed technical target candidates"`
