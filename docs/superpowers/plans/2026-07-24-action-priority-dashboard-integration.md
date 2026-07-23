# Action Priority Dashboard Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the result-first allocation-adjustment contract into the IPS judgment skill, Toss inspection engine, API/CLI, and read-only dashboard.

**Architecture:** Keep all status, priority, evaluability, suggestion, and ordering logic in a small backend action-contract module called by the inspection engine. Persist `phase5-v2` results with separate `adjustment_suggestions` and `review_queue` lists; keep historical payloads unchanged and expose an explicit contract gate. Render backend-owned suggestions first in Overview, followed by compact allocation bands and secondary account facts.

**Tech Stack:** Python 3.12, SQLite persistence, Typer CLI, FastAPI loopback API, React/TypeScript, Bun tests/build, pytest.

---

## Task 1: Lock the action-contract behavior with offline tests

**Files:**
- Create: `tests/test_action_contract.py`
- Modify: `tests/test_inspection_engine.py`

- [ ] **Step 1: Write failing contract tests**

Add tests for the exact mappings below:

```python
def test_cash_floor_is_p1_pace_reduction_review():
    item = evaluate_cash_action(current=0.08, minimum=0.10, maximum=0.20)
    assert item == {
        "priority": "P1",
        "priority_label": "다음 정기매수 전",
        "suggestion": {
            "code": "review_reduce_or_pause_regular_purchase_pace",
            "label": "향후 정기매수 속도 축소·중단 검토",
        },
    }

def test_valid_underweight_satellite_is_p3_allocation_review():
    item = evaluate_instrument_action(
        kind="instrument", layer="satellite", current=0.01,
        minimum=0.02, maximum=0.05, cash_current=0.15,
        cash_minimum=0.10,
        profile={
            "thesis_status": "valid",
            "overlap_status": "clear",
            "management_burden_status": "clear",
            "holdability_status": "clear",
            "etf_substitution_status": "not_applicable",
        },
    )
    assert item["priority"] == "P3"
    assert item["suggestion"]["code"] == "review_increase_regular_purchase_allocation"

def test_broken_thesis_and_own_max_is_p1_action():
    item = evaluate_instrument_action(
        kind="instrument", layer="satellite", current=0.06,
        minimum=0.01, maximum=0.05, cash_current=0.15,
        cash_minimum=0.10, profile={"thesis_status": "broken"},
    )
    assert item["status"] == "Action"
    assert item["priority"] == "P1"
    assert item["suggestion"]["code"] == "inspect_exceptional_intervention"

def test_priority_sort_precedes_status_severity():
    ordered = sort_review_items([
        {"priority": "P3", "status": "Action", "kind": "instrument", "identity": "US/A"},
        {"priority": "P1", "status": "Review", "kind": "cash", "identity": "cash_reserve"},
    ])
    assert [item["identity"] for item in ordered] == ["cash_reserve", "US/A"]
```

Also add engine cases for: optional performance missing while allocation remains evaluable; stale source producing no adjustment suggestions; all-cash projection producing a cash-only partial allocation result; configured absent instrument becoming `current = 0` when invested value is positive; and overweight plus secondary factor review choosing overweight normalization.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
uv run pytest tests/test_action_contract.py tests/test_inspection_engine.py -q
```

Expected: FAIL because the action-contract module, v2 fields, and decoupled allocation evaluation do not yet exist.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_action_contract.py tests/test_inspection_engine.py
git commit -m "test: define priority-led allocation actions"
```

## Task 2: Implement the backend action contract and phase5-v2 evaluation

**Files:**
- Create: `services/action_contract.py`
- Modify: `services/inspection_engine.py`
- Modify: `services/inspection_service.py`
- Modify: `tests/test_action_contract.py`
- Modify: `tests/test_inspection_engine.py`

- [ ] **Step 1: Add the closed contract vocabulary**

Implement constants and pure helpers in `services/action_contract.py`:

```python
ENGINE_VERSION = "phase5-v2"
PRIORITY_LABELS = {
    "P1": "다음 정기매수 전",
    "P2": "이번 월간 점검",
    "P3": "다음 정기매수 배분 반영",
    "P4": "관찰 유지",
}
SUGGESTION_LABELS = {
    "review_increase_regular_purchase_pace": "향후 정기매수 속도 확대 검토",
    "review_reduce_or_pause_regular_purchase_pace": "향후 정기매수 속도 축소·중단 검토",
    "review_increase_regular_purchase_allocation": "향후 정기매수 배분 확대 검토",
    "review_overweight_normalization": "정기매수 축소·중단 우선 후 초과비중 정상화 검토",
    "review_thesis_or_constraints": "논지와 제약 조건 검토",
    "inspect_exceptional_intervention": "예외적 개입 가능성 검토",
    "hold_and_observe": "관찰 유지",
}
```

The module also owns exact status severity, queue classes (`blocking`, `adjustment`, `observation`), layer-specific underweight readiness, trigger-to-suggestion precedence, priority-first sorting, and deterministic allocation blocking-reason precedence. It must not emit order quantity, order side, price, or execution fields.

- [ ] **Step 2: Refactor inspection evaluation around allocation state**

In `services/inspection_engine.py`:

1. Set `ENGINE_VERSION = "phase5-v2"`.
2. Build `allocation_state` as `complete`, `partial`, or `not_evaluable`.
3. Require complete/reconciled Toss source and valid policy/profile coverage for normal allocation evaluation.
4. Allow a valid cash unit to remain evaluable when invested value is zero; suppress layer/instrument weights in that `partial` state.
5. Evaluate the union of current positions and configured instruments; configured absent instruments are zero only when the invested denominator is valid.
6. Stop making a missing/partial performance run a global allocation blocker. Keep performance status and unavailable principal/YTD fields separate.
7. Add `adjustment_suggestions` from P1–P3 cash/layer/instrument items and retain all non-OK items in `review_queue` with `queue_class`.
8. Apply suggestion precedence after all triggers: exact Action, cash band, overweight, eligible underweight, thesis/profile constraints, hold/observe.
9. Keep the exact same-instrument Action conjunction.
10. Replace new result-level `state`/`non_evaluable_reason` with `allocation_state`/`allocation_reason`; the persistence wrapper maps complete/partial to persisted `state = complete`.

- [ ] **Step 3: Update service persistence inputs**

Update `services/inspection_service.py` to use `result["allocation_state"]` and `result["allocation_reason"]`, preserve the performance fingerprint when available, and persist the new engine version without mutating existing evaluations.

- [ ] **Step 4: Run the focused backend tests**

Run:

```bash
uv run pytest tests/test_action_contract.py tests/test_inspection_engine.py tests/test_inspection_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit backend judgment behavior**

```bash
git add services/action_contract.py services/inspection_engine.py services/inspection_service.py tests/test_action_contract.py tests/test_inspection_engine.py tests/test_inspection_service.py
git commit -m "feat: emit priority-led allocation suggestions"
```

## Task 3: Add API and CLI contract gating

**Files:**
- Modify: `api/app.py`
- Modify: `cli.py`
- Modify: `frontend/src/lib/api.ts`
- Modify: `tests/test_api_contract.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing gate tests**

Cover these exact response contracts:

```python
assert inspection["data"]["contract_supported"] is False  # historical v1
assert queue["data"]["contract_supported"] is False
assert run_payload["contract_supported"] is True  # phase5-v2 result
assert show_payload["contract_supported"] is True
```

Assert that historical raw priority fields remain untouched and that no v1 record is converted into a v2 suggestion at read time. Assert that the review-queue endpoint returns `adjustment_suggestions` only for a v2 evaluation.

- [ ] **Step 2: Implement the gate**

Add one helper that returns `evaluation.engine_version == "phase5-v2"`. Expose it at `/api/inspection.data.contract_supported`, `/api/review-queue.data.contract_supported`, and top-level CLI `inspection show`, `inspection preview`, and `inspection run`. Keep old evaluation JSON raw and return an explicit old-contract notice to the frontend.

- [ ] **Step 3: Run API and CLI tests**

```bash
uv run pytest tests/test_api_contract.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit the transport contract**

```bash
git add api/app.py cli.py frontend/src/lib/api.ts tests/test_api_contract.py tests/test_cli.py
git commit -m "feat: gate allocation action contract by engine version"
```

## Task 4: Rebuild the Overview around backend suggestions

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/presentation.ts`
- Modify: `frontend/tests/presentation.test.ts`
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Add explicit TypeScript shapes and presentation tests**

Define `AdjustmentSuggestion`, `QueueItem`, and v2 result fields in `frontend/src/lib/api.ts`. Add tests for unavailable account facts, exact Korean priority/suggestion labels, denominator labels, and old-contract gating.

- [ ] **Step 2: Render result-first Overview**

Implement these components in `App.tsx`:

1. `AdjustmentSuggestions`: render only `result.adjustment_suggestions.slice(0, 3)` in backend order; show priority label, status, target, current/band, suggestion label, meaning, and a collapsed evidence disclosure.
2. `AllocationBands`: render cash and three layers with explicit denominators; keep layers separate from cash.
3. `AccountContext`: render investment principal, account value, principal-relative return, and YTD TWR/10% target as compact secondary facts.
4. `ContractGateNotice`: when `contract_supported` is false, do not infer suggestions from old `review_queue`; show the explicit old-contract notice.

Keep full Review Queue, Performance, Allocation, Profiles, and Source Health tabs. Remove the duplicate full cash reserve/layer review blocks from Overview only. Do not add client-side status, priority, or action selection logic.

- [ ] **Step 3: Style and verify the narrow layout**

Keep existing sidebar collapse, focus-visible states, horizontal narrow navigation, scrollable tables, and denominator labels. Add visual distinction between priority and status chips without introducing buy/sell color semantics.

- [ ] **Step 4: Run frontend checks**

```bash
bun test --cwd frontend
bun run --cwd frontend typecheck
bun run --cwd frontend build
```

Expected: all tests pass and the production bundle builds.

- [ ] **Step 5: Commit the dashboard**

```bash
git add frontend/src/App.tsx frontend/src/lib/api.ts frontend/src/lib/presentation.ts frontend/tests/presentation.test.ts frontend/src/styles.css
git commit -m "feat: show priority-led allocation review"
```

## Task 5: Update and test the IPS judgment skill and durable project docs

**Files:**
- Modify: `.agents/skills/ips-judgment-filter/SKILL.md`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-22-cash-account-observability-roadmap-design.md`

- [ ] **Step 1: Run RED pressure scenarios without the new skill section**

Use independent agents against these prompts and record violations before editing the skill:

1. Cash below minimum with no damaged thesis: must be Review/P1 and future regular-purchase pace reduction review, never Action or sell.
2. YTD below target plus underweight core: performance Watch/P4; core may be Review/P3 only when cash and thesis allow.
3. Broken thesis plus same-instrument own maximum breach: Action/P1 exceptional-intervention inspection without size.
4. Toss source already stale with a large gap: non-evaluable and no allocation suggestion.

- [ ] **Step 2: Add the minimum guidance that fixes observed violations**

Extend `ips-judgment-filter` with the four-axis separation, exact P1–P4 timing labels, priority-first ordering, suggestion vocabulary, performance-as-context rule, and the exact Action conjunction. Keep the skill concise and do not introduce new numeric policy thresholds.

- [ ] **Step 3: Run GREEN/REFACTOR pressure scenarios**

Run the same prompts with the edited skill. Add explicit anti-rationalization wording only when an agent still conflates status with priority or emits direct trade semantics.

- [ ] **Step 4: Update durable project references**

Document that the v2 action contract is fixed product semantics while numeric cash/layer/instrument/performance thresholds remain versioned IPS policy values. Document the new Overview order and the historical-engine gate in README and the roadmap.

- [ ] **Step 5: Commit skill and documentation changes**

```bash
git add .agents/skills/ips-judgment-filter/SKILL.md README.md docs/superpowers/specs/2026-07-22-cash-account-observability-roadmap-design.md
git commit -m "docs: align IPS guidance with priority-led actions"
```

## Task 6: Full offline verification and browser review

**Files:**
- Modify only if verification discovers a defect in the implementation files above.

- [ ] **Step 1: Run the complete Python suite and lint**

```bash
uv run pytest -q
uv run ruff check services api cli.py storage tests
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 2: Run the complete Bun/frontend suite**

```bash
bun test --cwd frontend
bun run --cwd frontend typecheck
bun run --cwd frontend build
```

Expected: all tests pass and `frontend/dist` builds.

- [ ] **Step 3: Verify no forbidden semantics**

Search the new engine result, API payload fixtures, skill, and Overview code for `order size`, `buy`, `sell`, `execute`, `quantity`, or execution flags. Any occurrence must be explanatory guardrail text or existing hidden broker-source code, never a suggestion field.

- [ ] **Step 4: Browser-check the dashboard**

Run the existing loopback dashboard API and frontend development server. Verify on desktop and narrow viewport:

- v2 adjustment suggestions appear first and preserve backend order;
- priority timing and status are visually distinct;
- current/band and denominator labels are visible;
- investment principal, account value, account return, and YTD context appear below the action section;
- old-contract notice appears when the persisted evaluation is v1;
- no direct order instruction or sizing appears.

- [ ] **Step 5: Review the final diff and commit only scoped files**

```bash
git status --short
git diff --check HEAD~5..HEAD
```

Do not stage `data/state_store.db/*`, `node_modules/`, `draft.md`, the HTML mockup, or other user-owned unrelated files. Do not run live Toss sync or activate a new policy as part of this verification.
