# Review Queue Red-Team Implementation Plan

> **For agentic workers:** Execute this plan inline task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, inspection-only counterarguments to each Review Queue item and render them in the read-only workbench.

**Architecture:** Keep the inspection engine as the sole producer of Review Queue content. It will derive a `red_team` explanation from the existing item kind and blocking state while preserving all decision fields. The frontend will only render that payload.

**Tech Stack:** Python, pytest, React, TypeScript, Vite, Bun.

---

### Task 1: Extend the backend Review Queue contract

**Files:**
- Modify: `services/inspection_engine.py`
- Test: `tests/test_inspection_engine.py`

- [x] **Step 1: Write a failing focused queue test**

Add a test that evaluates an out-of-range instrument and asserts its queue row
contains:

```python
assert item["red_team"] == {
    "counterargument": "비중 범위 이탈만으로 거래나 예외 개입을 확정할 수 없습니다.",
}
```

Also assert the persisted `status`, `priority`, `suggestion`, and
`queue_class` have their existing values.

- [x] **Step 2: Run the focused test and confirm failure**

Run:

```bash
uv run pytest -q tests/test_inspection_engine.py -k red_team
```

Expected: failure because `red_team` is absent.

- [x] **Step 3: Add a deterministic explanation helper**

In `services/inspection_engine.py`, add a private helper that receives the
queue item and `blocking`. It returns exactly `counterargument`. Use separate
Korean counterarguments for blocking source, allocation (`cash`, `layer`,
`instrument`), and performance (`performance`, `account_risk`) items.

- [x] **Step 4: Attach the helper result only during queue projection**

Add `"red_team": _red_team(item, blocking=blocking)` to `_queue_item`'s
existing dictionary. Do not change `_queue_item` sorting inputs, item status,
priority, suggestion, queue class, or evaluation persistence.

- [x] **Step 5: Run focused and full engine tests**

Run:

```bash
uv run pytest -q tests/test_inspection_engine.py
```

Expected: pass.

### Task 2: Preserve the API and frontend data contract

**Files:**
- Modify: `tests/test_api_contract.py`
- Modify: `frontend/src/lib/api.ts`

- [x] **Step 1: Write an API assertion for persisted red-team data**

Extend the persisted evaluation fixture in
`test_api_returns_persisted_phase5_evidence_without_reclassification` with a
`red_team` object and assert the `/api/inspection` result returns it unchanged.

- [x] **Step 2: Type the read-only field**

Add an `InspectionRedTeam` type with an optional `counterargument` field. Add
`red_team?: InspectionRedTeam` to `InspectionItem`. Do not add client-side
status or priority computation.

- [x] **Step 3: Run focused backend contract tests**

Run:

```bash
uv run pytest -q tests/test_api_contract.py tests/test_inspection_engine.py
```

Expected: pass.

### Task 3: Render the red-team card in Review Queue

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [x] **Step 1: Render only server-provided text**

Within `ReviewQueue`, render a `반대 관점` block when `item.red_team` exists.
Display `counterargument`; do not render the block from triggers or infer its
content in the client.

- [x] **Step 2: Add compact responsive styling**

Add a `.red-team-card` style below the existing queue styles. It must visually
separate the explanatory block without changing Review Queue grid, source
order, status presentation, or existing controls.

- [x] **Step 3: Typecheck and build the frontend**

Run:

```bash
bun run --cwd frontend typecheck
bun run --cwd frontend build
```

Expected: both commands pass.

### Task 4: Adversarial review and final verification

**Files:**
- Verify: `services/inspection_engine.py`
- Verify: `tests/test_inspection_engine.py`
- Verify: `tests/test_api_contract.py`
- Verify: `frontend/src/lib/api.ts`
- Verify: `frontend/src/App.tsx`
- Verify: `frontend/src/styles.css`

- [x] **Step 1: Check decision fields and order remain backend-owned**

Inspect the final diff and confirm `red_team` is additive, is not consumed for
sorting, and does not modify status, priority, suggestion, queue class, or
allocation state.

- [x] **Step 2: Scan for execution-language regressions**

Run:

```bash
rtk rg -n '매수|매도|거래|주문|수량|가격|실행' services/inspection_engine.py frontend/src/App.tsx
```

Expected: all new matches say that a signal does not establish a transaction;
there must be no instruction, side, quantity, price, or execution field.

- [x] **Step 3: Run focused checks and review the working tree**

Run:

```bash
uv run pytest -q tests/test_inspection_engine.py tests/test_api_contract.py
bun run --cwd frontend build
rtk git diff --check
rtk git status --short
```

Expected: all checks pass and only the planned source, test, and design/plan
files are changed.
