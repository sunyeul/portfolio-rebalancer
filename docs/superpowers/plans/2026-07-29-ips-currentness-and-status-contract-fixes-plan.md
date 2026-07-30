# IPS Currentness and Status Contract Fixes Implementation Plan

> **For agentic workers:** Execute this plan inline task-by-task with focused tests before broad regression checks.

**Goal:** Stop presenting stale persisted evaluations as current, remove the annual-target status bug, and make evidence-only Review items use monthly observation semantics.

**Architecture:** Keep evaluation history immutable. Derive currentness at the API boundary from the latest persisted evaluation, the sole current evaluable Toss snapshot, and the active IPS policy. Keep status and priority logic in backend services; the React client only gates and explains server-owned state.

**Tech Stack:** Python, FastAPI, pytest, React, TypeScript, Vite, Bun.

---

### Task 1: Add evaluation-currentness API contract

**Files:**
- Modify: `api/app.py`
- Test: `tests/test_api_contract.py`

- [x] Add failing tests for matching snapshot/policy IDs and for snapshot or policy mismatch.
- [x] Implement a deterministic currentness object with stable reason codes.
- [x] Return currentness from `/api/inspection` and `/api/review-queue`.
- [x] Suppress Review Queue items and adjustment suggestions when currentness is false.
- [x] Suppress Change Brief items when its newest persisted evaluation is stale.
- [x] Run `uv run pytest -q tests/test_api_contract.py`.

### Task 2: Correct performance and Review decision semantics

**Files:**
- Modify: `services/inspection_engine.py`
- Modify: `services/action_contract.py`
- Test: `tests/test_inspection_engine.py`
- Test: `tests/test_action_contract.py`

- [x] Add a failing test proving below-target annual return remains descriptive.
- [x] Add failing tests proving evidence-only Review maps to P2, `hold_and_observe`, and observation queue class.
- [x] Remove the annual-target status trigger without removing its metric.
- [x] Update backend priority and queue-class logic while preserving cash and allocation precedence.
- [x] Run `uv run pytest -q tests/test_action_contract.py tests/test_inspection_engine.py`.

### Task 3: Gate stale evaluation results in the workbench

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/presentation.ts`
- Modify: `frontend/src/App.tsx`
- Test: `frontend/tests/presentation.test.ts`

- [x] Type the server currentness object.
- [x] Add a pure Korean currentness explanation formatter and tests.
- [x] Render a verification banner for stale evaluations and do not render or fetch their result-dependent panels.
- [x] Do not request or render Change Brief while evaluation currentness is false.
- [x] Run `bun test --cwd frontend` and `bun run --cwd frontend build`.

### Task 4: Align durable repository contracts

**Files:**
- Modify: `AGENTS.md`

- [x] Replace unavailable snapshot `created_at` selection with `synced_at`, then `id`.
- [x] Define stricter Satellite and Experiment checks using runtime-owned role, range, and drawdown evidence.

### Task 5: Adversarial regression review

**Files:**
- Verify all files above without changing snapshots, policies, journal entries, or persistent app state.

- [x] Run focused backend and frontend suites.
- [x] Run the broader inspection/API suite that covers shared contracts.
- [x] Confirm stale data cannot surface as current through either API route or the React result path.
- [x] Confirm no order side, quantity, price, execution flag, or automatic refresh was added.
- [x] Run `git diff --check` and distinguish these changes from the pre-existing Review Queue red-team work.
