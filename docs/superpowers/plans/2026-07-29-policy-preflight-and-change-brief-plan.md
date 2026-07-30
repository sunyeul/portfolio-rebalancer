# Policy Preflight and Change Brief Implementation Plan

> **For agentic workers:** Execute this plan inline task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the dashboard ask the missing policy-change questions first and expose a read-only brief of the newest inspection changes.

**Architecture:** Two pure backend services create immutable response payloads from the active policy and persisted evaluations. The API and JSON CLI expose those payloads; the React workbench renders them without sending or persisting answers.

**Tech Stack:** Python, SQLite, FastAPI, Typer, pytest, React, TypeScript, Vite, Bun.

---

### Task 1: Add evaluation history and pure briefing services

**Files:**
- Create: `services/change_brief.py`
- Create: `services/policy_preflight.py`
- Modify: `storage/evaluation_store.py`
- Test: `tests/test_change_brief.py`
- Test: `tests/test_policy_preflight.py`
- Test: `tests/test_evaluation_store.py`

- [x] Write failing tests for two newest evaluation rows, a source-blocked
  preflight question, baseline briefing, queue additions/changes/resolutions,
  and source alerts.
- [x] Add `list_evaluation_runs(limit, account_alias)` ordered by
  `created_at DESC, id DESC` without changing existing latest lookup.
- [x] Implement pure preflight and brief builders that return only structured
  questions, existing verification needs, source state, identifiers, and queue
  diffs; do not mutate any input.
- [x] Run focused service and store tests.

### Task 2: Expose read-only API and CLI surfaces

**Files:**
- Modify: `api/app.py`
- Modify: `cli.py`
- Modify: `tests/test_api_contract.py`
- Modify: `tests/test_cli.py`

- [x] Add `GET /api/policy-preflight` and `GET /api/change-brief` that use the
  services with current persisted data and return no write action.
- [x] Add `ips-pilot inspection brief`, emitting exactly one JSON object with
  current/previous run IDs, the brief, and the existing contract gate.
- [x] Test API payload passthrough and CLI single-object output.
- [x] Run focused API and CLI tests.

### Task 3: Render local-only questions and brief

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`

- [x] Add typed response shapes and fetch both endpoints during dashboard
  reload, keeping failures as non-blocking read warnings.
- [x] Add a Policy Preflight tab that renders server questions with local
  textarea answers and a clear notice that nothing is saved or applied.
- [x] Add a Change Brief tab that renders source alerts and only queue deltas.
- [x] Run frontend typecheck, unaffected frontend tests, and production build.

### Task 4: Adversarial review and full verification

**Files:**
- Verify: all files above plus existing Review Queue red-team files

- [x] Confirm no new function writes to policy, snapshot, evaluation, journal,
  scheduler, notification channel, or broker endpoint.
- [x] Confirm the brief compares stored fields but never computes status,
  priority, suggestion, or transaction direction.
- [x] Scan new wording for order authority and run focused then full backend and
  frontend tests, build, and `git diff --check`.
