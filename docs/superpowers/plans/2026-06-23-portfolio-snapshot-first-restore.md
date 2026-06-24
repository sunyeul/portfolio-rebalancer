# Portfolio Snapshot First Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the first frontend slice of saved portfolio and snapshot management.

**Architecture:** Keep the change local to `frontend/src/App.tsx` and existing CSS utility classes. Reuse the existing persistence API client functions, and map loaded portfolio assets back into the existing editable input row shape.

**Tech Stack:** React 19, TypeScript, Vite, existing FastAPI JSON endpoints.

---

### Task 1: Wire Persistence State

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Import React effect and persistence API helpers**

Use `useEffect` alongside the existing state hooks, and import the existing saved portfolio and snapshot API types/functions.

- [ ] **Step 2: Add helpers for loaded asset rows**

Create a helper that converts `AssetRow[]` into `PortfolioInputRow[]`, mapping legacy `intact` and current `valid` to editable `valid`.

- [ ] **Step 3: Add management state and refresh functions**

Add saved portfolio list, selected portfolio id, snapshots, active snapshot id, new portfolio name, management loading state, and management error state.

### Task 2: Add Management Actions

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Load portfolios on mount**

Call `listPortfolios()` in an effect and display errors in the management panel.

- [ ] **Step 2: Load selected portfolio context**

When the selected portfolio changes, call `getCurrentState()` and `listSnapshots()`. Treat missing current state as non-fatal.

- [ ] **Step 3: Create portfolio**

Use `createPortfolio({ name })`, refresh the list, select the created id, and clear the input.

- [ ] **Step 4: Save current state**

Use `saveCurrentState(selectedPortfolioId)`, apply the returned state to the workbench, and refresh portfolio summaries and snapshots.

- [ ] **Step 5: Load snapshot**

Use `loadSnapshot(snapshot.id)`, apply the returned snapshot state, mark it active, and refresh portfolio summaries.

### Task 3: Render The First-Restore Panel

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Add the panel next to the preview/settings area**

Render portfolio select, create input/button, current-state save button, and a snapshot list with load buttons.

- [ ] **Step 2: Keep advanced snapshot controls out**

Do not render edit, copy, or delete actions in this first pass.

- [ ] **Step 3: Run verification**

Run `npm run build` from `frontend`. If backend response-shape issues appear, run the relevant `uv run pytest` tests.
