# Table Sorting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sortable headers to every visible column in the layer dashboard and asset evaluation table.

**Architecture:** Keep sorting presentation-only inside `frontend/src/App.tsx`. Add shared sort state, comparison helpers, and a reusable header component, then let each table define its own sortable column accessors and sorted row memo.

**Tech Stack:** React 19, TypeScript, lucide-react icons, Vite.

---

### Task 1: Shared Sorting Helpers

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Add sort icon imports**

Add `ArrowDown`, `ArrowUp`, and `ArrowUpDown` to the existing `lucide-react` import.

- [ ] **Step 2: Add reusable sort types and helpers**

Add `SortDirection`, `SortState`, `SortValue`, `nextSortState`, `compareSortValues`, and `sortRows` near the existing UI helper functions in `frontend/src/App.tsx`.

- [ ] **Step 3: Add `SortableHeader`**

Add a compact button component that renders the label, active direction icon, and accessible sort state.

### Task 2: Layer Dashboard Sorting

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Add layer column metadata**

Define all visible layer dashboard columns with labels and accessors: layer, current weight, gap, period return, MDD, efficiency, status.

- [ ] **Step 2: Wire sort state and sorted rows**

Use `useState` for the active column/direction and `useMemo` to derive sorted rows from the shared helper.

- [ ] **Step 3: Replace layer table headers**

Render every layer header through `SortableHeader`.

### Task 3: Asset Evaluation Table Sorting

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Add asset column metadata**

Define all visible asset table columns with labels and accessors: layer, ticker, weight, layer-internal weight, period return, CAGR, MDD, risk contribution, thesis, status.

- [ ] **Step 2: Wire sort state and sorted rows**

Use `useState` for the active column/direction and `useMemo` to derive sorted rows from the shared helper.

- [ ] **Step 3: Replace asset table headers**

Render every asset header through `SortableHeader`.

### Task 4: Verification

**Files:**
- Test: `frontend/src/App.tsx`

- [ ] **Step 1: Run frontend typecheck**

Run: `bun run typecheck`

Expected: TypeScript passes without errors.

- [ ] **Step 2: Inspect git diff**

Run: `git diff -- frontend/src/App.tsx docs/superpowers/specs/2026-06-24-table-sorting-design.md docs/superpowers/plans/2026-06-24-table-sorting.md`

Expected: Diff is limited to the sorting UI, design doc, and plan doc.
