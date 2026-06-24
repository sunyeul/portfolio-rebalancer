# Current Work Target Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the workbench "현재 작업 대상" panel with the approved left-hand dashboard-card layout while preserving all existing behavior.

**Architecture:** Keep the implementation inside the existing `frontend/src/App.tsx` workbench panel and `frontend/src/styles/app.css` component styles. The React markup remains wired to the current state, mutations, and error handling; CSS classes define the new sectioned card, responsive control row, and snapshot action rows.

**Tech Stack:** React 19, TypeScript, Vite, Tailwind v4 component layer, `lucide-react`.

---

### Task 1: Rebuild Current-Target Panel Markup

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Replace the panel body with three sections**

Edit the JSX around the `current-context-panel` section so it has:

```tsx
<section className="current-context-panel mx-auto w-full max-w-6xl" hidden={activeView !== 'workbench'}>
  <div className="context-panel-header">...</div>
  <div className="context-panel-portfolio-row">...</div>
  <div className="context-panel-snapshot-area">...</div>
</section>
```

Keep all existing handlers:

```tsx
onChange={(event) => {
  setSelectedPortfolioId(event.target.value ? Number(event.target.value) : null);
  setActiveSnapshotId(null);
}}
onClick={createNamedPortfolio}
onClick={saveCurrentSnapshot}
onClick={() => loadSnapshotMutation.mutate(snapshot.id)}
onClick={() => copySnapshotForEditing(snapshot)}
onClick={() => startEditingSnapshot(snapshot)}
onClick={() => removeSnapshot(snapshot)}
```

- [ ] **Step 2: Preserve existing error surfaces**

Keep the portfolio create error below the portfolio row and keep the snapshot mutation error below the snapshot list:

```tsx
<ErrorLine error={createPortfolioMutation.error} />
<ErrorLine error={saveSnapshotMutation.error ?? editSnapshotMutation.error ?? loadSnapshotMutation.error ?? copySnapshotMutation.error ?? updateSnapshotMutation.error ?? deleteSnapshotMutation.error} />
```

- [ ] **Step 3: Run TypeScript check**

Run:

```bash
cd frontend
npm run typecheck
```

Expected: `tsc -b` completes without TypeScript errors.

### Task 2: Add Layout Styling And Verify

**Files:**
- Modify: `frontend/src/styles/app.css`

- [ ] **Step 1: Replace old context panel CSS**

Update the existing `.current-context-panel`, `.context-panel-header`, `.context-panel-grid`, `.context-panel-fields`, `.context-panel-snapshots`, and `.snapshot-list-compact` styles so they support:

```css
.current-context-panel { overflow: hidden; padding: 0; }
.context-panel-header { padding: 20px 24px; }
.context-panel-portfolio-row { display: flex; align-items: end; justify-content: space-between; gap: 24px; padding: 20px 24px; border-bottom: 1px solid #eef2f7; }
.context-field-group { display: grid; gap: 8px; min-width: 0; }
.context-panel-snapshot-area { display: grid; gap: 16px; padding: 20px 24px; }
.snapshot-list-compact { max-height: 240px; overflow-y: auto; display: grid; gap: 12px; }
```

Add responsive rules so the portfolio row and snapshot rows stack on small screens.

- [ ] **Step 2: Run build**

Run:

```bash
cd frontend
npm run build
```

Expected: TypeScript and Vite production build complete successfully.

- [ ] **Step 3: Browser verification**

Start the dev server:

```bash
cd frontend
npm run dev
```

Open the app in the in-app browser. Verify:

- The workbench panel has header, portfolio row, and snapshot area.
- Snapshot rows look like larger card rows with right-side status and icon actions.
- Narrow viewport stacks fields and actions without text overlap.
