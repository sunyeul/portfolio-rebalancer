# Review Copilot Sidebar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Review Copilot's floating popup with a collapsible right-side sidebar.

**Architecture:** Keep `ReviewCopilot` as the integration point for CopilotKit context and frontend tools. Swap only the rendered CopilotKit chat shell from `CopilotPopup` to `CopilotSidebar`, using the library's built-in right-positioned collapsible behavior.

**Tech Stack:** React, TypeScript, CopilotKit React Core v2, Vite/Bun frontend.

---

### Task 1: Swap Review Copilot Shell

**Files:**
- Modify: `frontend/src/copilot/ReviewCopilot.tsx`

- [ ] **Step 1: Update the CopilotKit import**

Replace:

```tsx
import { CopilotKit, CopilotPopup, useAgentContext, useFrontendTool } from '@copilotkit/react-core/v2';
```

With:

```tsx
import { CopilotKit, CopilotSidebar, useAgentContext, useFrontendTool } from '@copilotkit/react-core/v2';
```

- [ ] **Step 2: Replace the rendered chat shell**

Replace:

```tsx
return <CopilotPopup labels={REVIEW_COPILOT_LABELS} defaultOpen={false} width={420} />;
```

With:

```tsx
return <CopilotSidebar labels={REVIEW_COPILOT_LABELS} defaultOpen={true} position="right" width={420} />;
```

- [ ] **Step 3: Run frontend validation**

Run:

```bash
bun run --cwd frontend build
```

Expected: build succeeds without TypeScript or bundling errors.

- [ ] **Step 4: Visual smoke check**

Run the existing dev workflow if needed and confirm:

- Review Copilot appears as a right-side sidebar.
- The sidebar starts open.
- The sidebar toggle collapses and reopens it.
- Existing Review Copilot labels and disclaimer remain visible.

- [ ] **Step 5: Commit only this task if requested**

```bash
git add frontend/src/copilot/ReviewCopilot.tsx docs/superpowers/specs/2026-06-27-review-copilot-sidebar-design.md docs/superpowers/plans/2026-06-27-review-copilot-sidebar.md
git commit -m "feat: show review copilot as sidebar"
```

## Self-Review

- Spec coverage: The single task covers the requested right-side collapsible sidebar.
- Placeholder scan: No TBD/TODO placeholders remain.
- Type consistency: Uses exported CopilotKit v2 names confirmed from installed package declarations.
