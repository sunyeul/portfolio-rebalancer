# Remove Review Copilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Remove CopilotKit, the Review Copilot runtime, and A2UI while keeping the ordinary portfolio, snapshot, analysis, evaluation, Review Queue, and journal-data views operational.

**Architecture:** The React workbench will render the API evaluation response directly. A small typed helper will group existing Review Queue records by their v2 inspection status; no browser state will accept agent patches and no request will depend on a sidecar runtime. The Python API and CLI contracts remain unchanged.

**Tech Stack:** React 19, TypeScript, Bun/Vite, FastAPI, pytest.

---

## File map

- Create: frontend/src/lib/reviewQueue.ts — native Review Queue grouping.
- Create: frontend/tests/reviewQueue.test.ts — grouping contract test.
- Modify: frontend/src/App.tsx — direct result rendering and no Copilot wrapper.
- Modify: frontend/package.json, frontend/bun.lock, frontend/vite.config.ts, frontend/src/styles/app.css, Taskfile.yml, .env.example, README.md, .gitignore.
- Delete: agent-runtime/, frontend/src/copilot/, frontend/src/a2ui/, frontend/tests/a2ui.validation.test.ts.
- Delete: docs/superpowers/specs/2026-06-27-review-copilot-sidebar-design.md and docs/superpowers/plans/2026-06-27-review-copilot-sidebar.md.

### Task 1: Add the native Review Queue grouping contract

**Files:**
- Create: frontend/tests/reviewQueue.test.ts
- Create: frontend/src/lib/reviewQueue.ts

- [ ] **Step 1: Write the failing test**

    import { describe, expect, test } from 'bun:test';

    import type { ReviewItem } from '../src/lib/api';
    import { groupReviewQueueItems } from '../src/lib/reviewQueue';

    function item(status: ReviewItem['status'], name: string): ReviewItem {
      return {
        level: 'asset',
        name,
        parent_layer: 'core',
        status,
        triggered_by: [],
        metrics_snapshot: {},
        thesis: null,
        counter_scenario: null,
        suggested_next_step: '다음 정기 리뷰에서 다시 확인'
      };
    }

    describe('groupReviewQueueItems', () => {
      test('uses Action, Review, Watch order and preserves each group order', () => {
        const groups = groupReviewQueueItems([
          item('Watch', 'GLD'),
          item('Action', 'QQQ'),
          item('Review', 'SMH'),
          item('Action', 'VOO')
        ]);

        expect(groups.map((group) => [group.status, group.items.map((entry) => entry.name)])).toEqual([
          ['Action', ['QQQ', 'VOO']],
          ['Review', ['SMH']],
          ['Watch', ['GLD']]
        ]);
      });
    });

- [ ] **Step 2: Run the test to verify it fails**

Run: cd frontend && bun test tests/reviewQueue.test.ts

Expected: FAIL because src/lib/reviewQueue does not exist.

- [ ] **Step 3: Implement the minimal helper**

    import type { ReviewItem } from './api';

    export const reviewQueueStatusOrder = ['Action', 'Review', 'Watch'] as const;

    export type ReviewQueueStatus = (typeof reviewQueueStatusOrder)[number];

    const reviewTriggerExplanations: Record<string, string> = {
      risk_contribution: '위험 기여도가 커서 이 부담이 의도된 것인지 확인합니다.',
      risk_contribution_high: '위험 기여도가 커서 이 부담이 의도된 것인지 확인합니다.',
      target_gap_outside_tolerance: '현재 비중과 IPS 목표 범위의 차이를 확인합니다.',
      thesis_watch: '기록된 투자 논리가 관찰 또는 재검토 상태인지 확인합니다.',
      thesis_broken: '투자 논리가 훼손됐는지 근거를 다시 확인합니다.',
      volatility_exceeded: '변동성이 허용 기준을 넘었는지 확인합니다.',
      mdd_exceeded: '낙폭이 점검 기준을 넘었는지 확인합니다.',
      efficiency_below_threshold: '성과 대비 위험 효율이 낮아졌는지 확인합니다.',
      high_burden: '관리 부담이나 계층 내 부담이 커졌는지 확인합니다.',
      max_weight_exceeded: '항목 또는 계층 비중이 상한을 넘었는지 확인합니다.'
    };

    export function describeReviewTrigger(code: string) {
      return reviewTriggerExplanations[code] ?? '기록된 점검 신호와 데이터 근거를 확인합니다.';
    }

    export function groupReviewQueueItems(queue: ReviewItem[]) {
      return reviewQueueStatusOrder.map((status) => ({
        status,
        items: queue.filter((item) => item.status === status)
      }));
    }

- [ ] **Step 4: Run the focused test**

Run: cd frontend && bun test tests/reviewQueue.test.ts

Expected: PASS.

- [ ] **Step 5: Commit**

    git add frontend/src/lib/reviewQueue.ts frontend/tests/reviewQueue.test.ts
    git commit -m "test: cover native review queue grouping"

### Task 2: Replace generated evaluation surfaces with native workbench output

**Files:**
- Modify: frontend/src/App.tsx:47-58, 763-1127, 1815-1824, 2124-2319

- [ ] **Step 1: Remove the agent-only imports and add the grouping helper**

Delete the imports from ./copilot and ./a2ui. Add this import alongside the existing API/schema imports:

    import { groupReviewQueueItems } from './lib/reviewQueue';

Keep BarChart3 and ClipboardList because native dashboard and Review Queue still use them.

- [ ] **Step 2: Remove generated graphs and simplify focus state**

Delete EvaluationGraphs entirely. In EvaluationResults, keep only hovered and selected ticker state, reset those two values on a new evaluation, and render the base tables and data sections in this order:

    <EvaluationRunHeader evaluationRun={evaluationRun} />
    <LayerDashboard rows={evaluation.layer_evaluations} />
    <AssetEvaluationTable
      focusedTicker={focusedTicker}
      rows={evaluation.asset_evaluations}
      onTickerFocus={handleTickerFocus}
      onTickerSelect={handleTickerSelect}
    />
    <ReviewQueue evaluation={evaluation} />
    <JournalDraft rows={evaluation.journal_draft} />

Remove focusedLayer from LayerDashboard and AssetEvaluationTable. Layer rows use the ordinary border class. Asset rows are dimmed only when focusedTicker is not null and does not match the row ticker.

- [ ] **Step 3: Replace ReviewQueue with an API-data-only section**

Use groupReviewQueueItems(evaluation.review_queue), then render Action, Review, and Watch groups. Show the evaluation period and item count, and present each item with its name, StatusBadge, layer name, raw trigger code plus a fixed human-readable explanation from describeReviewTrigger, and suggested_next_step. The Action heading must state that exceptional intervention is for human review, not permission to trade. Add the following fixed notice above the groups:

    이 목록은 IPS 점검 신호입니다. 매매 지시나 주문 수량 산정이 아닙니다.

Do not render agent explanations, generated summaries, action-disposition selects, generated UI fallback text, or any browser request outside the existing FastAPI API calls.

- [ ] **Step 4: Keep only the API journal data**

Delete this block from JournalDraft:

    <div className="mt-3">
      <GeneratedSurfaceHost target="journal_draft" />
    </div>

- [ ] **Step 5: Remove root providers and Copilot settings**

Delete copilotLayerBenchmarks and copilotSettings, but retain normalizedLayerBenchmarks because normal analysis/evaluation requests use it. Remove ReviewCopilotHost, GenerativeUiProvider, and ReviewCopilot from the root render tree; main and all existing modals remain direct children of the normal app render.

- [ ] **Step 6: Typecheck**

Run: cd frontend && bun run typecheck

Expected: PASS with no deleted-symbol diagnostics.

- [ ] **Step 7: Commit**

    git add frontend/src/App.tsx frontend/src/lib/reviewQueue.ts frontend/tests/reviewQueue.test.ts
    git commit -m "refactor: render workbench without generated surfaces"

### Task 3: Remove the embedded agent implementation and packages

**Files:**
- Modify: frontend/package.json, frontend/bun.lock, frontend/vite.config.ts, frontend/src/styles/app.css
- Delete: frontend/src/copilot/ReviewCopilot.tsx, frontend/src/a2ui/, frontend/tests/a2ui.validation.test.ts, agent-runtime/

- [ ] **Step 1: Remove CopilotKit and update its lockfile**

Run: cd frontend && bun remove @copilotkit/react-core

Expected: package.json and bun.lock contain no @copilotkit entries.

- [ ] **Step 2: Remove the sidecar route and layout override**

Replace the Vite proxy with:

    proxy: {
      '/api': 'http://localhost:8000'
    }

Reduce the body style to:

    body {
      margin: 0;
      min-width: 320px;
      background: #f5f7fb;
    }

Remove the three agent-runtime ignore entries from .gitignore. Preserve the user's local .serena/ and .tokensave/ ignore entries.

- [ ] **Step 3: Delete the feature implementation**

    git rm -r frontend/src/copilot frontend/src/a2ui frontend/tests/a2ui.validation.test.ts agent-runtime

Expected: source control stages every tracked feature file. Do not delete ignored or untracked local files under agent-runtime without separately inspecting and approving those exact paths; source removal does not require deleting installed dependencies.

- [ ] **Step 4: Build**

Run: cd frontend && bun run build

Expected: PASS with no CopilotKit chunk or /copilotkit route.

- [ ] **Step 5: Commit**

    git add frontend/package.json frontend/bun.lock frontend/vite.config.ts frontend/src/styles/app.css
    git add -u frontend agent-runtime
    git commit -m "chore: remove review copilot runtime"

### Task 4: Remove obsolete configuration and documentation

**Files:**
- Modify: Taskfile.yml, .env.example, README.md, .gitignore
- Delete: docs/superpowers/specs/2026-06-27-review-copilot-sidebar-design.md, docs/superpowers/plans/2026-06-27-review-copilot-sidebar.md

- [ ] **Step 1: Remove agent-dev from Taskfile.yml**

Replace dev with:

    dev:
      desc: Run API and frontend development servers
      deps:
        - run
        - frontend-dev

- [ ] **Step 2: Remove unused runtime environment variables**

Replace .env.example with:

    # Optional local database override
    # PORTFOLIO_DB_PATH=data/portfolio_rebalancer.sqlite3

- [ ] **Step 3: Rewrite README for the standalone workbench**

- Remove the Review Copilot stack bullet and the complete Review Copilot A2UI section.
- Remove task agent-dev from development setup and common commands.
- State that Vite proxies only /api to http://localhost:8000.
- Remove OPENAI_API_KEY, COPILOT_MODEL, and COPILOT_RUNTIME_PORT setup text.
- Change the task dev description to API + frontend dev servers.
- Retain CLI agent-brief and review-queue examples because they are unchanged inspection commands.

- [ ] **Step 4: Delete obsolete feature documents and commit cleanup**

    git rm docs/superpowers/specs/2026-06-27-review-copilot-sidebar-design.md
    git rm docs/superpowers/plans/2026-06-27-review-copilot-sidebar.md
    git add Taskfile.yml .env.example README.md
    git add -u docs/superpowers
    git commit -m "docs: remove review copilot setup"

### Task 5: Verify the reduced workbench and preserved CLI contract

**Files:**
- Verify only. Do not modify the Python API, CLI, SQLite data, snapshots, or journals.

- [ ] **Step 1: Run frontend checks**

    cd frontend
    bun install --frozen-lockfile
    bun test
    bun run typecheck
    bun run build

Expected: PASS. The native Review Queue test, typecheck, and production build all succeed.

- [ ] **Step 2: Run the relevant CLI test**

Run: uv run pytest tests/test_api_v1.py tests/test_evaluation_units.py tests/test_cli.py -q

Expected: PASS. The single-JSON inspection CLI contract remains unchanged.

- [ ] **Step 3: Search for active feature references**

    rg -n -i 'copilotkit|review copilot|agent-runtime|/copilotkit|a2ui|COPILOT_RUNTIME|COPILOT_MODEL|OPENAI_API_KEY' \
      --glob '!**/node_modules/**' \
      --glob '!docs/superpowers/specs/2026-07-22-remove-review-copilot-design.md' \
      --glob '!docs/superpowers/plans/2026-07-22-remove-review-copilot.md' \
      .
    git diff --check
    git status --short

Expected: no search matches or whitespace errors. Do not stage the unrelated untracked .serena/ directory.

- [ ] **Step 4: Inspect the workbench at desktop and narrow widths**

Start task dev and verify portfolio input, snapshot management, analysis, and evaluation are usable; tables retain local sorting, ticker highlighting, API errors, and pending/disabled states; Review Queue contains only inspection data; journal draft contains only API-returned data; and no sidebar, popup, OpenAI warning, or agent connection attempt appears.

- [ ] **Step 5: Confirm the verification creates no commit-worthy source change**

Run: git status --short

Expected: only user-owned or ignored local state may remain, such as .serena/, .env, installed modules, build output, or the pre-existing AGENTS.md and .gitignore edits. Do not stage or commit any of them; Tasks 1-4 already commit every intended implementation change.
