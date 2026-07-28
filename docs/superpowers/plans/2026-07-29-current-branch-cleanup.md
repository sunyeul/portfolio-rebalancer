# Current Branch Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove demonstrably obsolete branch artifacts, commit every retained working-tree change by product responsibility, and fast-forward the verified result directly into `main`.

**Architecture:** Preserve the branch's existing 102 commits and partition only the working tree into ACE infrastructure, IPS backend contracts, isolated Qlib research, frontend presentation, and final documentation/artifact cleanup. Keep active policy JSON/Markdown and all normalized Toss data contracts; remove generated or retired paths only after reference checks.

**Tech Stack:** Python 3.12, FastAPI, Typer, SQLite, uv, pytest, Ruff, React 19, TypeScript, Vite, Bun, Git, GitHub CLI.

---

## File Map

- ACE workflow: `.agent/`, `.codex/config.toml`, `.codex/hooks.json`,
  `.codex/hooks/ace_session_start.py`, `AGENTS.md`,
  `tests/test_ace_session_start.py`.
- Retired AgentMemory assets: `.agents/skills/agentmemory-*`,
  `.agents/skills/{commit-context,commit-history,forget,handoff,recall,recap,remember,session-history,write-agentmemory-skill}`,
  `skills-lock.json`, `data/state_store.db/`, `data/stream_store/`.
- IPS backend contract: `api/app.py`, `services/account_performance.py`,
  `services/account_projection.py`, `services/action_contract.py`,
  `services/inspection_engine.py`, `services/inspection_service.py`,
  `storage/evaluation_store.py`, `storage/market_store.py`,
  `storage/performance_store.py`, `storage/policy_store.py`, and their tests.
- Retired manual profile path: `storage/instrument_profile_store.py` and
  `tests/test_instrument_profile_store.py`.
- Backend guidance: `README.md`, `Taskfile.yml`, and the three modified local
  IPS skill files under `.agents/skills/ips-*`.
- Qlib forecast: `research/qlib_validation/` and `tests/research/`.
- Frontend tables: `frontend/src/App.tsx`, `frontend/src/lib/api.ts`,
  `frontend/src/lib/presentation.ts`, `frontend/src/lib/tableControls.ts`,
  `frontend/src/styles.css`, `frontend/tests/`.
- Final cleanup: `.gitignore`, fulfilled `docs/superpowers/plans/*.md` and
  `docs/superpowers/specs/*.md`, except the active Pattern B Markdown; preserve
  all three approved policy artifacts.

### Task 1: Record and Commit the Approved Design

**Files:**
- Create: `docs/superpowers/specs/2026-07-29-current-branch-cleanup-design.md`

- [x] **Step 1: Write the approved cleanup design**

Record scope, retained guardrails, exact deletion classes, commit boundaries,
verification, and fast-forward integration behavior.

- [x] **Step 2: Scan for placeholders and whitespace errors**

Run:

```bash
rtk rg -n '(T[B]D|T[O]DO|implement l[a]ter|fill i[n])' docs/superpowers/specs/2026-07-29-current-branch-cleanup-design.md
rtk git diff --check -- docs/superpowers/specs/2026-07-29-current-branch-cleanup-design.md
```

Expected: no matches and no diff errors.

- [x] **Step 3: Commit the design**

```bash
rtk git add docs/superpowers/specs/2026-07-29-current-branch-cleanup-design.md
rtk git commit -m "docs: design current branch cleanup"
```

Expected: one documentation-only commit.

### Task 2: Replace Retired AgentMemory State with ACE

**Files:**
- Create: `.agent/playbooks/collaboration-lessons.md`
- Create: `.agent/skills/ace-collaboration-memory/SKILL.md`
- Create: `.codex/hooks.json`
- Create: `.codex/hooks/ace_session_start.py`
- Modify: `.codex/config.toml`
- Modify: `AGENTS.md`
- Create: `tests/test_ace_session_start.py`
- Delete: retired AgentMemory skill directories listed in the File Map
- Delete: `skills-lock.json`
- Delete: `data/state_store.db/`
- Delete: `data/stream_store/`

- [ ] **Step 1: Verify the ACE hook independently**

Run:

```bash
rtk uv run pytest -q tests/test_ace_session_start.py
```

Expected: 4 tests pass; the hook remains read-only and fail-open.

- [ ] **Step 2: Verify retired AgentMemory files have no live references**

Run:

```bash
rtk rg -n --hidden -g '!node_modules/**' -g '!.git/**' '(agentmemory|state_store|stream_store|skills-lock)' .
```

Expected: only the files scheduled for deletion or historical cleanup docs are
reported; no runtime product module imports them.

- [ ] **Step 3: Remove the tracked AgentMemory stores and stage only ACE scope**

Delete the exact tracked state-store directories, then stage the ACE files,
retired skill deletions, lockfile deletion, and ACE test. Do not stage modified
IPS product skills in this commit.

- [ ] **Step 4: Check the staged boundary and commit**

Run:

```bash
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "chore: replace agentmemory with ACE collaboration memory"
```

Expected: only ACE infrastructure and retired AgentMemory assets are committed.

### Task 3: Consolidate IPS Inspection Sources

**Files:**
- Modify: `api/app.py`
- Modify: `README.md`
- Modify: `Taskfile.yml`
- Modify: `.agents/skills/ips-judgment-filter/SKILL.md`
- Modify: `.agents/skills/ips-pilot-cli-review/SKILL.md`
- Modify: `.agents/skills/ips-pilot-frontend-workbench/SKILL.md`
- Modify: `services/account_performance.py`
- Modify: `services/account_projection.py`
- Modify: `services/action_contract.py`
- Modify: `services/inspection_engine.py`
- Modify: `services/inspection_service.py`
- Modify: `storage/evaluation_store.py`
- Delete: `storage/instrument_profile_store.py`
- Modify: `storage/market_store.py`
- Modify: `storage/performance_store.py`
- Modify: `storage/policy_store.py`
- Modify/Delete: corresponding root `tests/test_*.py`

- [ ] **Step 1: Confirm the retired profile vocabulary is absent from runtime code**

Run:

```bash
rtk rg -n '(instrument_profile|ips_instrument_profiles|thesis_status|overlap_status|management_burden_status|holdability_status|etf_substitution_status)' api services storage cli.py
```

Expected: no runtime matches.

- [ ] **Step 2: Run the complete application suite outside isolated Qlib tests**

Run:

```bash
rtk uv run pytest -q tests --ignore=tests/research
rtk uv run ruff check .
```

Expected: 190 application tests pass and Ruff reports no violations.

- [ ] **Step 3: Stage the backend contract as one unit**

Stage the exact files listed for this task, including profile-store deletions and
the backend-aligned README, Taskfile, and IPS skill changes. Exclude all Qlib,
frontend, ACE, and fulfilled-plan files.

- [ ] **Step 4: Check the staged boundary and commit**

```bash
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "refactor: consolidate IPS inspection sources"
```

Expected: one backend contract commit with its own tests and documentation.

### Task 4: Commit Causal Qlib Forecast Validation

**Files:**
- Modify: `research/qlib_validation/README.md`
- Modify: `research/qlib_validation/artifacts.py`
- Modify: `research/qlib_validation/cli.py`
- Modify: `research/qlib_validation/contracts.py`
- Create: `research/qlib_validation/forecast.py`
- Modify: `research/qlib_validation/report.py`
- Modify: `research/qlib_validation/source.py`
- Modify: `tests/research/test_qlib_artifacts.py`
- Modify: `tests/research/test_qlib_cli.py`
- Create: `tests/research/test_qlib_forecast.py`
- Modify: `tests/research/test_qlib_report.py`
- Modify: `tests/research/test_qlib_source.py`

- [ ] **Step 1: Run the isolated Qlib suite**

```bash
rtk uv run --project research/qlib_validation pytest -q tests/research
```

Expected: 48 tests pass. Deprecation warnings from Qlib are non-blocking.

- [ ] **Step 2: Check research code style**

```bash
rtk uv run --project research/qlib_validation ruff check research/qlib_validation tests/research
```

Expected: no Ruff violations.

- [ ] **Step 3: Stage, inspect, and commit the research unit**

```bash
rtk git add research/qlib_validation tests/research
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "feat: add causal Qlib forecast validation"
```

Expected: no application or frontend files are staged.

### Task 5: Commit Frontend Inspection Table Controls

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/presentation.ts`
- Create: `frontend/src/lib/tableControls.ts`
- Modify: `frontend/src/styles.css`
- Create: `frontend/tests/tableControls.test.ts`
- Modify: `frontend/tests/presentation.test.ts`

- [ ] **Step 1: Run unit tests and type checking**

```bash
rtk bun test
rtk bun run typecheck
```

Run these commands from `frontend/`. Expected: 11 tests pass and TypeScript
reports no errors.

- [ ] **Step 2: Build the production bundle**

```bash
rtk bun run build
```

Run this command from `frontend/`. Expected: Vite produces `frontend/dist`
successfully.

- [ ] **Step 3: Stage, inspect, and commit the frontend unit**

```bash
rtk git add frontend/src frontend/tests
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "feat(frontend): add inspection table controls"
```

Expected: the untracked HTML mockup is not staged.

### Task 6: Remove Fulfilled Documents and Generated Artifacts

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Delete: fulfilled Markdown under `docs/superpowers/plans/` and
  `docs/superpowers/specs/`, including this plan and the cleanup design
- Preserve: `docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.md`
- Preserve: `docs/superpowers/specs/2026-07-23-pattern-b-policy-draft.json`
- Preserve: `docs/superpowers/specs/2026-07-24-neutral-dynamic-policy.json`
- Delete: root `package.json`, root `node_modules/`, `.codex_tmp/`,
  `frontend/portfolio-operating-mockup.html`, and empty `data/portfolio.db`

- [ ] **Step 1: Add narrow ignore rules**

Append these entries to `.gitignore` in the existing generated-artifact groups:

```gitignore
/node_modules/
/.codex_tmp/
data/*.db
data/stream_store/
```

- [ ] **Step 2: Remove exact generated and fulfilled targets**

Validate each target is inside the repository. Confirm `data/portfolio.db` is
zero bytes before deleting it. Remove all fulfilled Markdown while preserving
the three policy artifacts exactly.

- [ ] **Step 3: Repair README references and verification commands**

Remove the deleted roadmap link and describe Phase 6 as not implemented. Replace
the failing monolithic test command with:

```bash
uv run pytest -q tests --ignore=tests/research
uv run --project research/qlib_validation pytest -q tests/research
```

- [ ] **Step 4: Prove no stale references remain**

```bash
rtk rg -n -g '!node_modules/**' '(cash-account-observability-roadmap|agentmemory|state_store|stream_store|portfolio-operating-mockup|japan_csv_inspect)' .
rtk git diff --check
rtk git status --short
```

Expected: no live references; only intentional policy Markdown/JSON remain under
`docs/superpowers`.

- [ ] **Step 5: Stage and commit final cleanup**

```bash
rtk git add -A .gitignore README.md docs
rtk git diff --cached --check
rtk git diff --cached --stat
rtk git commit -m "docs: remove fulfilled development artifacts"
```

Expected: generated untracked content is gone and fulfilled tracked docs are
deleted. The cleanup design and plan disappear from the final tree by design.

### Task 7: Full Verification and Adversarial Review

**Files:**
- Review only; modify any affected source or test only if a verified issue is
  found.

- [ ] **Step 1: Run the full split verification matrix**

```bash
rtk uv run ruff check .
rtk uv run pytest -q tests --ignore=tests/research
rtk uv run --project research/qlib_validation pytest -q tests/research
rtk bun test
rtk bun run typecheck
rtk bun run build
```

Run the three Bun commands from `frontend/`. Expected: all commands pass.

- [ ] **Step 2: Verify Git and product guardrails**

```bash
rtk git diff --check main...HEAD
rtk git status --short
rtk git diff main...HEAD -- api services frontend/src
rtk git merge-base --is-ancestor main HEAD
```

Expected: clean working tree and local `main` remains an ancestor. Inspect the
diff for any newly introduced `buy`, `sell`, `execute`, order-size, quantity,
price, or execution field; none may grant trading authority.

- [ ] **Step 3: Perform mandatory pre-merge code review**

Review `main..HEAD` against this plan, emphasizing stale references, IPS status
or source-of-truth regressions, destructive omissions, and verification gaps.
Fix all Critical or Important findings and rerun affected checks.

### Task 8: Fast-Forward Main and Delete the Work Branch

**Files:**
- Git refs only.

- [ ] **Step 1: Refresh remote facts without changing the working tree**

```bash
rtk git fetch origin
rtk gh pr list --state open --json number,headRefName,baseRefName,url
rtk git rev-list --left-right --count origin/main...HEAD
```

Expected: no open PR targets the branch and `origin/main` has no unexpected
commits that prevent a safe fast-forward.

- [ ] **Step 2: Fast-forward local main**

```bash
rtk git switch main
rtk git merge --ff-only codex/phase-1-toss-account-observation
```

Expected: no merge commit or conflict.

- [ ] **Step 3: Push main and verify the remote commit**

```bash
rtk git push origin main
rtk git ls-remote --heads origin main
```

Expected: remote `main` points to local `main` HEAD.

- [ ] **Step 4: Delete the merged local and remote branch**

```bash
rtk git branch -d codex/phase-1-toss-account-observation
rtk git push origin --delete codex/phase-1-toss-account-observation
```

Expected: only `main` remains locally for this work and the remote feature ref
is absent. If the remote branch is already absent, report that verified no-op.
