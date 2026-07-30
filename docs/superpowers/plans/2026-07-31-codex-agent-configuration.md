# Codex Agent Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Terra/medium the project default and retain only IPS Pilot-specific read-only explorer and reviewer subagents.

**Architecture:** Project defaults remain in `.codex/config.toml`; custom agent files layer focused model, reasoning, sandbox, and instructions over those defaults. The explorer gathers code-path evidence, while the reviewer performs high-effort guardrail review without editing state.

**Tech Stack:** Codex TOML configuration; Python standard-library `tomllib` for validation.

---

### Task 1: Set project defaults and prune generic roles

**Files:**
- Modify: `.codex/config.toml`
- Delete: `.codex/agents/routine.toml`

- [x] **Step 1: Replace root model defaults and agent concurrency in `.codex/config.toml`**

```toml
model = "gpt-5.6-terra"
model_reasoning_effort = "medium"

[agents]
enabled = true
max_concurrent_threads_per_session = 2
default_subagent_model = "gpt-5.6-terra"
default_subagent_reasoning_effort = "medium"
```

- [x] **Step 2: Delete `.codex/agents/routine.toml`**

Run: `rm .codex/agents/routine.toml`

Expected: only `explorer.toml` and `reviewer.toml` remain in `.codex/agents/`.

### Task 2: Tune the retained agents for IPS Pilot work

**Files:**
- Modify: `.codex/agents/explorer.toml`
- Modify: `.codex/agents/reviewer.toml`

- [x] **Step 1: Update explorer instructions**

```toml
developer_instructions = """
Trace backend IPS evaluation, Toss snapshot normalization, API/CLI contracts, and
React workbench data flow. Return concise evidence with file paths and symbols.
Never edit files or persistent state; identify uncertainty instead of inferring
unobserved portfolio facts.
"""
```

- [x] **Step 2: Update reviewer instructions**

```toml
developer_instructions = """
Perform adversarial, read-only review of IPS inspection changes. Check that
OK/Watch/Review/Action vocabulary, layer-first evaluation, Toss-only holdings
facts, and the ban on order or trade recommendations remain intact. Report
regressions and missing verification with concrete file and symbol evidence.
Do not edit files or persistent state.
"""
```

### Task 3: Validate the project configuration

**Files:**
- Test: `.codex/config.toml`
- Test: `.codex/agents/explorer.toml`
- Test: `.codex/agents/reviewer.toml`

- [x] **Step 1: Parse TOML and assert the resolved project configuration**

Run:

```bash
.venv/bin/python -c 'import tomllib; from pathlib import Path; c = tomllib.loads(Path(".codex/config.toml").read_text()); assert (c["model"], c["model_reasoning_effort"]) == ("gpt-5.6-terra", "medium"); assert c["agents"]["max_concurrent_threads_per_session"] == 2; assert [p.name for p in sorted(Path(".codex/agents").glob("*.toml"))] == ["explorer.toml", "reviewer.toml"]; [tomllib.loads(p.read_text()) for p in Path(".codex/agents").glob("*.toml")]; print("Project Codex configuration is valid.")'
```

Expected: `Project Codex configuration is valid.`

- [x] **Step 2: Inspect the configuration diff**

Run: `git diff -- .codex/config.toml .codex/agents`

Expected: Terra/medium defaults, a two-thread cap, tailored explorer/reviewer guidance, and removal of `routine.toml`.
