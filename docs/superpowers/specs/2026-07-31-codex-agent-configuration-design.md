# Codex agent configuration

## Goal

Use Terra with medium reasoning as the project default, while retaining only
the two delegated roles that add project-specific value.

## Configuration

- Set the primary model to `gpt-5.6-terra` and reasoning effort to `medium`.
- Retain a maximum of two concurrent subagent threads.
- Keep `explorer` as a read-only Terra/medium agent for tracing backend,
  snapshot, API, and frontend execution paths with concise source evidence.
- Keep `reviewer` as a read-only Sol/high agent for adversarial review of IPS
  inspection contracts, safety guardrails, regressions, and verification gaps.
- Remove the generic `routine` agent. Repeatable extraction and summaries do
  not justify a dedicated role in this project.

## Boundaries

Both agents inherit the repository's `AGENTS.md` guidance. Neither may mutate
source, policy, snapshots, journals, or persistent state. They report findings
to the parent, which remains responsible for edits and for deciding whether a
subagent is warranted.

## Verification

Parse all project TOML files with Python's `tomllib` and assert the root model,
default subagent values, concurrency limit, and the two expected custom agent
files.
