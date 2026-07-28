# Current Branch Cleanup Design

**Date:** 2026-07-29

## Purpose

Reduce the current feature branch to product code, durable operating guidance,
and reproducible tests before integrating it directly into `main`. Preserve the
existing 102-commit branch history and do not broaden cleanup into code already
present on `main`.

## Scope

The cleanup covers `main..codex/phase-1-toss-account-observation` and the current
working tree. It removes only artifacts whose replacement or lack of runtime
use is demonstrable. Normalized Toss snapshots remain the only source of
holdings and account facts, and local portfolio state is not rewritten.

## Retained Product Boundaries

- Keep the four inspection statuses exactly: `OK`, `Watch`, `Review`, `Action`.
- Keep `core`, `satellite`, and `experiment` as first-class policy layers.
- Keep inspection output free of order side, quantity, price, execution, or
  automatic trading fields.
- Keep active policy artifacts:
  `2026-07-23-pattern-b-policy-draft.md`,
  `2026-07-23-pattern-b-policy-draft.json`, and
  `2026-07-24-neutral-dynamic-policy.json`.
- Keep operational documentation such as `README.md`, `AGENTS.md`, local IPS
  skills, and `research/qlib_validation/README.md`.

## Deletion Set

Remove the retired AgentMemory skills, its lockfile, and tracked demo/index
state. ACE collaboration memory replaces that workflow. Remove fulfilled
Superpowers plans and design documents other than the retained policy artifacts.
Remove the empty root npm manifest and root `node_modules`, the temporary Japan
CSV inspection script, the unused frontend HTML mockup, and the empty accidental
`data/portfolio.db` file. Add narrow ignore rules for those generated paths.

Remove the manual instrument-profile store and its tests. Instrument layer
classification now comes only from the active IPS policy's
`instruments[].layer` mapping. Do not recreate retired thesis, overlap, burden,
holdability, or ETF-substitution state.

## Logical Commit Boundaries

1. Replace retired AgentMemory assets with the bounded ACE hook, memory skill,
   tests, and repository guidance.
2. Make active IPS policy data the sole source of instrument layers and remove
   the manual profile store.
3. Align overview performance facts with current holding cost basis while
   keeping TWR and external-flow boundaries separate.
4. Complete stored dynamic-allocation evidence and API exposure.
5. Complete causal Qlib forecast validation and its isolated tests.
6. Add local frontend filtering and sorting without changing backend result
   meaning or queue order.
7. Remove fulfilled documentation and temporary/generated artifacts, update
   stale README references, and add narrow ignore rules.

Files shared by several changes may be staged by patch so every commit remains
coherent. Existing commits are not squashed or rewritten.

## Verification

Run Ruff over the repository. Run application tests outside the isolated Qlib
suite, then run Qlib tests with `research/qlib_validation` as the uv project.
Run frontend Bun tests, TypeScript type checking, a production frontend build,
`git diff --check`, and targeted reference searches for removed profile and
AgentMemory paths. A final adversarial review must challenge guardrail
regressions, stale references, hidden generated state, and incomplete commit
boundaries.

## Integration

After all checks and the final review pass, switch to local `main` and use a
fast-forward-only merge. Push `main` directly to `origin`. There is no GitHub PR
for the branch, so PR closure is a verified no-op. Delete the local and remote
`codex/phase-1-toss-account-observation` branches only after the pushed `main`
contains the final commit.

## Failure Handling

Stop before integration if tests fail, if `main` is no longer an ancestor, if a
remote update makes the push non-fast-forward, or if a removal still has a live
reference. Preserve user portfolio data and report the exact blocker rather
than guessing or forcing Git history.
