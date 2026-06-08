---
name: codex-cli
description: Use when the user asks about using Codex from a terminal, Codex CLI commands, non-interactive codex exec runs, reviews, sessions, plugins, MCP servers, sandboxing, approvals, or local Codex diagnostics.
---

# Codex CLI

## Overview

Use this skill to help users operate Codex from the command line. Prefer verifying the installed CLI with `codex --help` or subcommand help before giving exact flags, because CLI surfaces can change.

## First Checks

- Confirm the CLI exists: `codex --version`
- Inspect top-level commands: `codex --help`
- Inspect a subcommand: `codex <command> --help`
- Diagnose local setup: `codex doctor`
- Update when the user asks for current CLI behavior: `codex update`

If a command fails with path, auth, config, or sandbox symptoms, run `codex doctor` before guessing.

## Common Commands

| Task | Command pattern |
| --- | --- |
| Start interactive Codex | `codex` |
| Start with a prompt | `codex "make the tests pass"` |
| Run in a specific repo | `codex -C /path/to/repo "task"` |
| Use a model | `codex -m <model> "task"` |
| Run non-interactively | `codex exec "task"` |
| Read prompt from stdin | `codex exec -` |
| Review code | `codex review` or `codex exec review` |
| Resume a session | `codex resume` or `codex exec resume --last` |
| Fork a session | `codex fork` |
| Manage auth | `codex login`, `codex logout` |
| Manage plugins | `codex plugin --help` |
| Manage MCP servers | `codex mcp --help` |
| Generate completions | `codex completion --help` |

## Non-Interactive Runs

Use `codex exec` for scripts, CI-style checks, repeatable prompts, or when the user wants a one-shot result.

Useful flags:

- `--json` for JSONL event output.
- `-o, --output-last-message <FILE>` to save the final response.
- `--output-schema <FILE>` when the final response must match a JSON Schema.
- `--ephemeral` when session files should not be persisted.
- `--ignore-user-config` to avoid loading `$CODEX_HOME/config.toml` while keeping auth.
- `--skip-git-repo-check` only when intentionally running outside a Git repo.

Example:

```sh
codex exec -C /path/to/repo -o /tmp/codex-result.txt "summarize the failing tests and suggest a fix"
```

## Sandboxing And Approvals

Default to the least permissive mode that can complete the task.

- `-s read-only`: investigation and explanation only.
- `-s workspace-write`: edits inside the workspace.
- `-s danger-full-access`: only when the user explicitly accepts broad local access.
- `-a untrusted`: ask for approval for untrusted commands.
- `-a on-request`: let Codex decide when approval is needed.
- `-a never`: automation mode; failures return directly to the model.

Avoid recommending `--dangerously-bypass-approvals-and-sandbox` unless the user is inside a separate external sandbox and explicitly asks for a fully unattended run.

## Config Overrides

Use `-c key=value` for temporary config changes. Values are parsed as TOML when possible.

Examples:

```sh
codex -c model='"o3"' "explain this repository"
codex -c shell_environment_policy.inherit=all "run the local checks"
codex -c 'sandbox_permissions=["disk-full-read-access"]' "inspect dependencies"
```

Prefer command-line overrides for one-off runs; prefer `$CODEX_HOME/config.toml` for stable defaults.

## Working With Files And Images

- Use `-C <DIR>` to set the working root instead of relying on the caller's current directory.
- Use `--add-dir <DIR>` when another directory must be writable alongside the primary workspace.
- Attach images with `-i <FILE>`; repeat the flag for multiple images.

## Troubleshooting

- `codex` not found: check installation, shell PATH, and terminal restart.
- Auth failures: run `codex login`, then `codex doctor`.
- Unexpected config behavior: inspect `$CODEX_HOME/config.toml`; retry with `--strict-config` to catch unknown fields.
- Sandbox or approval confusion: rerun with explicit `-s <mode>` and `-a <policy>`.
- Plugin or MCP issues: use `codex plugin --help`, `codex mcp --help`, and then inspect the specific subcommand help.

## Response Style

When helping a user, include the exact command to run, briefly explain why the flags are chosen, and mention any safety tradeoff if sandbox or approval settings are relaxed.
