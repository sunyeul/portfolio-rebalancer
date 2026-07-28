---
name: ace-collaboration-memory
description: Maintain the IPS Pilot project's shared collaboration lessons and local user preferences without weakening higher-priority instructions.
---

# ACE Collaboration Memory

Use this workflow when starting work, changing scope or goals, receiving an explicit collaboration preference or correction, encountering an unexpected reusable failure, or finishing work that produced a memory candidate.

## Start and recovery protocol

1. Identify the task scope, task type, and tools that will be used.
2. Declare these four short fields together:
   - `PURPOSE`: the durable user value being protected.
   - `GOAL`: the verifiable result of this task.
   - `ALIGNMENT`: why the result supports the purpose.
   - `WORKING LOG`: only decisions, evidence, open questions, and next action.
3. Read the repository's `AGENTS.md` and this skill.
4. Search `.agent/playbooks/collaboration-lessons.md` for only the `Active` lessons relevant to the task.
5. Read `.serena/memories/local/user_preferences.md` and apply only complete `Confirmed` entries relevant to the task. If the file is missing or invalid, continue with no preferences and report that fact briefly.

Repeat the four fields immediately after any goal or scope change and before claiming completion. A trusted session hook may restore the fixed retrieval instruction and confirmed preferences, but it never replaces this protocol.

## Precedence and classification

Apply rules in this order:

1. Current system, developer, and user instructions.
2. Existing repository contracts, including `AGENTS.md`.
3. Relevant `Active` shared lessons.
4. Relevant `Confirmed` local preferences.

Classify each candidate observation before storing it:

| Observation | Destination | Status | Treatment |
| --- | --- | --- | --- |
| The user explicitly requests or corrects a collaboration preference | Local preference file | `Confirmed` | ADD or UPDATE now and use it in this task |
| A reusable correction or failure cause applies to future contributors | Shared lesson file | `Active` | Require a concrete trigger, executable rule, and evidence |
| A preference inferred from behavior but not stated | Local preference file | `Pending` | Do not apply or inject until the user confirms it |
| Expected TDD failure, one-off detail, temporary environment error, secret, transcript, or large command output | Nowhere | — | Keep it only in the current task |

Do not store secrets, tokens, credentials, raw conversation text, or large command output in either file.

## Memory updates

Only three update operations are allowed:

- `ADD`: create a new entry with a never-before-used stable ID.
- `UPDATE`: change the current entry without changing its ID or erasing its history.
- `MARK`: change lifecycle status, such as marking an obsolete lesson `Superseded` or a rejected preference `Rejected`.

Never rewrite a file as a global summary, delete an entry, or reuse an old ID. Shared lessons are Git-tracked; personal preferences stay in the ignored Serena-local path. A shared lesson needs all of these fields: `lesson ID`, `title`, `status`, `scope`, `trigger`, `rule`, `evidence`, `helpful`, and `harmful`. A local preference needs: `pref ID`, `title`, `status`, `context`, `prefer`, `avoid`, `source`, `helpful`, and `harmful`.

When a recurrence can be prevented from observable input, first find the nearest existing validator, test, or reporter. Add a failing reproduction, a regression check owned by that validator, and a bypass control before treating the lesson as prevention. A lesson sentence alone is not proof that recurrence is prevented. If human judgment is irreducible, keep the lesson narrow and state the judgment boundary in `trigger` or `rule`.

## Completion check

Before reporting completion:

- Re-declare the four fields and confirm the result serves `PURPOSE`.
- Confirm no higher-priority instruction was overridden by memory.
- Confirm shared lessons are tracked and local preferences are ignored.
- Record only a concrete reusable lesson or an explicit preference; otherwise record nothing.
- Report missing or untrusted hook state without inventing memory contents.
