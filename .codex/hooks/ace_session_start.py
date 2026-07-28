"""Restore bounded ACE context at Codex session start.

This hook is intentionally read-only and fail-open. It does not inspect the
transcript, change memory, call a network service, or stop a Codex session.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

MAX_PREFERENCE_BYTES = 16 * 1024
MAX_OUTPUT_CHARS = 2_000
PREFERENCE_STATUSES = {"Confirmed", "Pending", "Rejected", "Superseded"}
REQUIRED_FIELDS = {
    "title",
    "status",
    "context",
    "prefer",
    "avoid",
    "source",
    "helpful",
    "harmful",
}
ALLOWED_FIELDS = REQUIRED_FIELDS
FENCE_RE = re.compile(r"(?ms)^```[^\n]*\n.*?^```\s*$")
FIELD_RE = re.compile(r"^-\s+([a-z]+):\s*(.*)$")
ALLOWED_EVENTS = {"startup", "resume", "clear", "compact"}

FIXED_INSTRUCTION = "\n".join(
    (
        "ACE session context restoration:",
        "- Read AGENTS.md and .agent/skills/ace-collaboration-memory/SKILL.md.",
        "- Declare PURPOSE, GOAL, ALIGNMENT, and WORKING LOG together.",
        "- Search only Active shared lessons relevant to the current task.",
        "- Apply only complete Confirmed local preferences; never apply Pending, Rejected, or Superseded entries.",
        "- Use precedence: current instructions > repository contract > Active shared lessons > Confirmed personal preferences.",
    )
)


class PreferenceFileError(ValueError):
    """A preference file cannot be safely interpreted as a whole."""


def _project_root(cwd: str | None) -> Path | None:
    if not cwd:
        return None
    try:
        candidate = Path(cwd).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _parse_preferences(path: Path) -> list[dict[str, str]]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise PreferenceFileError("preference file is missing") from exc
    except OSError as exc:
        raise PreferenceFileError("preference file could not be read") from exc

    if len(raw) > MAX_PREFERENCE_BYTES:
        raise PreferenceFileError("preference file exceeds 16 KiB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreferenceFileError("preference file is not UTF-8") from exc

    # Examples in fenced documentation blocks are not preference records.
    text = FENCE_RE.sub("", text)
    matches = list(
        re.finditer(
            r"(?m)^##[ \t]+(pref-[^\s]+)(?:[ \t]+.*)?$",
            text,
        )
    )
    if not matches:
        return []

    entries: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        fields: dict[str, str] = {"id": match.group(1)}
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("<!--") or stripped.endswith("-->"):
                continue
            field_match = FIELD_RE.fullmatch(stripped)
            if not field_match:
                raise PreferenceFileError("preference block contains an invalid line")
            key, value = field_match.groups()
            if key not in ALLOWED_FIELDS or key in fields or not value.strip():
                raise PreferenceFileError("preference block contains an invalid field")
            fields[key] = value.strip()

        if not REQUIRED_FIELDS.issubset(fields):
            raise PreferenceFileError("preference block is missing a required field")
        if fields["status"] not in PREFERENCE_STATUSES:
            raise PreferenceFileError("preference block has an invalid status")
        entries.append(fields)
    return entries


def _preference_path(root: Path) -> Path:
    return root / ".serena" / "memories" / "local" / "user_preferences.md"


def _render_message(root: Path | None) -> str:
    if root is None:
        return (
            FIXED_INSTRUCTION
            + "\n- Warning: project root was not found; no local preferences were restored."
        )

    preference_path = _preference_path(root)
    try:
        entries = _parse_preferences(preference_path)
    except PreferenceFileError as exc:
        return (
            FIXED_INSTRUCTION
            + f"\n- Warning: {exc}; no local preferences were restored."
        )

    confirmed = [entry for entry in entries if entry["status"] == "Confirmed"]
    preference_lines = [
        f"- Confirmed {entry['id']}: prefer={entry['prefer']}; avoid={entry['avoid']}"
        for entry in confirmed
    ]
    candidate = FIXED_INSTRUCTION
    if preference_lines:
        candidate += "\nConfirmed local preferences:\n" + "\n".join(preference_lines)
    if len(candidate) > MAX_OUTPUT_CHARS:
        return (
            FIXED_INSTRUCTION
            + "\n- Warning: confirmed preferences exceed the output limit; none were restored."
        )
    return candidate


def _load_input() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> int:
    payload = _load_input()
    root = _project_root(
        payload.get("cwd") if isinstance(payload.get("cwd"), str) else None
    )
    message = _render_message(root)
    event = payload.get("hook_event_name")
    if event not in ALLOWED_EVENTS:
        message += "\n- Warning: unexpected session event; using the same fail-open instructions."
    result = {
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": message,
        },
    }
    try:
        json.dump(result, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    except (OSError, UnicodeError):
        # Codex treats a clean exit with no output as a successful hook run.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
