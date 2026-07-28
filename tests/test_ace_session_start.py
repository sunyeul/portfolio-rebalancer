from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".codex" / "hooks" / "ace_session_start.py"
SPEC = spec_from_file_location("ace_session_start", SCRIPT)
assert SPEC and SPEC.loader
HOOK = module_from_spec(SPEC)
SPEC.loader.exec_module(HOOK)


def _run_hook(
    cwd: Path, payload: object
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.stderr == ""
    return completed, json.loads(completed.stdout)


def _additional_context(result: dict[str, object]) -> str:
    output = result["hookSpecificOutput"]
    assert isinstance(output, dict)
    assert output["hookEventName"] == "SessionStart"
    context = output["additionalContext"]
    assert isinstance(context, str)
    return context


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    preference_dir = tmp_path / ".serena" / "memories" / "local"
    preference_dir.mkdir(parents=True)
    return preference_dir


def _preference_block(status: str, prefer: str) -> str:
    return "\n".join(
        (
            f"## pref-test-{status.lower()}",
            "- title: Test preference",
            f"- status: {status}",
            "- context: Test context",
            f"- prefer: {prefer}",
            "- avoid: Do not use the opposite behavior",
            "- source: Explicit user request",
            "- helpful: Keeps the workflow predictable",
            "- harmful: May be too narrow outside this context",
        )
    )


def test_restores_confirmed_preferences_only(tmp_path: Path) -> None:
    preference_dir = _make_repo(tmp_path)
    (preference_dir / "user_preferences.md").write_text(
        "# Preferences\n\n"
        + _preference_block("Confirmed", "show evidence before completion")
        + "\n\n"
        + _preference_block("Pending", "infer my preferred format"),
        encoding="utf-8",
    )

    completed, result = _run_hook(
        tmp_path,
        {"cwd": str(tmp_path), "hook_event_name": "startup"},
    )

    assert completed.returncode == 0
    assert result["continue"] is True
    message = _additional_context(result)
    assert "Read AGENTS.md" in message
    assert ".agents/skills/ace-collaboration-memory/SKILL.md" in message
    assert "show evidence before completion" in message
    assert "infer my preferred format" not in message


def test_malformed_and_oversized_preferences_fail_open(tmp_path: Path) -> None:
    preference_dir = _make_repo(tmp_path)
    preference_path = preference_dir / "user_preferences.md"
    preference_path.write_text(
        "## pref-bad\n- status: Confirmed\n- prefer: secret candidate\n",
        encoding="utf-8",
    )
    completed, malformed = _run_hook(
        tmp_path,
        {"cwd": str(tmp_path), "hook_event_name": "resume"},
    )
    assert completed.returncode == 0
    malformed_context = _additional_context(malformed)
    assert "no local preferences were restored" in malformed_context
    assert "secret candidate" not in malformed_context
    assert "Read AGENTS.md" in malformed_context

    preference_path.write_text("x" * (HOOK.MAX_PREFERENCE_BYTES + 1), encoding="utf-8")
    completed, oversized = _run_hook(
        tmp_path,
        {"cwd": str(tmp_path), "hook_event_name": "compact"},
    )
    assert completed.returncode == 0
    oversized_context = _additional_context(oversized)
    assert "exceeds 16 KiB" in oversized_context
    assert "Read AGENTS.md" in oversized_context

    preference_path.write_text(
        "\n\n".join(_preference_block("Confirmed", "x" * 100) for _ in range(15)),
        encoding="utf-8",
    )
    completed, output_limited = _run_hook(
        tmp_path,
        {"cwd": str(tmp_path), "hook_event_name": "startup"},
    )
    assert completed.returncode == 0
    output_limited_context = _additional_context(output_limited)
    assert "output limit" in output_limited_context
    assert "Confirmed local preferences:" not in output_limited_context


def test_missing_root_and_hook_configuration_are_safe(tmp_path: Path) -> None:
    completed, missing_root = _run_hook(
        tmp_path,
        {"cwd": str(tmp_path), "hook_event_name": "clear"},
    )
    assert completed.returncode == 0
    assert "project root was not found" in _additional_context(missing_root)

    repo_without_preferences = tmp_path / "repo-without-preferences"
    (repo_without_preferences / ".git").mkdir(parents=True)
    completed, missing_file = _run_hook(
        repo_without_preferences,
        {"cwd": str(repo_without_preferences), "hook_event_name": "clear"},
    )
    assert completed.returncode == 0
    assert "preference file is missing" in _additional_context(missing_file)

    hook_config = json.loads(
        (ROOT / ".codex" / "hooks.json").read_text(encoding="utf-8")
    )
    groups = hook_config["hooks"]["SessionStart"]
    assert len(groups) == 1
    assert groups[0]["matcher"] == "startup|resume|clear|compact"
    handler = groups[0]["hooks"][0]
    assert handler["type"] == "command"
    assert handler["command"].endswith('.codex/hooks/ace_session_start.py"')

    project_config = (ROOT / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "[features]\nhooks = true" in project_config


def test_invalid_stdin_still_returns_json() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        input="not json",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["continue"] is True
    assert "project root was not found" in _additional_context(result)
