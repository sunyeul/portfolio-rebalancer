"""Redaction helpers for Toss credentials and account identifiers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


REDACTED = "<redacted>"
SENSITIVE_HEADERS = frozenset({"authorization", "x-tossinvest-account"})


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return a copy safe for diagnostics."""
    return {
        name: REDACTED if name.lower() in SENSITIVE_HEADERS else value
        for name, value in headers.items()
    }


def redact_known_values(text: str, values: Iterable[str]) -> str:
    """Replace exact known secret/account values in diagnostic text."""
    redacted = text
    for value in values:
        if value:
            redacted = redacted.replace(value, REDACTED)
    return redacted
