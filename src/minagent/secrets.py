"""Secret redaction and approval previews.

Anything shown to the user before a tool runs - and any project file handed to
``/init`` - is scrubbed of values that look like credentials.
"""

from __future__ import annotations

import re
from typing import Any

from .jsutil import json_stringify

_SECRET_ASSIGNMENT = re.compile(
    r"""(^|[\s,{])([A-Za-z][A-Za-z0-9_.-]*\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;}]+)""",
    re.IGNORECASE | re.MULTILINE,
)
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")
_BEARER = re.compile(r"\b(Bearer\s+)[A-Za-z0-9._~+/-]+=*", re.IGNORECASE)
_KNOWN_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9_]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{12,}"
    r"|AKIA[0-9A-Z]{16})\b"
)
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_CAMEL_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ASSIGNMENT_SEPARATOR = re.compile(r"\s*[:=]")

_SECRET_WORDS = {"token", "secret", "password", "passwd", "credential", "authorization", "cookie"}
_SECRET_KEY_PREDECESSORS = {"api", "access", "private", "client"}


def is_secret_name(value: str) -> bool:
    """True when a variable, header, or key name suggests a credential."""
    spaced = _CAMEL_BOUNDARY.sub(r"\1_\2", str(value)).lower()
    words = [word for word in _NON_ALNUM.split(spaced) if word]
    if any(word in _SECRET_WORDS for word in words):
        return True
    for index, word in enumerate(words):
        if word == "key" and index > 0 and words[index - 1] in _SECRET_KEY_PREDECESSORS:
            return True
    return False


def _redact_assignment(match: re.Match[str]) -> str:
    prefix, assignment = match.group(1), match.group(2)
    name = _ASSIGNMENT_SEPARATOR.split(assignment, maxsplit=1)[0]
    return f"{prefix}{assignment}[REDACTED]" if is_secret_name(name) else match.group(0)


def redact_likely_secrets(value: Any) -> str:
    """Remove private keys, bearer tokens, and secret-looking assignments."""
    text = "" if value is None else str(value)
    text = _PRIVATE_KEY.sub("[REDACTED PRIVATE KEY]", text)
    text = _BEARER.sub(r"\1[REDACTED]", text)
    text = _KNOWN_TOKEN.sub("[REDACTED TOKEN]", text)
    text = _JWT.sub("[REDACTED JWT]", text)
    return _SECRET_ASSIGNMENT.sub(_redact_assignment, text)


def _mask(value: Any, depth: int = 0) -> Any:
    """Recursively replace secret-looking keys, bounding size and depth."""
    if depth > 8:
        return "[Nested value omitted]"
    if isinstance(value, (list, tuple)):
        return [_mask(item, depth + 1) for item in value[:50]]
    if isinstance(value, dict):
        masked = {}
        for key, child in list(value.items())[:100]:
            masked[key] = "[REDACTED]" if is_secret_name(str(key)) else _mask(child, depth + 1)
        return masked
    return value


def approval_preview(value: Any, max_chars: int = 1200) -> str:
    """Render masked tool arguments for the user's approval prompt."""
    try:
        preview = json_stringify(_mask(value))
    except Exception:
        return "[Arguments could not be displayed]"
    if len(preview) > max_chars:
        return f"{preview[:max_chars]}… [preview truncated]"
    return preview
