"""Small helpers that keep JavaScript semantics that Python spells differently.

The original code leaned on a handful of JS behaviours that have no direct
Python equivalent: ``Buffer.byteLength``, ``Number.isInteger`` (which rejects
``NaN``, floats and booleans), ``JSON.stringify`` spacing, and
``String.prototype.localeCompare`` ordering.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any


def byte_length(value: str) -> int:
    """UTF-8 byte length, matching ``Buffer.byteLength(value, "utf8")``."""
    return len(value.encode("utf-8", errors="surrogatepass"))


def is_int(value: Any) -> bool:
    """Mirror ``Number.isInteger``: integral numbers only, never booleans."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value) and value.is_integer()
    return False


def json_stringify(value: Any) -> str:
    """``JSON.stringify`` compatible serialisation: compact and non-escaping."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def locale_key(value: str) -> tuple[str, str, str]:
    """Sort key approximating ICU root collation used by ``localeCompare``.

    Accents are compared as a secondary level so ``"e"`` sorts next to ``"é"``,
    case only breaks ties, which is what the file listings depend on.
    """
    return (_strip_accents(value).casefold(), value.casefold(), value)


def decode_utf8(data: bytes) -> str:
    """Strict UTF-8 decode, matching ``TextDecoder("utf-8", { fatal: true })``."""
    return data.decode("utf-8")
