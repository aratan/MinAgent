"""Ara's own account of who it is, kept in files instead of in the code.

The identity used to be one line inside ``build_base_system_prompt``. That
line is the right place for a rule that never changes and the wrong place for
a self-description: an agent that cannot edit its own account of itself is
one that has to be redeployed to change its mind.

So the account lives in ``agente/IDENTITY.md`` and ``agente/USER.md`` beside
the code, in the repository, where an ordinary ``write_file`` reaches it and
every change lands in git with an author and a diff. Nothing new was given to
the agent: the file tools it already has can already reach these paths.

The size cap is the part that matters. A personality file that grows without
bound is a context bomb: it is paid for on every single request, it is
invisible while it is happening, and by the time a request fails the file is
already the largest thing in the prompt. A file over the cap is therefore
truncated *and labelled as truncated inside the prompt the model reads*, so
the model knows its own instructions are incomplete instead of quietly
behaving as though the missing rules were never written.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .errors import find_application_root

#: Prompt section name -> path relative to the application root.
PERSONA_FILES: dict[str, str] = {
    "Identity": os.path.join("agente", "IDENTITY.md"),
    "User": os.path.join("agente", "USER.md"),
}

#: Largest a persona file may be before it is cut and marked as cut.
#:
#: 8 KiB is roughly two thousand words, which is far more than a usable self
#: description and far less than a context budget. The cap is deliberately
#: not generous: a persona that outgrows it wants to be summarised, and the
#: summary is a decision the agent should make visibly rather than have the
#: loader make silently on its behalf.
PERSONA_MAX_BYTES = 8 * 1024

_TRUNCATION_NOTE = (
    "\n\n[truncated: this file is over {limit} bytes, so only the first "
    "{kept} were loaded. Treat any rule you expected to find below this line "
    "as absent rather than as never written, and say so if it matters.]"
)


def load_persona_file(path: str, *, limit: int = PERSONA_MAX_BYTES) -> str | None:
    """Read one persona file, truncated and labelled when it is too long.

    Returns ``None`` when the file does not exist: a missing persona is not an
    error, it just means the agent has not written one yet. Returns the
    contents, with an in-band note, when it is too long. Never raises for a
    file that is merely too big -- an overgrown persona should cost the agent
    some context, not take it offline.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read(limit + 1)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        # Unreadable is treated as absent rather than fatal: the prompt should
        # not fail to build because a persona file is corrupt.
        return None

    if len(raw.encode("utf-8")) <= limit:
        return raw.strip() or None

    kept = raw.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    return kept.strip() + _TRUNCATION_NOTE.format(limit=limit, kept=limit)


def persona_sections(
    root: str | None = None, *, limit: int = PERSONA_MAX_BYTES
) -> list[dict[str, str]]:
    """Build the prompt sections that carry Ara's own identity and user model.

    Returns them in a fixed order and omits any file that is not there, so an
    empty list means "no persona written yet" rather than a failure.
    """
    base = root or find_application_root()
    sections: list[dict[str, str]] = []
    for name, relative in PERSONA_FILES.items():
        text = load_persona_file(os.path.join(base, relative), limit=limit)
        if text:
            sections.append({"name": name, "content": text})
    return sections


def persona_overflows(
    root: str | None = None, *, limit: int = PERSONA_MAX_BYTES
) -> list[str]:
    """Names of the persona files that are over the cap, for the status view.

    The truncation is already visible to the model; this is the same fact
    reported to the human, so an overgrown persona does not need the context
    meter to be interpreted before anyone notices.
    """
    base = root or find_application_root()
    over: list[str] = []
    for name, relative in PERSONA_FILES.items():
        path = os.path.join(base, relative)
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if size > limit:
            over.append(f"{name} ({size} bytes)")
    return over


def describe_persona(sections: Sequence[dict[str, str]]) -> str:
    """One line per loaded section, for status output."""
    if not sections:
        return "sin personalidad escrita"
    return ", ".join(f"{section['name']} ({len(section['content'])}B)" for section in sections)
