"""Error types shared by every MinAgent module.

The original implementation relied on Node.js system errors exposing a string
``code`` such as ``ENOENT``. Python raises :class:`OSError` instead, so every
error crossing a module boundary is normalised into :class:`AgentError`, which
carries the same symbolic ``code`` alongside the ``may_have_changed`` hint that
write verification attaches to failures.
"""

from __future__ import annotations

import asyncio
import errno as _errno
import os


class AgentError(Exception):
    """A user-facing error with an optional symbolic system ``code``."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        # Set by write verification when a failure may still have changed a file.
        self.may_have_changed = False

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class OperationAborted(AgentError):
    """Raised when the user interrupts an in-flight model operation."""


class CancellationToken:
    """Cooperative cancellation shared between the prompt and a model request.

    The prompt keeps reading keys while a response streams, so stopping a turn
    means flipping a flag that the streaming loop checks between chunks.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


def error_code(exc: BaseException) -> str | None:
    """Return the symbolic errno name for an OS error, mirroring Node's codes."""
    if isinstance(exc, OSError) and exc.errno is not None:
        return _errno.errorcode.get(exc.errno)
    return None


def as_agent_error(exc: BaseException) -> AgentError:
    """Normalise any exception into an :class:`AgentError`."""
    if isinstance(exc, AgentError):
        return exc
    code = error_code(exc)
    message = getattr(exc, "message", None) or str(exc)
    return AgentError(message, code)


def is_missing(exc: BaseException) -> bool:
    """True when an exception reports a missing path (Node's ``ENOENT``)."""
    return as_agent_error(exc).code == "ENOENT"


def describe_system_error(exc: BaseException) -> str:
    """Render an error the way the workspace tools report it in messages."""
    normalized = as_agent_error(exc)
    return normalized.code or "access error"


def application_root_candidates() -> list[str]:
    """Directories that may hold ``.env`` and ``.minagent/mcp.json``."""
    override = os.environ.get("MINAGENT_ROOT")
    candidates: list[str] = []
    if override:
        candidates.append(override)
    # Walk up from this module looking for the project root.
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidates.append(here)
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    candidates.append(os.getcwd())
    return candidates


def find_application_root() -> str:
    """Locate the MinAgent project root.

    ``MINAGENT_ROOT`` wins, then the nearest ancestor of this module holding a
    ``pyproject.toml``, and finally the current working directory.
    """
    candidates = application_root_candidates()
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "pyproject.toml")):
            return candidate
    return candidates[-1]
