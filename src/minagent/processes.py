"""Process-tree termination.

Long-running shell commands and stdio MCP servers are started in their own
session, so killing the whole group is what stops a command that spawned
children of its own.
"""

from __future__ import annotations

import os
import signal
from typing import Any


def _has_exited(child: Any) -> bool:
    """Read the exit state of a ``subprocess.Popen`` or an asyncio child process."""
    poll = getattr(child, "poll", None)
    if callable(poll):
        return poll() is not None
    return getattr(child, "returncode", None) is not None


def terminate_process_tree(child: Any | None) -> None:
    """Kill a child process and every process it spawned.

    Prefers signalling the child's process group; falls back to the child alone
    when the group is already gone.
    """
    if child is None or child.pid is None:
        return
    if _has_exited(child) and os.name != "nt":
        return
    try:
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            child.kill()
        except (ProcessLookupError, OSError):
            pass
