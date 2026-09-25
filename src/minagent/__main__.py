"""Command-line entry point: ``minagent`` and ``python -m minagent``."""

from __future__ import annotations

import asyncio
import sys

from .app import main as _run


def main() -> int:
    """Run a MinAgent session and return the process exit code."""
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
