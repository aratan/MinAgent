"""Command-line entry point: ``minagent`` and ``python -m minagent``.

Two modes, deliberately. ``minagent`` with no arguments is the interactive
session, unchanged. ``minagent resident`` is the non-interactive one, built so
the improvement loop can run without a terminal attached - under systemd, cron
or a container. That second mode could not exist before, because the loop was
only ever started as a side effect of a session, so a machine nobody was typing
into was a machine that never improved anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys

from .app import AgentError
from .app import main as _run

USAGE = """usage: minagent [resident]

  (no arguments)  start an interactive session
  resident        run the improvement loop with no terminal, until stopped
"""


async def _run_resident() -> int:
    """Run the improvement loop alone, until asked to stop.

    Imported here rather than at module level so ``minagent`` with no arguments
    does not pay for the loop's imports on every start.
    """
    from .app import MinAgent

    agent = MinAgent()
    await agent.initialize_configuration()

    worker = agent._start_resident_worker()  # noqa: SLF001 - the same door the session uses
    if worker is None or not worker.enabled:
        # Not a silent success. An entrypoint that starts, does nothing and
        # exits 0 is indistinguishable from one that worked, and the difference
        # matters here because the whole point is that work happens.
        print(
            # `on`, not `true`: the configuration validator accepts only on/off
            # and rejects anything else, so advising `true` would send the
            # reader straight into a parse error.
            "The improvement loop is off. Set IMPROVEMENT_AUTONOMOUS=on "
            "in your .env before starting it.",
            file=sys.stderr,
        )
        return 1

    # systemd stops a service with SIGTERM, not SIGINT. Without this the loop
    # is killed mid-cycle and the shutdown handler that closes the trial
    # document never runs, leaving a half-written file for the next start.
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signame in ("SIGTERM", "SIGINT"):
        with contextlib.suppress(NotImplementedError, AttributeError, ValueError):
            loop.add_signal_handler(getattr(signal, signame), stop.set)

    print("Improvement loop running. Stop with SIGTERM or Ctrl-C.", file=sys.stderr, flush=True)
    try:
        await stop.wait()
    finally:
        # Not an `except`: the loop must also be shut down when the wait is
        # cancelled, because a cycle holding provider calls is still spending.
        await agent._stop_resident_worker()  # noqa: SLF001
    return 0


async def _run_mode() -> int:
    """Dispatch to the mode named on the command line."""
    mode = sys.argv[1:2]
    if mode == ["resident"]:
        return await _run_resident()
    if mode in (["-h"], ["--help"]):
        print(USAGE, end="")
        return 0
    if mode:
        # Refused rather than ignored. `minagent resident` and `minagent
        # resindent` are one character apart, and answering the typo with a
        # normal interactive session looks like it worked - the user watches a
        # prompt instead of a loop and has no way to know the argument was
        # dropped.
        print(f"Unknown argument: {mode[0]}\n\n{USAGE}", file=sys.stderr, end="")
        return 2
    return await _run()


def main() -> int:
    """Run MinAgent and return the process exit code."""
    try:
        return asyncio.run(_run_mode())
    except KeyboardInterrupt:
        return 0
    except AgentError as error:
        # Configuration and provider failures are reported, not raised: a
        # service that exits with a traceback tells nobody what to fix.
        print(f"Ara: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
