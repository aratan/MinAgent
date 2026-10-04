"""Contracts of the command-line entry point.

The improvement loop used to be reachable only as a side effect of a typed
session, so these cover a door that did not exist before: that it exists, that
it refuses to fake success, that a mistyped mode is refused rather than
swallowed, and that SIGTERM stops it cleanly.

The two that involve a running process use real subprocesses on purpose. Testing
signal handling with a stub proves the stub works; the contract is that the
installed program answers SIGTERM, and only the installed program can show it.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from minagent.resident import _claim_exclusive, _release

ROOT = Path(__file__).resolve().parent.parent


def _isolated_project(tmp_path, **settings: str) -> str:
    """A throwaway project the loop can be started in without touching the real one.

    Two things have to be true for the isolation to hold: the directory has to
    look like a project, or ``MINAGENT_ROOT`` is ignored and the app falls back
    to the repository this test is running in; and it needs its own ``.env``,
    because the app reads that rather than the environment it was handed. Long
    cycle and idle values mean no cycle ever fires: these tests are about the
    lifecycle, not the work.
    """
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "resident-test"\n', encoding="utf-8")
    lines = ["OPENAI_MODEL=test-model", "IMPROVEMENT_AUTONOMOUS=on", "IMPROVEMENT_CYCLE_SECONDS=3600"]
    lines += [f"{name}={value}" for name, value in settings.items()]
    (tmp_path / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(tmp_path)


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    environment.update(env or {})
    return subprocess.run(
        [sys.executable, "-m", "minagent", *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
        env=environment,
    )


def test_help_names_both_modes():
    result = _run(["--help"])

    assert result.returncode == 0
    assert "resident" in result.stdout


def test_a_misspelled_mode_is_refused_instead_of_running_a_session():
    """`resident` and `resindent` are one character apart.

    Answering the typo with a normal interactive session looks like it worked:
    the user gets a prompt instead of a loop, with no way to know the argument
    was dropped on the floor.
    """
    result = _run(["resindent"])

    assert result.returncode == 2
    assert "resindent" in result.stderr
    # It must not have fallen through to the interactive session, which would
    # have printed a startup panel to stdout.
    assert result.stdout == ""


def test_resident_refuses_when_the_loop_is_off():
    """Off means off, and the message has to be one the reader can follow.

    The configuration validator accepts only on/off, so a message advising
    `true` walks the reader into a parse error - which is the same failure as no
    message at all.
    """
    result = _run(["resident"], {"IMPROVEMENT_AUTONOMOUS": "off"})

    assert result.returncode == 1
    assert "IMPROVEMENT_AUTONOMOUS" in result.stderr
    assert "=on" in result.stderr
    assert "true" not in result.stderr


def test_a_value_the_validator_rejects_is_reported_not_traced():
    """A bad setting is a configuration error, not a crash to be read as a bug.

    A service that exits with a traceback tells the operator nothing about which
    value is wrong.
    """
    result = _run(["resident"], {"IMPROVEMENT_AUTONOMOUS": "maybe"})

    assert result.returncode == 1
    assert "IMPROVEMENT_AUTONOMOUS" in result.stderr
    assert "Traceback" not in result.stderr


def test_sigterm_stops_the_loop_and_waits_for_it(tmp_path):
    """systemd stops a service with SIGTERM, so SIGTERM alone has to be enough.

    The cycle lengths here are long enough that no cycle can start, so this
    proves the lifecycle - start, hold, stop cleanly - without spending a single
    provider call. The reason it matters: the default disposition for SIGTERM
    kills the process mid-cycle, and the shutdown that closes the trial document
    never runs, leaving a half-written file for the next start to find.
    """
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "IMPROVEMENT_IDLE_SECONDS": "3600",
        "MINAGENT_ROOT": _isolated_project(tmp_path),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "minagent", "resident"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
        env=environment,
    )
    ready = threading.Event()
    captured: list[str] = []

    def _watch(stream) -> None:
        for line in stream:
            captured.append(line)
            if "Improvement loop running" in line:
                ready.set()

    watcher = threading.Thread(target=_watch, args=(process.stderr,), daemon=True)
    watcher.start()
    try:
        assert ready.wait(timeout=60), f"it never announced itself: {''.join(captured)}"
        assert process.poll() is None, f"it exited right after starting: {''.join(captured)}"
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 0, "a signalled stop is not a clean stop"
    finally:
        if process.poll() is None:  # pragma: no cover - only on a failure above
            process.kill()
            process.wait(timeout=30)


def test_an_unknown_mode_never_reaches_the_session(monkeypatch):
    """In-process guard on the dispatcher, so a future mode cannot bypass it."""
    from minagent import __main__ as entry

    called = asyncio.Event()

    async def _session() -> int:
        called.set()
        return 0

    monkeypatch.setattr(entry, "_run", _session)
    monkeypatch.setattr(sys, "argv", ["minagent", "whatever"])

    assert asyncio.run(entry._run_mode()) == 2
    assert not called.is_set()


def test_a_second_resident_says_why_it_did_not_start_and_exits(tmp_path):
    """The service and an interactive session must not both work at once.

    A process that waits for a signal it will never get reports itself as
    running for as long as systemd leaves it up, while the loop inside it ended
    at the first cycle. Exiting with the reason is the only version an operator
    can act on.
    """
    project = _isolated_project(tmp_path)
    held, _, handle = _claim_exclusive(str(tmp_path / ".minagent" / "resident.lock"))
    assert held
    try:
        process = subprocess.run(
            [sys.executable, "-m", "minagent", "resident"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=project,
            env={**os.environ, "PYTHONPATH": str(ROOT), "MINAGENT_ROOT": project},
        )
    finally:
        _release(handle)

    assert process.returncode == 1
    assert "did not start" in process.stderr
    assert "resident.lock" in process.stderr
