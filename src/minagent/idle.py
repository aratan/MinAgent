"""Whether the machine is currently being used by anyone, including the user.

An improvement loop that runs because the agent decided to be useful is a loop
that competes with the person sitting in front of it. The agent has no
justification for the second core, and the user has no way to ask for the
throttle that would make it acceptable, so the loop has to measure the machine
before it starts rather than being trusted to be polite.

Two signals, in order of usefulness. **logind's ``IdleHint``** is the honest one:
it is set by the session manager on real user input and cleared by it, so it
answers the question that matters - *is anyone there* - rather than the proxy
question. **Load average** is the fallback for a machine with no session bus, and
it answers a different question: *is the machine busy*, which is close enough for
a throttle and not close enough to trust on its own. A machine can be at zero
load with the user reading a terminal, and a loop that treats that as permission
is the failure this module exists to prevent.

Both signals fail closed. If neither can be read, the answer is "not idle", and
the consequence is that the loop does not run - which costs a deferred
improvement, against a loop that runs unverified on a machine in use.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

# How long a cached reading stays valid. Long enough that a turn loop asking
# every turn spawns one subprocess per window rather than one per turn, short
# enough that sitting down at the machine stops the loop promptly.
DEFAULT_CACHE_SECONDS = 30.0

# IdleHint only says *whether*, not *how long*, so the duration comes from the
# agent's own clock: the first time idle is seen, and then every second after.
# Two minutes is long enough to be deliberate and short enough that a user who
# stepped away for a coffee does not come back to a machine churning.
DEFAULT_MIN_IDLE_SECONDS = 120.0

# Load average is a 1-minute average, so it lags. A session at exactly this load
# on this many cores is treated as in use, because the cost of being wrong in
# that direction is the user's attention and the cost in the other direction is
# one deferred improvement.
DEFAULT_CPU_THRESHOLD = 0.25

_LOGIND_SERVICE = "org.freedesktop.login1"
_LOGIND_MANAGER = "/org/freedesktop/login1"
_LOGIND_SESSION = "org.freedesktop.login1.Session"

DBUS_TIMEOUT_SECONDS = 3.0

_PATH_PATTERN = re.compile(r"'(/org/freedesktop/login1/session/_?\d+)'")
# One ListSessions tuple is (sessionId, userId, userName, seat, objectPath), but
# gdbus renders the numeric field with its type annotation and omits the
# ``objectpath`` label on the headless entry. Both details have to be tolerated
# or the pattern silently pairs one session's seat with the next session's path -
# which is exactly the case where a headless session gets read instead of the
# seated one, and ``IdleHint`` on a headless session is false forever.
_SESSION_TUPLE = re.compile(
    r"\(\s*'[^']*'\s*,\s*uint32\s+\d+\s*,\s*'[^']*'\s*,\s*'([^']*)'\s*,\s*(?:objectpath\s*)?'([^']*)'\s*\)"
)
# gdbus prints a typed tuple as ``(<false>,)`` - the value is wrapped in angle
# brackets inside the parentheses, so a pattern that expects ``(false)`` matches
# nothing and the caller silently falls through to the load-average fallback.
_BOOLEAN_PATTERN = re.compile(r"[<(]\s*(true|false)\s*[>,)]")

SOURCE_LOGIND = "logind"
SOURCE_LOAD = "load"
SOURCE_UNKNOWN = "unknown"

IDLE = "idle"
BUSY = "busy"


@dataclass(frozen=True)
class IdleState:
    """One reading, with the reason, because a throttle that cannot explain
    itself is indistinguishable from a throttle that is broken."""

    state: str
    source: str
    idle_seconds: float | None = None
    detail: str = ""

    @property
    def idle(self) -> bool:
        return self.state == IDLE


@dataclass
class IdleReader:
    """Reads the machine's idleness, caching briefly and never raising.

    The cache is not an optimisation. Reading the session bus means a
    subprocess, and a turn loop that spawns one per turn turns a throttle into
    a cost.
    """

    threshold: float = DEFAULT_CPU_THRESHOLD
    minimum_idle_seconds: float = DEFAULT_MIN_IDLE_SECONDS
    cache_seconds: float = DEFAULT_CACHE_SECONDS
    now: object = time.time

    _cached: IdleState | None = None
    _cached_at: float = 0.0
    _idle_since: float | None = None

    def read(self) -> IdleState:
        """One reading, from cache when it is still young enough."""
        moment = float(self.now())  # type: ignore[operator]
        if self._cached is not None and moment - self._cached_at < self.cache_seconds:
            return self._cached
        state = _read_uncached(moment, self.threshold, self.minimum_idle_seconds)
        if not state.idle:
            # Leaving the running idle clock early is deliberate: the moment the
            # user comes back, the agent owes them the machine immediately, not
            # after the minimum has elapsed again.
            self._idle_since = None
        elif self._idle_since is None:
            self._idle_since = moment
        state = _apply_minimum(state, moment, self._idle_since, self.minimum_idle_seconds)
        self._cached = state
        self._cached_at = moment
        return state

    def clear(self) -> None:
        """Forget the reading and the running clock. Used when the loop is stopped."""
        self._cached = None
        self._cached_at = 0.0
        self._idle_since = None


def _apply_minimum(
    state: IdleState, moment: float, idle_since: float | None, minimum: float
) -> IdleState:
    """Hold a short idle back until it has lasted long enough to be real."""
    if not state.idle:
        return state
    if idle_since is None or minimum <= 0 or moment - idle_since >= minimum:
        return state
    remaining = max(0.0, minimum - (moment - idle_since))
    return IdleState(
        state=BUSY,
        source=state.source,
        idle_seconds=0.0,
        detail=f"idle for {remaining:.0f}s of the {minimum:.0f}s the loop waits",
    )


def _read_uncached(moment: float, threshold: float, minimum: float) -> IdleState:
    hint = _logind_idle_hint()
    if hint is not None:
        if hint:
            return IdleState(IDLE, SOURCE_LOGIND, 0.0, "logind reports no user input")
        return IdleState(BUSY, SOURCE_LOGIND, 0.0, "logind reports recent user input")
    load = _normalised_load()
    if load is None:
        return IdleState(
            BUSY,
            SOURCE_UNKNOWN,
            None,
            "neither the session bus nor /proc/loadavg could be read; refusing to assume",
        )
    if load <= threshold:
        return IdleState(
            IDLE,
            SOURCE_LOAD,
            0.0,
            f"load {load:.2f} of capacity is at or below the {threshold:.2f} threshold",
        )
    return IdleState(
        BUSY,
        SOURCE_LOAD,
        0.0,
        f"load {load:.2f} of capacity is above the {threshold:.2f} threshold",
    )


def _run(command: list[str]) -> str:
    """One subprocess, bounded, never raising. A missing helper is an answer."""
    if not shutil.which(command[0]):
        return ""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=DBUS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def _logind_session_path() -> str:
    """The object path of the local session, preferring one that owns a seat.

    A machine can hold more than one session - a headless one and the desktop
    one, typically. ``IdleHint`` on the headless session is false forever
    because nothing ever sends it input, so reading that one reports the machine
    as permanently busy and the loop never runs at all. The seated session is
    the one whose idle state means anything.
    """
    output = _run([
        "gdbus", "call", "--system",
        "-d", _LOGIND_SERVICE,
        "-o", _LOGIND_MANAGER,
        "-m", f"{_LOGIND_SERVICE}.Manager.ListSessions",
    ])
    if not output:
        return ""
    seated = _SESSION_TUPLE.findall(output)
    for seat, path in seated:
        if seat and _is_session_path(path):
            return path
    # No seated session found. A headless path is better than none - it will read
    # as permanently busy, which is the fail-closed answer - but it is reported
    # as such rather than passed off as a real reading.
    for _seat, path in seated:
        if _is_session_path(path):
            return path
    paths = _PATH_PATTERN.findall(output)
    return paths[0] if paths else ""


def _is_session_path(path: str) -> bool:
    return bool(_PATH_PATTERN.fullmatch(path))


def _logind_idle_hint() -> bool | None:
    """``True`` when no user input has arrived, ``None`` when it cannot be read."""
    path = _logind_session_path()
    if not path:
        return None
    output = _run([
        "gdbus", "call", "--system",
        "-d", _LOGIND_SERVICE,
        "-o", path,
        "-m", "org.freedesktop.DBus.Properties.Get",
        _LOGIND_SESSION, "IdleHint",
    ])
    if not output:
        return None
    match = _BOOLEAN_PATTERN.search(output)
    if not match:
        return None
    return match.group(1) == "true"


def _normalised_load() -> float | None:
    """One-minute load average as a fraction of the machine's capacity."""
    try:
        with open("/proc/loadavg", encoding="utf-8") as handle:
            first = handle.read().split()[0]
        load = float(first)
    except (OSError, IndexError, ValueError):
        return None
    cores = os.cpu_count() or 1
    return load / cores


def should_run(enabled: bool, reader: IdleReader | None = None, *, minimum_idle_seconds: float = DEFAULT_MIN_IDLE_SECONDS) -> IdleState:
    """The gate itself, in the shape a caller can put in an ``if``.

    Disabled is reported as a distinct state rather than as idle, so a
    transcript can say the loop is switched off instead of saying nothing
    happened. ``minimum_idle_seconds`` is accepted for symmetry with
    :class:`IdleReader` and deliberately not applied here: the reader already
    holds a short idle back, and applying the wait a second time would mean the
    gate's own threshold silently overrode the one the reader was configured
    with.
    """
    if not enabled:
        return IdleState(BUSY, "disabled", None, "autonomous improvement is switched off")
    if reader is None:
        return IdleState(BUSY, "disabled", None, "no idle reader was supplied")
    return reader.read()
