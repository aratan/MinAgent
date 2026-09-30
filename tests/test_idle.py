"""Whether anyone is at the machine, which decides whether the agent may start.

The whole feature this serves is work the user did not ask for. Running it
because the agent decided it would be useful means taking the second core from
someone who is sitting in front of the machine, and there is no setting the user
can reach to throttle it afterwards. So the gate has to be measurable, and every
failure mode below resolves to "not idle" - a deferred improvement is cheap, and
an unthrottled loop is not.
"""

from minagent import idle
from minagent.idle import (
    BUSY,
    IDLE,
    SOURCE_LOAD,
    SOURCE_LOGIND,
    SOURCE_UNKNOWN,
    IdleReader,
    IdleState,
    should_run,
)

SESSIONS = (
    "([('3', uint32 1000, 'victor', 'seat0', objectpath "
    "'/org/freedesktop/login1/session/_33'), ('1', 1000, 'victor', '', "
    "'/org/freedesktop/login1/session/_31')],)"
)
SESSIONS_HEADLESS_FIRST = (
    "([('1', 1000, 'victor', '', '/org/freedesktop/login1/session/_31'), "
    "('3', uint32 1000, 'victor', 'seat0', objectpath "
    "'/org/freedesktop/login1/session/_33')],)"
)


def _hint(monkeypatch, value: str):
    monkeypatch.setattr(
        idle, "_run", lambda command: SESSIONS if "ListSessions" in command[-1] else value
    )


def _quiet(monkeypatch):
    """A machine with no user input, reported by logind."""
    monkeypatch.setattr(
        idle, "_run", lambda command: SESSIONS if "ListSessions" in command[-1] else "(<true>,)"
    )


def _reader(monkeypatch, *, output: str, idle_hint: str, **kwargs) -> IdleReader:
    monkeypatch.setattr(idle, "_run", lambda command: output if "ListSessions" in command[-1] else idle_hint)
    return IdleReader(cache_seconds=kwargs.pop("cache_seconds", 0.0), **kwargs)


# --- Reading logind ----------------------------------------------------------


def test_the_seated_session_is_the_one_read(monkeypatch):
    """The headless session's IdleHint is false forever, because nothing ever
    sends it input. Reading that one reports the machine as permanently busy and
    the loop never runs at all."""
    reader = _reader(monkeypatch, output=SESSIONS, idle_hint="(<false>,)")

    state = reader.read()

    assert state.source == SOURCE_LOGIND
    assert state.state == BUSY


def test_a_seated_session_is_found_even_when_listed_second(monkeypatch):
    """The bug this pins: taking the first path in the list returns the
    headless session whenever it happens to be listed first."""
    reader = _reader(monkeypatch, output=SESSIONS_HEADLESS_FIRST, idle_hint="(<false>,)")

    assert reader.read().source == SOURCE_LOGIND


def test_the_typed_boolean_is_read(monkeypatch):
    """gdbus prints ``(<true>,)``. A pattern expecting ``(true)`` matches nothing
    and the caller silently falls through to the load-average fallback."""
    monkeypatch.setattr(idle, "_run", lambda command: SESSIONS if "ListSessions" in command[-1] else "(<true>,)")

    state = IdleReader(cache_seconds=0.0, minimum_idle_seconds=0.0).read()

    assert state.state == IDLE
    assert state.source == SOURCE_LOGIND


def test_an_unreadable_session_bus_falls_back_to_load(monkeypatch):
    monkeypatch.setattr(idle, "_run", lambda command: "")
    monkeypatch.setattr(idle, "_normalised_load", lambda: 0.05)

    state = IdleReader(cache_seconds=0.0, minimum_idle_seconds=0.0).read()

    assert state.source == SOURCE_LOAD
    assert state.state == IDLE


# --- Failing closed ----------------------------------------------------------


def test_neither_signal_readable_means_not_idle(monkeypatch):
    """The important one. Guessing wrong in this direction is the whole
    failure this module exists to prevent."""
    monkeypatch.setattr(idle, "_run", lambda command: "")
    monkeypatch.setattr(idle, "_normalised_load", lambda: None)

    state = IdleReader(cache_seconds=0.0, minimum_idle_seconds=0.0).read()

    assert state.state == BUSY
    assert state.source == SOURCE_UNKNOWN
    assert "refusing" in state.detail


def test_a_busy_machine_reports_busy(monkeypatch):
    monkeypatch.setattr(idle, "_run", lambda command: "")
    monkeypatch.setattr(idle, "_normalised_load", lambda: 0.9)

    state = IdleReader(cache_seconds=0.0, minimum_idle_seconds=0.0).read()

    assert state.state == BUSY


def test_load_is_measured_against_the_number_of_cores(monkeypatch, tmp_path):
    """A load average of 4 means something very different on 4 cores than on
    64, so it is normalised before it is compared to a threshold."""
    loadavg = tmp_path / "loadavg"
    loadavg.write_text("4.00 2.00 1.00 1/200 12345\n")
    monkeypatch.setattr(idle.os, "cpu_count", lambda: 8)
    _redirect_proc(monkeypatch, loadavg)

    assert idle._normalised_load() == 0.5


def test_unreadable_loadavg_is_none(monkeypatch, tmp_path):
    _redirect_proc(monkeypatch, tmp_path / "does-not-exist")

    assert idle._normalised_load() is None


def _redirect_proc(monkeypatch, path):
    """Point the /proc read at a fixture file."""
    real_open = __builtins__["open"] if isinstance(__builtins__, dict) else __builtins__.open
    monkeypatch.setattr(
        "builtins.open",
        lambda file, mode="r", *a, **k: real_open(path, mode),
    )


# --- The minimum wait --------------------------------------------------------


def test_a_short_idle_is_held_back(monkeypatch):
    """Sitting down for four seconds is not the same as having gone to bed."""
    _quiet(monkeypatch)
    clock = {"now": 1000.0}
    reader = IdleReader(cache_seconds=0.0, minimum_idle_seconds=120.0, now=lambda: clock["now"])

    first = reader.read()
    clock["now"] += 30
    held = reader.read()
    clock["now"] += 30
    still_held = reader.read()
    clock["now"] += 61
    released = reader.read()

    # The very first idle sample is held too, not waved through. A fresh reader
    # has no idea how long the machine has been quiet, so letting the first
    # sample through would start the loop on a machine the user left two
    # seconds ago.
    assert first.state == BUSY
    assert "120s" in first.detail
    assert held.state == BUSY
    assert "90s" in held.detail
    assert still_held.state == BUSY
    assert "60s" in still_held.detail
    assert released.state == IDLE


def test_the_wait_restarts_the_moment_the_user_returns(monkeypatch):
    """The user coming back is owed the machine immediately, and the wait after
    they leave again starts over rather than resuming where it was."""
    _quiet(monkeypatch)
    clock = {"now": 500.0}
    reader = IdleReader(cache_seconds=0.0, minimum_idle_seconds=120.0, now=lambda: clock["now"])

    _hint(monkeypatch, "(<true>,)")
    reader.read()          # seeds the idle clock at 500
    clock["now"] += 200
    assert reader.read().state == IDLE      # 200s of quiet clears the 120s wait

    _hint(monkeypatch, "(<false>,)")
    assert reader.read().state == BUSY      # the user is back

    _hint(monkeypatch, "(<true>,)")
    held = reader.read()

    assert held.state == BUSY
    assert "120s" in held.detail            # the wait restarted, it did not resume


# --- Caching and the gate ----------------------------------------------------


def test_a_reading_is_cached_so_a_turn_loop_does_not_spawn_per_turn(monkeypatch):
    calls = {"n": 0}

    def counted(command):
        calls["n"] += 1
        return SESSIONS if "ListSessions" in command[-1] else "(<true>,)"

    monkeypatch.setattr(idle, "_run", counted)
    reader = IdleReader(cache_seconds=30.0, minimum_idle_seconds=0.0)

    for _ in range(5):
        reader.read()

    assert calls["n"] == 2  # one ListSessions, one Properties.Get


def test_clearing_drops_the_cached_reading(monkeypatch):
    """A cached answer that survives a stop is a gate that can start the loop
    on a machine the user has since come back to."""
    _quiet(monkeypatch)
    reader = IdleReader(cache_seconds=3600.0, minimum_idle_seconds=0.0)

    assert reader.read().state == IDLE
    _hint(monkeypatch, "(<false>,)")
    assert reader.read().state == IDLE      # still the cached answer

    reader.clear()
    assert reader.read().state == BUSY      # the truth, re-read


def test_clearing_restarts_the_wait_rather_than_keeping_the_old_one(monkeypatch):
    """Consistent with the rule above: after a stop, a fresh reader has no idea
    how long the machine has been quiet, so it waits again."""
    _quiet(monkeypatch)
    clock = {"now": 0.0}
    reader = IdleReader(cache_seconds=0.0, minimum_idle_seconds=120.0, now=lambda: clock["now"])

    reader.read()
    reader.clear()
    clock["now"] += 500

    assert reader.read().state == BUSY


def test_the_gate_reports_a_disabled_loop_distinctly_from_an_idle_one():
    """A transcript should say the loop is switched off, not say nothing
    happened."""
    state = should_run(False)

    assert state.state == BUSY
    assert state.source == "disabled"
    assert not state.idle


def test_the_gate_refuses_without_a_reader():
    state = should_run(True)

    assert state.state == BUSY
    assert "no idle reader" in state.detail


def test_the_gate_passes_an_idle_reading_through(monkeypatch):
    monkeypatch.setattr(idle, "_run", lambda command: SESSIONS if "ListSessions" in command[-1] else "(<true>,)")
    reader = IdleReader(cache_seconds=0.0, minimum_idle_seconds=0.0)

    assert should_run(True, reader).idle


def test_the_state_exposes_idle_as_a_readable_flag():
    assert IdleState(IDLE, SOURCE_LOGIND).idle is True
    assert IdleState(BUSY, SOURCE_LOGIND).idle is False
