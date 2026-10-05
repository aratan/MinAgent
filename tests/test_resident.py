"""Whether the loop stays inside its budget and out of the user's way.

Each test drives a whole run with a clock it owns and a reflection that only
records what it was asked to do, so nothing here waits on wall time, logind, or
a model.
"""

import asyncio
import json
import os

from minagent.idle import BUSY, IDLE, IdleState
from minagent.memory import MemoryStore
from minagent.resident import (
    DISABLED,
    ERROR,
    EXHAUSTED,
    STOPPED,
    Budget,
    ResidentWorker,
    _claim_exclusive,
    _release,
    build_worker,
)
from tests.test_memory import _memory_app


class Clock:
    """A sleep that records the wait and returns immediately.

    ``waits`` is appended synchronously on entry, so a test can count wake-ups
    exactly instead of racing an event that may or may not have been cleared
    between two turns of the loop.
    """

    def __init__(self):
        self.waits = []

    async def __call__(self, seconds):
        self.waits.append(seconds)
        await asyncio.sleep(0)


async def drive(built, clock, ticks: int):
    """Let the worker wake ``ticks`` times, then stop it and wait for it out."""
    task = asyncio.ensure_future(built.run())
    while len(clock.waits) < ticks and not task.done():
        await asyncio.sleep(0)
    built.stop()
    await task
    return task


def worker(**kwargs):
    """A worker wired to a journal, with the clock and the gate stubbed."""
    journal = kwargs.pop("journal", [])
    clock = kwargs.pop("clock", None) or Clock()
    states = kwargs.pop("states", None) or [IdleState(IDLE, "logind")]

    async def reflect():
        journal.append("worked")

    remaining = list(states)

    def gate():
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    built = ResidentWorker(
        reflect=reflect,
        budget=kwargs.pop("budget", Budget(max_cycles=3, max_model_calls=6)),
        sleep=clock,
        **kwargs,
    )
    built._gate = gate
    return built, journal, clock


# --- The budget is a hard cap ------------------------------------------------


def test_a_run_stops_at_its_cycle_cap():
    """The cap the user set is the cap that runs, however the loop feels."""
    built, journal, _ = worker()
    built.budget.max_cycles = 2
    built.budget.max_model_calls = 100

    outcome = asyncio.run(built.run())

    assert outcome == EXHAUSTED
    assert len(journal) == 2
    assert built.budget.cycles == 2


def test_the_budget_is_exhausted_even_with_calls_to_spare():
    built, journal, _ = worker()
    built.budget.max_cycles = 1
    built.budget.max_model_calls = 99

    asyncio.run(built.run())

    assert len(journal) == 1


def test_a_cycle_that_would_overdraw_the_call_cap_never_starts():
    """Starting it and discovering the overspend mid-flight is how a budget
    stops being a budget."""
    built, journal, _ = worker()
    built.budget.max_cycles = 5
    built.budget.max_model_calls = 1

    outcome = asyncio.run(built.run())

    assert outcome == EXHAUSTED
    assert journal == []
    assert "model calls" in built.detail


def test_waiting_costs_nothing(monkeypatch):
    """A machine that stayed busy for a week must not use up a run's budget
    without doing a single cycle of work."""
    busy = IdleState(BUSY, "logind", detail="someone is at the keyboard")
    built, journal, clock = worker(states=[busy])

    asyncio.run(drive(built, clock, 50))

    assert journal == []
    # The last wake-up is abandoned by the stop, which wins over the gate; the
    # claim is that none of them cost anything, not that one was wasted.
    assert built.skipped_idle >= 49
    assert built.budget.cycles == 0
    assert built.budget.model_calls == 0
    assert built.cycles == []


def test_a_long_wait_does_not_grow_the_log(monkeypatch):
    """The worker is expected to outlive a week. An entry per skipped check is
    a slow leak that only ever shows up on the host it leaked on."""
    built, _, clock = worker(states=[IdleState(BUSY, "logind")])

    asyncio.run(drive(built, clock, 200))

    assert built.skipped_idle >= 199
    assert len(built.cycles) == 0
    # And the reason is still recoverable without the log.
    assert "not free" in built.summary()
    assert built.last_gate is not None and built.last_gate.state == BUSY


def test_a_reflection_that_crashes_does_not_end_the_run():
    """A failed model call is the normal case for this loop, not an
    exceptional one. If it killed the worker, the agent would lose its
    upkeep precisely when it is least able to look after itself."""
    journal = []

    async def reflect():
        journal.append("attempt")
        if len(journal) == 1:
            raise RuntimeError("upstream refused")

    built = ResidentWorker(
        reflect=reflect,
        budget=Budget(max_cycles=3, max_model_calls=6),
        sleep=Clock(),
    )
    built._gate = lambda: IdleState(IDLE, "logind")

    asyncio.run(built.run())

    assert len(journal) == 3
    assert built.cycles[0].outcome == ERROR
    assert "upstream refused" in built.cycles[0].detail
    assert built.cycles[1].outcome == IDLE


def test_a_refund_lets_a_later_run_survive_an_early_crash():
    """Without this, one bad cycle would end every run for the rest of the day."""
    journal = []

    async def reflect():
        journal.append("attempt")
        if len(journal) == 1:
            raise RuntimeError("boom")

    built = ResidentWorker(
        reflect=reflect,
        budget=Budget(max_cycles=2, max_model_calls=4),
        sleep=Clock(),
    )
    built._gate = lambda: IdleState(IDLE, "logind")
    asyncio.run(built.run())

    # The crash still consumed its cycle; the point is the loop carried on.
    assert len(journal) == 2
    assert built.budget.cycles == 2


# --- Standing aside for the user --------------------------------------------


def test_the_worker_stands_down_while_a_turn_of_yours_is_running():
    """Idle to logind, busy to us: the user submitted a turn in the moment the
    machine went quiet. Interleaving output into a live turn corrupts it."""
    in_flight = {"value": True}
    built, journal, clock = worker(in_flight=lambda: in_flight["value"])

    asyncio.run(drive(built, clock, 20))

    assert journal == []
    assert built.skipped_in_flight >= 19
    assert built.cycles == []
    assert "a turn of yours" in built.summary()


def test_the_worker_resumes_once_your_turn_is_done():
    """Idle to logind, busy to us: the user submitted a turn in the moment the
    machine went quiet. Interleaving output into a live turn corrupts it, so
    the worker waits its turn."""
    journal = []
    clock = Clock()
    seen = {"n": 0}

    def your_turn_is_running():
        # The gate is consulted once per wake-up, so this maps one-to-one onto
        # the loop's turns: fifteen refusals, then the machine is free.
        seen["n"] += 1
        return seen["n"] <= 15

    async def reflect():
        journal.append("worked")

    built = ResidentWorker(
        reflect=reflect,
        budget=Budget(max_cycles=1, max_model_calls=2),
        sleep=clock,
        in_flight=your_turn_is_running,
    )
    built._gate = lambda: IdleState(IDLE, "logind")

    asyncio.run(drive(built, clock, 20))

    assert built.skipped_in_flight == 15
    assert journal == ["worked"]
    assert built.budget.cycles == 1
    assert built.cycles[0].outcome == IDLE


# --- Stopping ----------------------------------------------------------------


def test_stopping_ends_the_run_even_mid_wait():
    """Shutdown has to be immediate: a plain sleep would make closing the
    process take up to a full cycle, a quarter of an hour by default."""
    clock = Clock()
    journal = []

    async def reflect():
        journal.append("worked")

    built = ResidentWorker(
        reflect=reflect, cycle_seconds=900.0, sleep=clock,
        budget=Budget(max_cycles=9, max_model_calls=99),
    )
    built._gate = lambda: IdleState(IDLE, "logind")

    # A quarter of an hour of simulated wait, abandoned on the first wake-up.
    asyncio.run(drive(built, clock, 1))

    assert built.outcome == STOPPED
    assert journal == []


def test_stopping_twice_is_harmless():
    built, _, _ = worker()
    built.stop()
    built.stop()

    assert built.stopped


def test_a_disabled_loop_says_so_instead_of_pretending_to_run():
    """A transcript should be able to tell the difference between 'I did
    nothing' and 'I was not allowed to'."""
    journal = []

    async def reflect():
        journal.append("worked")

    built = ResidentWorker(reflect=reflect, enabled=False, sleep=Clock())

    assert asyncio.run(built.run()) == DISABLED
    assert journal == []
    assert "switched off" in built.detail


# --- The budget arithmetic ---------------------------------------------------


def test_a_call_is_refused_before_it_is_made_when_the_ceiling_is_reached():
    """The cap is a gate on the request, not a note about it afterwards.

    Every call is asked for one at a time and held before it is made, so a cycle
    that has spent its budget cannot spend more - it is told no, and the request
    is never made. This is the whole reason for the reservation: a cap checked
    after the fact has already let the request out, and the bill with it.
    """
    budget = Budget(max_cycles=2, max_model_calls=1)

    assert budget.reserve_call() is True
    budget.commit_call()

    assert budget.model_calls == 1
    assert budget.reserve_call() is False
    assert budget.model_calls == 1, "a refused call was charged anyway"


def test_a_held_call_counts_against_the_ceiling_while_it_is_in_flight():
    """A call in flight is already spent, so it is no longer free.

    Without this the run could reserve, commit and reserve again, and the
    ceiling would be checked against a total that ignored every request still
    outstanding - which is how N concurrent calls spend N times the cap.
    """
    budget = Budget(max_cycles=2, max_model_calls=1)

    assert budget.reserve_call() is True

    assert budget.reserved == 1
    assert budget.remaining_model_calls() == 0
    assert budget.reserve_call() is False


def test_a_call_that_was_never_made_is_given_back():
    """A hold for a request that was not sent is not a bill.

    The one that can be returned is the one that never left the process, so
    `release_call` says exactly that and `commit_call` says exactly the
    opposite. Charging only on success is a cap a flaky endpoint walks
    straight through, so the distinction is the whole contract.
    """
    budget = Budget(max_cycles=1, max_model_calls=2)

    assert budget.reserve_call() is True
    budget.release_call()

    assert budget.model_calls == 0
    assert budget.reserved == 0
    # Given back, so it can be spent - once.
    assert budget.remaining_model_calls() == 2
    assert budget.reserve_call() is True
    budget.commit_call()
    assert budget.remaining_model_calls() == 1


def test_a_ceiling_can_never_drive_a_counter_negative():
    """Settling more holds than were taken must not invent budget.

    A double commit is a bug, and the bug must not also become credit: a counter
    that clamps instead of going negative hands out calls nobody paid for.
    """
    budget = Budget(max_cycles=1, max_model_calls=2)

    budget.release_call()
    budget.release_call()
    budget.commit_call()
    budget.commit_call()

    assert budget.reserved == 0
    assert budget.model_calls == 2
    assert budget.remaining_model_calls() == 0


def test_a_cycle_has_a_ceiling_of_its_own():
    """One runaway cycle must not spend the whole run.

    The run's cap alone does not bound a single pass: a cycle that kept
    reviewing hypotheses would happily eat every remaining call and leave the
    loop with nothing for its next pass.
    """
    budget = Budget(max_cycles=4, max_model_calls=4 * 6, max_calls_per_cycle=2)
    budget.start_cycle()

    for _ in range(2):
        assert budget.reserve_call() is True
        budget.commit_call()

    assert budget.reserve_call() is False
    assert budget.model_calls == 2
    # The run is untouched: the next cycle gets its own envelope.
    assert budget.remaining_model_calls() == 4 * 6 - 2
    budget.start_cycle()
    assert budget.remaining_in_cycle() == 2


def test_a_cycle_that_cannot_afford_a_review_is_not_started():
    """Reflection without review is a request spent to learn nothing.

    A cycle needs the reflection and one review of what it found. Starting a
    cycle that cannot pay for both means reflecting, finding itself unable to
    judge the result, and discarding it - with the refusals filed in the
    admission log as though they were judgements on merit.
    """
    budget = Budget(max_cycles=1, max_model_calls=1)

    assert budget.can_start_cycle() is False

    enough = Budget(max_cycles=1, max_model_calls=2)
    assert enough.can_start_cycle() is True


def test_a_cancelled_cycle_gives_back_its_slot_but_not_its_calls():
    """A cut-short pass costs the calls it made; only the slot comes back.

    Returning the calls too would let a shutdown walk the cap up: every cycle
    interrupted would be free, and the cap would measure only the cycles that
    finished.
    """
    budget = Budget(max_cycles=1, max_model_calls=4)
    budget.start_cycle()
    assert budget.reserve_call() is True
    budget.commit_call()

    budget.cancel_cycle()

    assert budget.remaining_cycles() == 1
    assert budget.model_calls == 1


def test_a_stopped_run_refuses_every_further_call():
    """Stopping the loop must also stop its spending.

    Otherwise a cycle already in flight asks the endpoint for one more call on
    its way down, and the budget says yes because it was never told.
    """
    built = build_worker(FakeAgent(), FakeConfig())
    assert built.budget.reserve_call() is True
    built.budget.release_call()

    built.stop()

    # Stopped through the worker, not by reaching for the budget directly: a
    # `stop()` that forgot to tell the budget would still pass a test that only
    # ever holds a `Budget`.
    assert built.budget.reserve_call() is False
    assert built.budget.can_start_cycle() is False


def test_the_interval_is_the_configured_one():
    """Fifteen minutes by default: short enough that a day's upkeep is
    plausible, long enough not to hammer the provider."""
    built, _, clock = worker(cycle_seconds=900.0)

    asyncio.run(drive(built, clock, 1))

    assert clock.waits[0] == 900.0


async def test_a_reflection_with_no_budget_left_is_never_sent(tmp_path):
    """The refusal happens before the request, at the level that asks for it.

    The budget is a unit: refusing a call is its own job and is tested as one.
    What is not obvious - and what a caller can quietly skip - is whether the
    code that wants a reflection actually asks the budget first. This is the
    count of requests that left, which is the only number that matters.
    """
    app = _memory_app(tmp_path)
    app.improvement_enabled = True
    app.improvement_auto = False
    app.application_root = str(tmp_path)
    app.memory_store = MemoryStore(str(tmp_path / ".minagent" / "memory.db"))
    await app.memory_store.initialize()
    await app.memory_store.remember(
        "procedure",
        "Regla previa",
        "Conviene archivar las capturas viejas antes de que el indice se degrade.",
        ["regla"],
    )
    (tmp_path / ".env").write_text("MEMORY_REFLECTION_INTERVAL=10\n")

    class _Counter:
        def __init__(self):
            self.calls = 0

        async def complete(self, messages, options=None):
            self.calls += 1
            return {"message": {"role": "assistant", "content": "{}"}}

    counter = _Counter()
    app.open_ai_client = counter
    built = build_worker(app, FakeConfig(), sleep=Clock())
    app.resident_worker = built

    # The previous work in this cycle spent both calls it was allowed. Whatever
    # the app wants now, it is not going to get it.
    built.budget.start_cycle()
    for _ in range(built.budget.max_calls_per_cycle):
        assert built.budget.reserve_call() is True
        built.budget.commit_call()
    assert built.budget.remaining_in_cycle() == 0

    await app.reflect_on_session("idle")

    assert counter.calls == 0, "a reflection went out with no budget left"
    assert built.budget.model_calls == 2, "a refused reflection was charged anyway"


async def test_a_cycle_that_runs_out_of_budget_stops_calling_the_model(tmp_path, monkeypatch):
    """The ceiling holds against a model that keeps having ideas.

    The reflection model may return up to twelve hypotheses and the reviewer is
    one call each, so a cycle capped below that has to say no partway through.
    The proof is the request count: the stub counts what actually left, so a
    cap enforced after the fact cannot pass this.
    """
    app = _memory_app(tmp_path)
    app.improvement_enabled = True
    # Auto off, so this is only about the ceiling and not about a trial.
    app.improvement_auto = False
    app.application_root = str(tmp_path)
    monkeypatch.setenv("MEMORY_REFLECTION_INTERVAL", "10")
    (tmp_path / ".env").write_text("MEMORY_REFLECTION_INTERVAL=10\n")
    app.memory_store = MemoryStore(str(tmp_path / ".minagent" / "memory.db"))
    await app.memory_store.initialize()
    # One unrelated lesson in context, so the consistency reviewer is genuinely
    # consulted per hypothesis instead of being answered by an empty store. A
    # reviewer that is never asked is how a ceiling looks like it was never
    # reached: the reflection alone fits inside two calls.
    await app.memory_store.remember(
        "procedure",
        "Regla previa",
        "Cuando el disco se llena de capturas conviene archivarlas antes de que el "
        "indice se degrade y volverlas a numerar despues del corte.",
        ["regla"],
    )

    class _Ideas:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, options=None):
            self.calls += 1
            asked = " ".join(str(m.get("content", "")) for m in messages)
            if "verdict" in asked:
                return {
                    "message": {
                        "role": "assistant",
                        "content": '{"verdict": "valid", "reason": "ok"}',
                    }
                }
            body = json.dumps(
                {
                    "hypotheses": [
                        {
                            "title": f"La cola se atasca {index}",
                            "kind": "improvement",
                            "target": "agent",
                            "statement": (
                                f"Acotar la cola de cómputo al hypothesis {index} evita que "
                                "la memoria de reintentos crezca cuando se acumulan tareas."
                            ),
                            "evidence": "3 rechazos de VRAM en la última hora",
                            "expected": "Menos rechazos por saturación",
                            "verify": "Que la cola no supere dos en espera",
                            "setting": "MEMORY_REFLECTION_INTERVAL",
                            "value": str(index % 9 + 1),
                            "reason": "porque la cola se atasca",
                        }
                        for index in range(12)
                    ]
                }
            )
            return {"message": {"role": "assistant", "content": body}}

    ideas = _Ideas()
    app.open_ai_client = ideas

    class _Tight(FakeConfig):
        improvement_max_cycles = 1
        improvement_model_calls_per_cycle = 2

    built = build_worker(app, _Tight(), sleep=Clock())
    built._gate = lambda: IdleState(IDLE, "logind")
    app.resident_worker = built

    await built.run()

    # One reflection and one review. Twelve hypotheses wanted fourteen calls and
    # got two: the ceiling is the reason the rest were never asked about.
    assert ideas.calls == 2
    assert built.budget.model_calls == 2
    assert built.budget.remaining_model_calls() == 0
    # And the cycle says so out loud, instead of dropping the rest in silence.
    assert "unexamined" in built.cycles[-1].detail
    assert built.outcome == EXHAUSTED
    # Only the hypothesis that was actually reviewed was kept.
    stored = await app.memory_store.recent(limit=20)
    assert len([row for row in stored if row["kind"] == "hypothesis"]) == 1


async def test_a_cycle_that_runs_out_of_budget_is_not_a_stuck_cycle(tmp_path, monkeypatch):
    """Refusing the eleventh call is not the same as failing the tenth.

    A crash in the reflection is an ERROR and leaves the run going. Running out
    of money is neither a crash nor a reason to stop early: the run is finished
    either way, so it reports the honest reason - the bill - rather than a fault
    that never happened.
    """
    app = _memory_app(tmp_path)
    app.improvement_enabled = True
    app.improvement_auto = False
    app.application_root = str(tmp_path)
    app.memory_store = MemoryStore(str(tmp_path / ".minagent" / "memory.db"))
    await app.memory_store.initialize()

    class _Boom:
        async def complete(self, messages, options=None):
            raise OSError("endpoint is down")

    app.open_ai_client = _Boom()

    class _Tight(FakeConfig):
        improvement_max_cycles = 4
        improvement_model_calls_per_cycle = 1

    built = build_worker(app, _Tight(), sleep=Clock())
    built._gate = lambda: IdleState(IDLE, "logind")
    app.resident_worker = built

    await built.run()

    # Every cycle dispatched its one call, and every one of them failed. They
    # were still charged, so the run stopped at four - a flaky endpoint does not
    # get to spend the budget for free.
    assert built.budget.model_calls == 4
    assert built.outcome == EXHAUSTED
    assert "ran 4 of 4 cycles" in built.detail
    assert all(cycle.charged == 1 for cycle in built.cycles)


# --- Wiring ------------------------------------------------------------------


class FakeConfig:
    improvement_autonomous = True
    improvement_cycle_seconds = 60.0
    improvement_max_cycles = 3
    improvement_idle_seconds = 45.0


class GenerousConfig(FakeConfig):
    improvement_model_calls_per_cycle = 8


def test_the_builder_reads_the_settings_and_guards_the_two_hard_caps():
    agent = FakeAgent()
    built = build_worker(agent, FakeConfig())

    assert built.enabled is True
    assert built.cycle_seconds == 60.0
    assert built.budget.max_cycles == 3
    assert built.idle_seconds == 45.0
    # No calls-per-cycle given, so the floor of two is the per-cycle ceiling and
    # three cycles cannot become more than six requests between them.
    assert built.budget.max_calls_per_cycle == 2
    assert built.budget.max_model_calls == 6


def test_the_builder_reads_the_calls_per_cycle_setting():
    """The setting that was parsed and then never read.

    A number in the README describing a cap that was never in force is worse
    than no number at all, so the setting is read and the run's ceiling follows
    from it.
    """
    built = build_worker(FakeAgent(), GenerousConfig())

    assert built.budget.max_calls_per_cycle == 8
    assert built.budget.max_model_calls == 3 * 8


def test_a_ceiling_too_small_to_review_anything_is_raised_to_the_floor():
    """A config asking for less than a viable cycle gets a cycle that can run.

    The floor is not a permission slip: it is two calls, which is the least that
    reflects anything and judges what it reflected. Anything lower produces a
    loop that starts and can only ever learn that it may not run.
    """
    class Impossible(FakeConfig):
        improvement_model_calls_per_cycle = 1

    built = build_worker(FakeAgent(), Impossible())

    assert built.budget.max_calls_per_cycle == 2
    assert built.budget.can_start_cycle() is True


class FakeAgent:
    _active_request_in_flight = False

    def __init__(self):
        self.reasons = []

    async def reflect_on_session(self, reason):
        self.reasons.append(reason)
        return ""


def test_the_loop_says_out_loud_what_it_did_and_what_it_skipped():
    """A resident loop nobody watches has to leave a trace.

    Counted skips are not a report: a service that runs all night with an empty
    journal is indistinguishable from one that spent the night asleep, and only
    one of them is the product working. The line per cycle answers "did it run
    and what did it find"; the line per skip answers "why not", which is the
    question that has no other answer.
    """
    lines: list[str] = []
    built = build_worker(FakeAgent(), FakeConfig(), sleep=Clock())
    built.report = lines.append
    built._gate = lambda: IdleState(IDLE, "logind")

    asyncio.run(built.run())

    assert len(lines) == built.budget.max_cycles
    assert all(line.startswith(f"cycle {index}:") for index, line in enumerate(lines))
    assert "nothing to report" in lines[0] or "model call(s)" in lines[0]


def test_a_busy_machine_is_reported_rather_than_only_counted():
    lines: list[str] = []
    clock = Clock()
    built = build_worker(FakeAgent(), FakeConfig(), sleep=clock)
    built.report = lines.append
    built._gate = lambda: IdleState(BUSY, "logind", 0.0, "logind reports recent user input")

    # A gate that never opens never spends the budget, so the loop only ends
    # when the test stops it: `drive`, not `run`.
    asyncio.run(drive(built, clock, 5))

    assert built.cycles == []
    assert lines
    assert all(line == "skipped: logind reports recent user input" for line in lines)


def test_a_cycle_keeps_what_the_reflection_said():
    """The reflection's answer reaches the cycle that paid for it.

    It used to be discarded at the `await`: the return value was never bound,
    so a cycle recorded "idle" and nothing else, and the only thing a reader
    could learn from a night of work was what research had queued.
    """

    class Talkative(FakeAgent):
        async def reflect_on_session(self, reason):
            await super().reflect_on_session(reason)
            return "12 hipótesis; propuso REVIEW_EVERY_TURNS=2"

    built = build_worker(Talkative(), FakeConfig(), sleep=Clock())
    built._gate = lambda: IdleState(IDLE, "logind")

    asyncio.run(built.run())

    assert all("12 hipótesis" in cycle.detail for cycle in built.cycles)
    # And it composes rather than replaces: the budget notice is about work that
    # did not happen, and must not push out the finding that did.
    assert built.cycles[-1].detail.startswith("12 hipótesis")





def test_the_builder_reflects_with_a_reason_the_transcript_can_show():
    agent = FakeAgent()
    built = build_worker(agent, FakeConfig(), sleep=Clock())
    built._gate = lambda: IdleState(IDLE, "logind")

    asyncio.run(built.run())

    # Every cycle is attributable, and the count is the budget's, not the test's.
    assert agent.reasons == ["idle", "idle", "idle"]
    assert len(agent.reasons) == built.budget.max_cycles


def test_a_cycle_reports_what_it_spent_not_what_it_was_allowed_to_spend():
    """`charged` is the bill, and it is a count of dispatches.

    A transcript that shows the ceiling tells you nothing about the cost, and
    this is the field that answers "what did the night spend". A worker that
    reserves and spends nothing reports zero - which is the truth, and a good
    reason to look twice.
    """
    class Spender:
        _active_request_in_flight = False

        def __init__(self, calls: int) -> None:
            self.left = calls
            self.reasons = []

        async def reflect_on_session(self, reason):
            self.reasons.append(reason)
            self.left -= 1

    agent = Spender(0)
    built = build_worker(agent, FakeConfig(), sleep=Clock())
    built._gate = lambda: IdleState(IDLE, "logind")

    asyncio.run(built.run())

    assert [cycle.charged for cycle in built.cycles] == [0, 0, 0]
    assert built.budget.model_calls == 0


def test_the_builder_leaves_autonomy_off_when_the_config_does():
    class Off(FakeConfig):
        improvement_autonomous = False

    built = build_worker(FakeAgent(), Off())

    assert built.enabled is False


def test_the_builder_notices_a_turn_in_flight():
    class Busy(FakeAgent):
        _active_request_in_flight = True

    assert build_worker(Busy(), FakeConfig()).in_flight() is True
    assert build_worker(FakeAgent(), FakeConfig()).in_flight() is False


# --- One loop per project, across processes ---------------------------------


def test_a_lock_is_taken_by_one_process_and_refused_by_the_next(tmp_path):
    """Two loops would each stay inside their own budget and spend twice."""
    lock = str(tmp_path / ".minagent" / "resident.lock")

    held, reason, handle = _claim_exclusive(lock)
    try:
        assert held and reason == ""
        again, why, second_handle = _claim_exclusive(lock)
        assert again is False
        assert "resident.lock" in why
        assert second_handle is None
    finally:
        _release(handle)


def test_the_lock_file_names_the_pid_that_holds_it(tmp_path):
    """An operator reading the file has to know whose loop this is."""
    lock = tmp_path / "resident.lock"
    held, _, handle = _claim_exclusive(str(lock))
    try:
        assert held
        assert str(os.getpid()) in lock.read_text(encoding="utf-8")
    finally:
        _release(handle)


def test_a_released_lock_can_be_taken_again(tmp_path):
    lock = str(tmp_path / "resident.lock")
    held, _, handle = _claim_exclusive(lock)
    assert held
    _release(handle)
    assert _claim_exclusive(lock)[0] is True


async def test_a_run_holds_the_lock_while_it_works_and_lets_it_go_after(tmp_path):
    lock = str(tmp_path / "resident.lock")
    built, _, clock = worker(lock_path=lock, cycle_seconds=900.0)
    states = []

    original = built._run_locked

    async def watched():
        states.append(("during", _claim_exclusive(lock)[0]))
        return await original()

    built._run_locked = watched
    await drive(built, clock, 1)

    assert states == [("during", False)], "the run itself should hold the lock"
    assert _claim_exclusive(lock)[0] is True


async def test_a_second_worker_does_nothing_at_all_while_one_is_running(tmp_path):
    """Not a slower loop: no cycle, no reflection, no call."""
    lock = str(tmp_path / "resident.lock")
    held, _, handle = _claim_exclusive(lock)
    try:
        built, journal, clock = worker(lock_path=lock, cycle_seconds=0.01)
        outcome = await asyncio.wait_for(built.run(), timeout=5)
        assert outcome == DISABLED
        assert "already holds" in built.detail
        assert journal == []
        assert clock.waits == []
    finally:
        _release(handle)


async def test_no_lock_path_means_no_lock_and_no_complaint():
    """A test or a one-shot run is not a second agent; it should just run."""
    built, journal, clock = worker(cycle_seconds=0.01)
    await drive(built, clock, 2)
    # One wake is enough: the point is that it ran at all, without a lock to hold.
    assert journal == ["worked"]
    assert built.outcome != DISABLED


async def test_a_lock_file_that_cannot_be_created_runs_unlocked_and_says_so(tmp_path):
    """A read-only checkout must not silence the loop; two loops are the real risk."""
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    built, journal, clock = worker(lock_path=str(blocked / "resident.lock"), cycle_seconds=0.01)
    await drive(built, clock, 2)
    assert journal == ["worked"]
    assert built.outcome != DISABLED
