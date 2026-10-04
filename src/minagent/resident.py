"""The loop that keeps working while you do not need the machine.

Everything in this module exists to make one claim true: when MinAgent decides
to work on its own, you can always tell what it did and how to stop it. Three
properties carry that claim, and dropping any one of them breaks it:

- it only runs when logind says nobody is at the keyboard,
- it stands down the moment a turn of yours is in flight, and
- every run has a budget it cannot exceed, even if the code below is wrong.

The first two are about not taking what belongs to you. The third is about not
trusting the code below. A budget that lives in the caller is a budget the
caller can raise, so it lives here, next to the thing it limits.

The worker deliberately does not touch the interactive editor. It receives a
callable and calls it, which is what lets the same loop run from a terminal
session today and from a systemd service tomorrow without changing a line.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .idle import IdleReader, IdleState, should_run

LOCK_FILENAME = "resident.lock"


def _claim_exclusive(path: str) -> tuple[bool, str, Any]:
    """Take an exclusive lock for this process, or say who is holding it.

    Returns ``(held, reason, handle)``. An empty path means no lock was asked
    for, which is the test path and not a failure. A lock file that cannot be
    created is also not treated as a refusal: refusing there would turn a
    read-only checkout into a machine that quietly never learns again, and the
    thing worth stopping is two loops, not one.
    """
    if not path:
        return True, "", None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the run
    except OSError as error:
        return True, f"lock unavailable, running unlocked: {error}", None
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False, f"another improvement loop already holds {os.path.basename(path)}", None
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
    except OSError:
        # The lock is taken; failing to note the pid in it is not a reason to
        # give it up and let a second loop in.
        pass
    return True, "", handle


def _release(handle: Any) -> None:
    if handle is None:
        return
    with contextlib.suppress(OSError):
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    with contextlib.suppress(OSError):
        handle.close()

#: The fewest calls that make a cycle worth starting: one for the reflection and
#: one for the reviewer of the single hypothesis it produced.
#:
#: This is deliberately *not* the worst case. A cycle's real ceiling is
#: 2 + ``MAX_HYPOTHESES``, because the reviewer is asked once per hypothesis
#: the screen did not already settle, and charging that up front against the
#: default budget of six would mean no cycle could ever start. So the ceiling is
#: enforced per call as it is made, and this is only the floor below which
#: starting is pointless: a cycle that cannot afford a reflection *and* one
#: review would reflect, find itself unable to judge what it found, and discard
#: the findings - a request spent to learn nothing, leaving refusals filed in
#: the admission log as though they were judgements on merit.
MIN_CALLS_PER_CYCLE = 2

#: A complete investigation costs two requests: one to state the claim, one to
#: ask whether it contradicts what Ara already believes. The second decides, so
#: a cycle that cannot pay for both cannot produce a promotable finding.
RESEARCH_MIN_CALLS = 2

#: How many research passes one cycle may fund.
#:
#: This is a floor on the budget, not a substitute for it: the loop still stops
#: when :meth:`Budget.can_fund_research` says no, so a cycle with room for two
#: runs two. It exists because the number of passes a *cycle* may make is not
#: the same question as how many calls are left, and a loop that only reads the
#: latter would let one quiet stretch run six searches while nothing was watching.
#: The budget is what is spent; this is what one cycle may attempt.
MAX_RESEARCH_PASSES_PER_CYCLE = 3

#: What the loop is doing, for a transcript. A user reading a log at 2am should
#: be able to tell the difference between "I did nothing" and "I was not
#: allowed to".
IDLE = "idle"
BUSY = "busy"
DONE = "done"
STOPPED = "stopped"
EXHAUSTED = "exhausted"
DISABLED = "disabled"
ERROR = "error"

FOREVER = float("inf")


@dataclass
class Budget:
    """Hard caps for one run, held before each call and charged after it.

    The counters are conserved: ``max_model_calls`` is ``committed + reserved +
    free``, where free is whatever is left. A call is admitted by
    :meth:`reserve_call` and settled by :meth:`commit_call`, and that pair is
    the auth/capture pattern payments use. It is here for one reason: a cap
    checked only *after* a request has gone out is an accounting report, not a
    cap. Reserving first means the over-budget path refuses mechanically and
    the request is never made.

    Two levels, because the settings say two different things.
    ``max_model_calls`` bounds the whole run; ``max_calls_per_cycle`` bounds one
    cycle. Without the second, a single cycle that ran away with the reviewer
    would eat the entire run and the loop would stop on its first useful pass.

    ``refund`` is gone and its test went with it. Returning a dispatched call's
    cost made the cap walkable by any slow shutdown, which is the one thing a
    cap exists to prevent; :meth:`cancel_cycle` gives back the *slot* and keeps
    the calls.
    """

    max_cycles: int = 4
    max_model_calls: int = 24
    max_calls_per_cycle: int = 6
    cycles: int = 0
    model_calls: int = 0
    reserved: int = 0
    _cycle_calls: int = 0
    _cycle_reserved: int = 0
    live: bool = True

    def remaining_cycles(self) -> int:
        return max(0, self.max_cycles - self.cycles)

    def remaining_model_calls(self) -> int:
        """Free calls for the run. Calls in flight count against it."""
        return max(0, self.max_model_calls - self.model_calls - self.reserved)

    def remaining_in_cycle(self) -> int:
        """Free calls for the cycle in progress, once one has started."""
        return max(0, self.max_calls_per_cycle - self._cycle_calls - self._cycle_reserved)

    def can_start_cycle(self, cost: int = MIN_CALLS_PER_CYCLE) -> bool:
        """Whether a cycle started now can do anything at all.

        A run-level floor as well as a cycle-level one: a cycle that starts with
        no room left cannot be productive, and starting it to discover that is a
        request spent to learn it was not allowed to run.
        """
        return (
            self.live
            and self.remaining_cycles() >= 1
            and self.remaining_model_calls() >= cost
            and self.max_calls_per_cycle >= cost
        )

    def start_cycle(self) -> None:
        """Take the slot and hand the cycle a fresh envelope of its own."""
        self.cycles += 1
        self._cycle_calls = 0
        self._cycle_reserved = 0

    def can_fund_research(self, cost: int = RESEARCH_MIN_CALLS) -> bool:
        """Whether this cycle can still pay for a whole investigation.

        Research is funded from the same envelope as the reflection, and only
        from what the reflection left. That is the whole of the policy, and it
        is derived rather than chosen: :attr:`remaining_in_cycle` already tracks
        the remainder, the run is meant to reflect, and a second budget here
        would be the one thing :class:`Budget` exists to prevent - a cap
        enforced somewhere other than where the calls are reserved.

        The check is for the *whole* investigation, not for its first call. A
        finding that cannot be consistency-checked is a finding that may not be
        promoted, so starting one and abandoning it halfway spends the calls
        without ever producing the thing they were for. Two is the floor
        because extraction and consistency are two separate requests, and the
        second is the one that decides.

        This is a pre-check, not the cap. The calls themselves are still
        reserved one at a time by the caller; this only avoids beginning work
        that is already unaffordable.
        """
        return self.live and self.remaining_in_cycle() >= cost

    def cancel_cycle(self) -> None:
        """Give back the slot of a cycle that was cut short.

        Only the slot. A model call that was dispatched was paid for whatever
        came back, so returning its cost would let a shutdown walk the cap up.
        """
        self.cycles = max(0, self.cycles - 1)

    def reserve_call(self) -> bool:
        """Hold one call against both ceilings. ``False`` means do not make it.

        Fails closed on every path: stopped, over the run's cap, or over the
        cycle's. The caller has to ask, and a caller that does not ask is a bug
        this cannot catch - which is the same reason the cap lives here and not
        in the configuration.
        """
        if not self.live:
            return False
        if self.remaining_model_calls() < 1 or self.remaining_in_cycle() < 1:
            return False
        self.reserved += 1
        self._cycle_reserved += 1
        return True

    def commit_call(self) -> None:
        """Settle a hold against what it cost. Always charges.

        A request that was dispatched is charged even when it failed, timed out,
        or returned nothing useful: the provider billed it either way. Charging
        here is what makes ``model_calls`` a count of money spent rather than a
        count of answers received.
        """
        if self.reserved > 0:
            self.reserved -= 1
            self._cycle_reserved = max(0, self._cycle_reserved - 1)
        self.model_calls += 1
        self._cycle_calls += 1

    def release_call(self) -> None:
        """Return a hold for a call that was never made.

        Distinct from :meth:`commit_call` because the two answer different
        questions: this one says we decided not to spend, and the other says we
        spent it. Conflating them is how a budget ends up paying for requests
        that never left the process.
        """
        if self.reserved > 0:
            self.reserved -= 1
            self._cycle_reserved = max(0, self._cycle_reserved - 1)

    def kill(self) -> None:
        """Refuse every further call. The run is over."""
        self.live = False

    @property
    def exhausted(self) -> bool:
        return self.remaining_cycles() == 0 or self.remaining_model_calls() == 0


@dataclass
class Cycle:
    """One pass, recorded well enough to be audited after the fact.

    ``charged`` is what the cycle actually spent, not what it was allowed to
    spend. A transcript that shows the ceiling tells you nothing about the bill,
    and this is the field that answers "what did the night cost".
    """

    index: int
    outcome: str
    detail: str = ""
    charged: int = 0


@dataclass
class ResidentWorker:
    """Runs ``reflect`` on a cycle, while the machine is free.

    Every dependency is a callable passed in, so a test can drive a whole run
    with a clock it controls and a reflection that records what it was asked,
    and never touch the terminal or the network.

    Only cycles that actually ran are appended to :attr:`cycles`. A machine that
    stays busy for a week produces a hundred skipped checks an hour, and a
    resident process is expected to outlive the week, so the skips are counted
    rather than listed - an unbounded log in a process that never exits is a
    slow leak that only shows up on the host it has been leaking on.
    """

    reflect: Callable[[], Awaitable[Any]]
    #: Optional second pass over the same envelope, run after the reflection.
    #: Returns a line for the cycle record, or an empty string for "nothing to
    #: ask". Kept optional so a host that wants the reflection alone is not made
    #: to supply a research callable it will never use.
    research: Callable[[], Awaitable[str]] | None = None
    enabled: bool = True
    #: Where the cross-process lock lives. Empty means no lock, which is what a
    #: test or a one-shot run wants; a live loop on a real project must set it.
    lock_path: str = ""
    _lock_handle: Any = None
    cycle_seconds: float = 900.0
    budget: Budget = field(default_factory=Budget)
    idle_reader: IdleReader | None = None
    idle_seconds: float = 120.0
    in_flight: Callable[[], bool] = lambda: False
    sleep: Callable[[float], Awaitable[None]] | None = None
    cycles: list[Cycle] = field(default_factory=list)
    outcome: str = ""
    detail: str = ""
    skipped_idle: int = 0
    skipped_in_flight: int = 0
    last_gate: IdleState | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def stop(self) -> None:
        """Ask the loop to finish. Safe to call from anywhere, including twice.

        The budget dies with it. A loop that has been asked to stop must not
        start another model call on its way down, and the caller of a cycle is
        the one place that would otherwise be free to ask for one.
        """
        self._stop.set()
        self.budget.kill()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def summary(self) -> str:
        """One line a transcript can show, answering 'why did it not run?'."""
        if self.skipped_in_flight:
            return f"stood aside {self.skipped_in_flight} time(s) for a turn of yours"
        if self.last_gate is not None and not self.last_gate.idle:
            return self.last_gate.detail or "the machine was not free"
        if self.cycles:
            return f"ran {len(self.cycles)} cycle(s)"
        return "never ran"

    async def _wait(self, seconds: float) -> None:
        """Sleep, but wake up the moment we are asked to stop.

        A plain ``asyncio.sleep`` here would make shutdown take up to a full
        cycle, which for the default interval is a quarter of an hour of a
        process the user is trying to close. The injected ``sleep`` is held to
        the same contract so a test clock cannot make the worker unstoppable
        either.
        """
        waiter = self.sleep(seconds) if self.sleep is not None else self._stop.wait()
        if self.sleep is None:
            # Waiting on the event directly: it is already the interrupt.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(waiter, timeout=seconds)
            return
        sleeping = asyncio.ensure_future(waiter)
        stopping = asyncio.ensure_future(self._stop.wait())
        try:
            await asyncio.wait(
                {sleeping, stopping}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in (sleeping, stopping):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def run(self) -> str:
        """Cycle until stopped or out of budget. Returns why it finished."""
        if not self.enabled:
            self.outcome, self.detail = DISABLED, "autonomous improvement is switched off"
            return self.outcome

        # One loop per project, across processes. The budget caps a single run
        # very well and says nothing about a second one: an interactive session
        # and a resident service both read the same .env, and two loops would
        # spend roughly twice for half the attention. The lock is what makes the
        # cap mean what it says, and it is held for the whole run rather than
        # re-taken per cycle, so two of them cannot interleave either.
        held, reason, handle = _claim_exclusive(self.lock_path)
        if not held:
            self.outcome, self.detail = DISABLED, reason
            return self.outcome
        self._lock_handle = handle
        try:
            return await self._run_locked()
        finally:
            _release(self._lock_handle)
            self._lock_handle = None

    async def _run_locked(self) -> str:
        """The loop itself, with the cross-process lock already held."""
        while not self.stopped:
            if not self.budget.can_start_cycle():
                self.outcome, self.detail = EXHAUSTED, _exhausted_detail(self.budget)
                break

            await self._wait(self.cycle_seconds)
            if self.stopped:
                break

            state = self._gate()
            self.last_gate = state
            if not state.idle:
                # Not allowed to work, and deliberately not charged and not
                # listed: waiting is free, and charging for it would let a
                # machine that stayed busy for a week exhaust the budget
                # without a single cycle of work.
                self.skipped_idle += 1
                continue

            if self.in_flight():
                # Idle to logind, busy to us. The user submitted a turn in the
                # moment the machine went quiet, or a long tool call is still
                # running. Standing down costs a cycle of delay, which is the
                # cheapest possible error.
                self.skipped_in_flight += 1
                continue

            self.budget.start_cycle()
            spent_before = self.budget.model_calls
            self.cycles.append(
                Cycle(
                    index=len(self.cycles),
                    outcome=IDLE,
                )
            )
            try:
                await self.reflect()
            except asyncio.CancelledError:
                self.budget.cancel_cycle()
                raise
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not end the run
                # Recorded and survived. A reflection that crashes is the normal
                # case for a model call, not an exceptional one, and letting it
                # kill the loop would mean the loop dies exactly when the agent
                # is least able to look after itself.
                self.detail = f"{type(exc).__name__}: {exc}"
                self.cycles[-1].outcome, self.cycles[-1].detail = ERROR, self.detail

            # Research runs after the reflection and out of what it left, so a
            # cycle that reflected well is not also charged for going looking.
            # It shares the reflection's error handling on purpose: a research
            # pass that raises is the same kind of event as a reflection that
            # raises, and neither is a reason to stop a loop nobody is watching.
            #
            # Repeated while the budget still affords a whole investigation,
            # rather than once per cycle: a pass that returns a note and leaves
            # budget behind is a pass being asked to stop for no reason, and the
            # budget is the only thing here entitled to say when to stop.
            notes: list[str] = []
            passes = 0
            while (
                self.research is not None
                and self.cycles[-1].outcome != ERROR
                and passes < MAX_RESEARCH_PASSES_PER_CYCLE
                and self.budget.can_fund_research()
            ):
                passes += 1
                try:
                    note = (await self.research()).strip()
                except asyncio.CancelledError:
                    self.budget.cancel_cycle()
                    raise
                except Exception as exc:  # noqa: BLE001
                    note = f"research failed: {type(exc).__name__}: {exc}"
                if note:
                    notes.append(note)
                if not note:
                    # Nothing queued, nothing worth asking: the loop has run dry
                    # and another pass would only spend another cycle's look.
                    break
            if notes:
                joined = "; ".join(notes)
                self.cycles[-1].detail = (
                    f"{self.cycles[-1].detail}; {joined}" if self.cycles[-1].detail else joined
                )
            self.cycles[-1].charged = self.budget.model_calls - spent_before
            if self.budget.remaining_in_cycle() == 0 and not self.cycles[-1].detail:
                # Said out loud because the alternative is a night that produced
                # fewer findings than it looks like it should have, with no
                # record of why. The reflection model is allowed to return up to
                # twelve hypotheses and the reviewer is one call each, so a
                # cycle capped below that reviews only as many as it can afford
                # - and says so here rather than dropping the rest in silence.
                self.cycles[-1].detail = "out of cycle budget; the rest went unexamined"
        else:
            self.outcome, self.detail = STOPPED, "stopped on request"

        if not self.outcome:
            self.outcome, self.detail = STOPPED, "stopped on request"
        return self.outcome

    def _gate(self) -> IdleState:
        return should_run(True, self.idle_reader, minimum_idle_seconds=self.idle_seconds)

def _exhausted_detail(budget: Budget) -> str:
    if budget.remaining_cycles() == 0:
        return f"ran {budget.cycles} of {budget.max_cycles} cycles"
    return (
        f"{budget.model_calls} of {budget.max_model_calls} model calls spent"
    )


def build_worker(
    agent: Any,
    config: Any,
    *,
    idle_reader: IdleReader | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> ResidentWorker:
    """Wire a worker to a live agent and its configuration.

    Kept apart from :class:`ResidentWorker` so the loop has no idea what a
    ``MinAgent`` is, and so a future non-interactive entrypoint can build the
    same worker from a different host without this module importing a terminal.
    """
    application_root = str(getattr(agent, "application_root", "") or "")
    return ResidentWorker(
        reflect=lambda: agent.reflect_on_session("idle"),
        # The research hook, wired. It was left optional so a host that only
        # wanted the reflection was not made to supply a callable it would never
        # use; this is that host, and the call it has been waiting for.
        research=lambda: agent.run_research_pass(),
        enabled=bool(getattr(config, "improvement_autonomous", False)),
        # One loop per project. An interactive session and a resident service
        # read the same configuration and would otherwise both work, each
        # inside its own budget and neither knowing about the other. A host
        # that names no project gets no lock rather than a lock on whatever
        # directory it happened to be started in.
        lock_path=os.path.join(application_root, ".minagent", LOCK_FILENAME) if application_root else "",
        cycle_seconds=float(getattr(config, "improvement_cycle_seconds", 900.0) or 900.0),
        budget=_budget_from(config),
        idle_reader=idle_reader or IdleReader(
            cache_seconds=30.0,
            minimum_idle_seconds=float(
                getattr(config, "improvement_idle_seconds", 120.0) or 120.0
            ),
        ),
        idle_seconds=float(getattr(config, "improvement_idle_seconds", 120.0) or 120.0),
        in_flight=lambda: bool(getattr(agent, "_active_request_in_flight", False)),
        sleep=sleep,
    )


def _budget_from(config: Any) -> Budget:
    """Build the run's envelope from the settings that describe it.

    ``IMPROVEMENT_MODEL_CALLS_PER_CYCLE`` was parsed, validated and documented,
    and then never read: the ceiling was ``max_cycles * 2`` with the two written
    into this module. A setting that parses and does nothing is worse than one
    that is absent, because the number in the README then describes a cap that
    was never in force. It is read here, and the floor keeps a config asking
    for less than a viable cycle from producing a loop that can never run.
    """
    cycles = int(getattr(config, "improvement_max_cycles", 4) or 4)
    per_cycle = int(getattr(config, "improvement_model_calls_per_cycle", 0) or 0)
    if per_cycle < MIN_CALLS_PER_CYCLE:
        per_cycle = MIN_CALLS_PER_CYCLE
    return Budget(
        max_cycles=cycles,
        max_calls_per_cycle=per_cycle,
        max_model_calls=cycles * per_cycle,
    )
