"""Measure whether a change the agent made actually helped, and undo it if not.

This is the fourth phase of the loop, and the only one that makes the other
three mean anything. Without it the agent proposes a change, applies it, and
never finds out whether it was right - which is not a loop, it is a sequence of
unverified edits with a growing pile of settings nobody chose.

What can honestly be measured here is deliberately narrow:

* **Failures per turn** - tool errors, jobs refused for lack of VRAM, and jobs
  that failed. This is the primary number, because it is the one that maps onto
  the user actually being blocked.
* **Tool-result tokens per turn** - the context the agent spends having the
  tools it chose to use. The prompt tokens are deliberately *not* here: they
  mostly track how long the conversation is, which is not something the agent
  decides and not something a change can be judged on.

Both are rates, never totals. A session with fewer errors because it was
shorter is not a better setting; dividing by turns is what stops the loop from
learning to end conversations early.

There is no "intelligence" counter here and there should not be. A system that
reported one would be reporting a number it cannot defend, and the first time
it went up nobody would know whether the system got worse or the number got
sloppy.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .evidence import (
    Arm,
    EvidenceLedger,
    Holdout,
    compare_arms,
    read_holdout,
    split_holdout,
)

# A window shorter than this cannot tell a real change from a quiet afternoon.
# Below it the trial is left open rather than judged: reverting on thin evidence
# is worse than leaving a change in place for a while.
MIN_TURNS_PER_TRIAL = 8
# A change has to be worth something to stay. Five percent is small enough to
# catch a real effect and large enough that noise does not pass as one.
MIN_GAIN = 0.05
# A change may cost a little of the other metric, but not this much.
MAX_TOLERATED_REGRESSION = 0.05
TRIAL_NAME = os.path.join(".minagent", "prueba.json")


@dataclass
class Scorecard:
    """What a stretch of turns cost and how often they went wrong."""

    turns: int = 0
    tool_errors: int = 0
    job_refusals: int = 0
    job_failures: int = 0
    tool_tokens: int = 0

    def failures(self) -> int:
        return self.tool_errors + self.job_refusals + self.job_failures

    def failure_rate(self) -> float:
        return self.failures() / self.turns if self.turns else 0.0

    def cost_rate(self) -> float:
        return self.tool_tokens / self.turns if self.turns else 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(text: str) -> Scorecard:
        try:
            payload = json.loads(text or "{}")
        except ValueError:
            return Scorecard()
        if not isinstance(payload, dict):
            return Scorecard()
        card = Scorecard()
        for name in asdict(card):
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                setattr(card, name, value)
        return card


@dataclass
class Trial:
    """One automatic change, and the measurement that will judge it."""

    setting: str
    previous: str
    proposed: str
    reason: str
    started_at: str = ""
    before: Scorecard = field(default_factory=Scorecard)
    after: Scorecard = field(default_factory=Scorecard)
    # The orchestrator counts its whole life, so the window is a difference
    # against where those counters stood when the change was made.
    judged: bool = False
    kept: bool = False
    verdict: str = ""
    # --- evidence state ---
    # Set once the arms have enough runs to be compared. Until then the trial
    # is collecting, and ``verdict`` stays empty on purpose.
    decided: bool = False
    # Which value is live right now. Alternates window by window.
    live: str = "baseline"
    # Turns accumulated into the window currently being measured.
    window_turns: int = 0
    # Index of the current run. Shared by both arms, so the run numbers interleave
    # and each arm still counts its own distinct runs.
    run: int = 0
    # The frozen case split, kept as text so the trial file stays JSON.
    holdout_json: str = ""
    # What the reserved cases said, reported and never gated on.
    surprise: str = ""

    def holdout(self) -> Holdout:
        """The reserved cases, rebuilt. Empty before the trial set one."""
        return Holdout.from_json(self.holdout_json) if self.holdout_json else Holdout()

    def to_json(self) -> str:
        return json.dumps(
            {
                "setting": self.setting,
                "previous": self.previous,
                "proposed": self.proposed,
                "reason": self.reason,
                "started_at": self.started_at,
                "before": json.loads(self.before.to_json()),
                "after": json.loads(self.after.to_json()),
                "judged": self.judged,
                "kept": self.kept,
                "verdict": self.verdict,
                "decided": self.decided,
                "live": self.live,
                "window_turns": self.window_turns,
                "run": self.run,
                "holdout": self.holdout_json,
                "surprise": self.surprise,
            },
            indent=2,
        )

    @staticmethod
    def from_json(text: str) -> Trial | None:
        try:
            payload = json.loads(text or "")
        except ValueError:
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("setting"), str):
            return None
        before = payload.get("before")
        after = payload.get("after")
        return Trial(
            setting=payload["setting"],
            previous=str(payload.get("previous", "")),
            proposed=str(payload.get("proposed", "")),
            reason=str(payload.get("reason", "")),
            started_at=str(payload.get("started_at", "")),
            before=Scorecard.from_json(json.dumps(before) if isinstance(before, dict) else ""),
            after=Scorecard.from_json(json.dumps(after) if isinstance(after, dict) else ""),
            judged=bool(payload.get("judged")),
            kept=bool(payload.get("kept")),
            verdict=str(payload.get("verdict", "")),
            decided=bool(payload.get("decided")),
            live=str(payload.get("live") or "baseline"),
            window_turns=_as_int(payload.get("window_turns")),
            run=_as_int(payload.get("run")),
            holdout_json=str(payload.get("holdout", "")),
            surprise=str(payload.get("surprise", "")),
        )


def load_trial(application_root: str) -> Trial | None:
    """The change currently being measured, if there is one."""
    if not application_root:
        return None
    try:
        with open(os.path.join(application_root, TRIAL_NAME), encoding="utf-8") as handle:
            return Trial.from_json(handle.read())
    except OSError:
        return None


def save_trial(application_root: str, trial: Trial | None) -> str:
    """Persist the trial, or remove the file when there is none.

    Persisted on every turn so that a restart mid-window does not silently
    restart the measurement, which would let a change that was never really
    judged sit there forever.
    """
    if not application_root:
        return ""
    path = os.path.join(application_root, TRIAL_NAME)
    if trial is None:
        try:
            os.remove(path)
        except OSError:
            pass
        return ""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(trial.to_json())
    except OSError:
        return ""
    return path


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def judge(trial: Trial) -> str:
    """Decide whether the change earned its place, and say why.

    ``keep``, ``revert``, or ``undecided`` while the window is still too short
    to tell. The rule is "no harm, and some gain": the failure rate is the
    primary number and may not get meaningfully worse, and at least one of the
    two has to be clearly better. A change that neither helps nor hurts is
    reverted, because leaving it in spends a setting's cooldown on nothing.
    """
    if trial.after.turns < MIN_TURNS_PER_TRIAL:
        return (
            f"undecided: {trial.after.turns} of {MIN_TURNS_PER_TRIAL} turns measured, "
            "not enough to tell a change from a quiet afternoon"
        )
    before_failures, after_failures = trial.before.failure_rate(), trial.after.failure_rate()
    before_cost, after_cost = trial.before.cost_rate(), trial.after.cost_rate()

    if before_failures > 0 and after_failures > before_failures * (1 + MAX_TOLERATED_REGRESSION):
        return (
            f"revert: failures per turn went from {before_failures:.2f} to {after_failures:.2f}. "
            "The change made things worse, not better."
        )
    if before_failures == 0 and after_failures > 0:
        return f"revert: {after_failures:.2f} failures per turn appeared where there were none."

    failure_gain = (before_failures - after_failures) / before_failures if before_failures else 0.0
    cost_gain = (before_cost - after_cost) / before_cost if before_cost else 0.0
    if before_cost > 0 and after_cost > before_cost * (1 + MAX_TOLERATED_REGRESSION):
        # The agent got smoother but more expensive, which is a trade and not
        # an improvement, and the one nobody asked for.
        return (
            f"revert: tool tokens per turn went from {before_cost:.0f} to {after_cost:.0f}. "
            "Fewer failures bought with more context is a trade, not an improvement."
        )

    if failure_gain >= MIN_GAIN or cost_gain >= MIN_GAIN:
        gained = "failures" if failure_gain >= MIN_GAIN else "context cost"
        return (
            f"keep: {gained} per turn improved by {max(failure_gain, cost_gain) * 100:.0f}% "
            f"({trial.after.turns} turns measured)."
        )
    return (
        "revert: nothing measurable improved. Failures "
        f"{before_failures:.2f} -> {after_failures:.2f}, tokens {before_cost:.0f} -> {after_cost:.0f}."
    )


def describe_trial(trial: Trial | None) -> str:
    """A line for ``/mejoras`` saying what is being measured right now."""
    if trial is None:
        return "No hay ningún cambio en medición."
    if trial.judged:
        state = "se queda" if trial.kept else "se ha revertido"
        return f"{trial.setting} {state}: {trial.verdict}"
    return (
        f"{trial.setting} {trial.previous} -> {trial.proposed} en medición, "
        f"{trial.after.turns} de {MIN_TURNS_PER_TRIAL} turnos."
    )


# --- Evidence-backed trials -------------------------------------------------
# What a single window is judged on. Each case can independently be fine or
# broken, so a change that fixes one and breaks another shows up as two
# different facts instead of averaging into a wash.
TRIAL_CASES = ("tool_errors", "job_refusals", "job_failures", "tool_tokens")
EVIDENCE_NAME = os.path.join(".minagent", "evidencia.json")


def load_ledger(application_root: str) -> EvidenceLedger:
    """The outcome history every trial is compared against.

    Read as empty rather than raised when the file is missing or damaged: a
    corrupt ledger should cost the loop its memory, not its ability to start.
    """
    if not application_root:
        return EvidenceLedger()
    try:
        with open(os.path.join(application_root, EVIDENCE_NAME), encoding="utf-8") as handle:
            return EvidenceLedger.from_json(handle.read())
    except (OSError, ValueError):
        return EvidenceLedger()


def save_ledger(application_root: str, ledger: EvidenceLedger) -> None:
    if not application_root:
        return
    target = os.path.join(application_root, EVIDENCE_NAME)
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(ledger.to_json())
    except OSError:
        # An unwritable ledger is a degraded loop, not a failed turn. The next
        # turn will try again, and the cost is that a decision restarts its runs.
        pass


def cases_from_card(card: Scorecard, cases: Sequence[str], *, cost_bar: float) -> dict[str, bool]:
    """How each case fared in one window, from what that window observed.

    The error cases pass only on zero, because a window with a single tool
    error is a window where a tool was misused, and a rate would let one
    mistake hide inside an average of seven good turns. The cost case passes
    against a bar the caller supplies, because an absolute token limit would
    be either trivially satisfied by a quiet session or impossible for a busy
    one.
    """
    results: dict[str, bool] = {}
    for case in cases:
        if case == "tool_errors":
            results[case] = card.tool_errors == 0
        elif case == "job_refusals":
            results[case] = card.job_refusals == 0
        elif case == "job_failures":
            results[case] = card.job_failures == 0
        elif case == "tool_tokens":
            results[case] = card.cost_rate() <= cost_bar
    return results


def _arm_over(ledger: EvidenceLedger, setting: str, value: str, cases: Sequence[str]) -> Arm:
    """One arm restricted to exactly these cases.

    Needed because ``compare_arms`` takes its mean from the whole arm but its
    flips from the case list. Handing it an arm that also contains the reserved
    cases would quietly pull the holdout into the mean that decides the change,
    which is the leak the holdout exists to prevent - and invisible, because
    every value it reads is real.
    """
    wanted = set(cases)
    full = ledger.arm(setting, value=value)
    return Arm(
        name=full.name,
        outcomes=tuple(outcome for outcome in full.outcomes if outcome.case in wanted),
    )


@dataclass
class TrialStep:
    """What one turn of a trial did, said in the caller's language."""

    message: str = ""
    # The value the setting should be moved to now, or "" to leave it alone.
    apply: str = ""
    decided: bool = False
    kept: bool = False
    surprise: str = ""


def start_trial(
    *,
    setting: str,
    previous: str,
    proposed: str,
    reason: str,
    started_at: str,
    holdout_fraction: float,
) -> Trial:
    """A trial with its reserved cases frozen before the first window.

    The split is made here, once, and never recomputed. A holdout chosen again
    per run is a different holdout every run, and the reservation protects
    nothing.
    """
    holdout = split_holdout(TRIAL_CASES, holdout_fraction)
    return Trial(
        setting=setting,
        previous=previous,
        proposed=proposed,
        reason=reason,
        started_at=started_at,
        holdout_json=holdout.to_json(),
        live="baseline",
    )


def advance_trial(
    trial: Trial,
    ledger: EvidenceLedger,
    card: Scorecard,
    *,
    min_runs: int,
    max_regressions: int,
    cost_bar: float,
) -> TrialStep:
    """Fold one turn into the trial, and decide when the evidence is in.

    The arms alternate window by window rather than running in blocks. Measuring
    the baseline for fifty turns and the candidate for the next fifty confounds
    the change with everything that happened in between - the hour of day, the
    kind of work, whether the network was up. Alternating keeps both arms
    exposed to the same conditions, which is the only reason their numbers can
    be subtracted at all.

    The reserved cases are recorded in the same windows but never passed to the
    gate, and are read exactly once at the end. That is sound here and would not
    be in general: it holds because the candidate value is written into the trial
    before the first window and cannot be retuned while the numbers are visible,
    so there is no fitting to the visible cases to catch. A system that changed
    its proposal mid-trial would need a separate collection phase.
    """
    if trial.decided:
        return TrialStep()

    trial.window_turns += 1
    trial.after = card
    if trial.window_turns < MIN_TURNS_PER_TRIAL:
        return TrialStep()

    holdout = trial.holdout()
    gating = holdout.gating_cases(TRIAL_CASES)
    # Every case is recorded, reserved ones included. The gate below only ever
    # sees the gating cases; the reserved ones are collected in the same windows
    # so that a surprise stays a surprise instead of becoming extra data that
    # happened to arrive after the decision.
    outcomes = cases_from_card(card, TRIAL_CASES, cost_bar=cost_bar)
    arm_value = trial.previous if trial.live == "baseline" else trial.proposed
    for case, passed in outcomes.items():
        ledger.record(trial.setting, case=case, passed=passed, run=trial.run, value=arm_value)

    trial.run += 1
    trial.window_turns = 0
    next_live = "candidate" if trial.live == "baseline" else "baseline"
    next_value = trial.proposed if next_live == "candidate" else trial.previous
    # Ask the caller to move the setting only when the next arm is a different
    # value. Compared against the value that just ran, not against the
    # environment: a caller can hold a setting in a file, a session attribute and
    # an environment variable, and only the trial knows which one it last changed.
    switch = next_value if next_value != arm_value else ""

    baseline = _arm_over(ledger, trial.setting, trial.previous, gating)
    candidate = _arm_over(ledger, trial.setting, trial.proposed, gating)
    if baseline.runs() < min_runs or candidate.runs() < min_runs:
        trial.live = next_live
        return TrialStep(apply=switch)

    verdict = compare_arms(
        baseline,
        candidate,
        cases=gating,
        min_runs=min_runs,
        max_regressions=max_regressions,
    )
    trial.decided = True
    trial.judged = True
    trial.kept = verdict.promote
    trial.verdict = verdict.reason
    trial.live = next_live

    decided_value = trial.proposed if verdict.promote else trial.previous
    surprise = ""
    if verdict.promote:
        # Only worth reading when the change is about to be kept. A reverted
        # change has nothing left to validate, and spending the reserve on it
        # would burn the only unbiased sample the next attempt could have used.
        reading = read_holdout(
            holdout,
            _arm_over(ledger, trial.setting, trial.previous, holdout.cases),
            _arm_over(ledger, trial.setting, trial.proposed, holdout.cases),
        )
        surprise = reading.surprise
        trial.surprise = surprise
        # read_holdout spent the reserve in memory. Persist that, or a restart
        # before the next trial reads the same cases a second time.
        trial.holdout_json = holdout.to_json()
    return TrialStep(
        message=f"Medición: {verdict.reason}",
        apply=decided_value if decided_value != arm_value else "",
        decided=True,
        kept=verdict.promote,
        surprise=surprise,
    )


def describe_progress(trial: Trial, ledger: EvidenceLedger, min_runs: int = 9) -> str:
    """Where the trial is, in a form a person can decide to keep waiting for.

    Reports both arms' measured rate, not just which one is live. A progress line
    that only says "measuring, 3/8 turns" tells a person nothing about whether
    the change is heading anywhere, which is the only reason they would be
    reading it. When an arm has nothing observed yet its rate is left out rather
    than shown as 0%, because a mean over nothing is not a result.

    ``min_runs`` is the target the caller is holding this trial to. It is a
    parameter and not a constant so the progress line cannot claim a target the
    run will not actually enforce.
    """
    if trial is None:
        return ""
    if trial.decided:
        return f"{trial.setting} {trial.previous} -> {trial.proposed}: {trial.verdict}"

    holdout = trial.holdout()
    gating = holdout.gating_cases(TRIAL_CASES)
    baseline = _arm_over(ledger, trial.setting, trial.previous, gating)
    candidate = _arm_over(ledger, trial.setting, trial.proposed, gating)

    def side(arm: Arm) -> str:
        rate = arm.rate()
        measured = "sin datos aún" if rate is None else f"{rate:.0f}% de acierto"
        return f"{arm.name}: {arm.runs()}/{min_runs} corridas, {measured}"

    live = "midiendo la anterior" if trial.live == "baseline" else "midiendo la propuesta"
    # Named by value, not by arm. The arm names are internal ("baseline",
    # "candidate") and would reach the reader verbatim, and a line saying
    # "changes applied to candidate" is both untranslated and less useful than the
    # value the reader actually recognises.
    live_value = trial.previous if trial.live == "baseline" else trial.proposed
    reserve = "reservada" if not holdout.spent else "ya leída"
    return (
        f"{trial.setting} {trial.previous} -> {trial.proposed}\n"
        f"  {live} · turnos {trial.window_turns}/{MIN_TURNS_PER_TRIAL} de esta ventana, "
        f"cambios aplicados a {live_value}\n"
        f"  {side(baseline)} | {side(candidate)}\n"
        f"  holdout: {len(holdout.cases)} casos, {reserve}"
    )
