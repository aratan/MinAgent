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
from dataclasses import asdict, dataclass, field
from typing import Any

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
    baseline_refusals: int = 0
    baseline_failures: int = 0
    judged: bool = False
    kept: bool = False
    verdict: str = ""

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
                "baseline_refusals": self.baseline_refusals,
                "baseline_failures": self.baseline_failures,
                "judged": self.judged,
                "kept": self.kept,
                "verdict": self.verdict,
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
            baseline_refusals=_as_int(payload.get("baseline_refusals")),
            baseline_failures=_as_int(payload.get("baseline_failures")),
            judged=bool(payload.get("judged")),
            kept=bool(payload.get("kept")),
            verdict=str(payload.get("verdict", "")),
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


def trial_to_memory(trial: Trial) -> dict[str, Any]:
    """The trial as a memory-shaped record, for the log and for tests."""
    return {"setting": trial.setting, "previous": trial.previous, "proposed": trial.proposed,
            "verdict": trial.verdict, "kept": trial.kept, "turns": trial.after.turns}
