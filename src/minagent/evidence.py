"""Whether a self-change is real, measured before it is allowed to count.

An agent that changes one of its own settings has to answer a question that is
easy to skip and expensive to skip: did that make things better, or did it look
like it did? This module is the answer, and it exists because both halves of the
naive answer are wrong in ways that only show up after you have acted on them.

**One run is not a measurement.** Over 60,000 trajectories on SWE-Bench-Verified
across three models and two scaffolds, single-run pass@1 moved 2.2 to 6.0
percentage points depending on which run happened to be the one you kept, and the
standard deviation exceeded 1.5 points *at temperature 0* - variance sometimes
rose when thinking was disabled. Temperature 0 is not determinism. A power
analysis at 80% needs roughly nine runs per arm to resolve a 2-point change and
roughly thirty-six to resolve a 1-point one, so anything under :data:`NOISE_FLOOR`
points is not a small improvement, it is an unmeasurable one. A loop that
measures once before and once after a change does not learn whether its change
helped; it learns what the dice did.

**The mean hides the damage.** Gating on an aggregate score is what AgentDevel
removed to see what would happen, and the run without the gate produced its
*best* scores together with a 14.8% regression rate and four bad releases. A mean
can improve while specific cases that used to work stop working, and a release
decision made on the mean alone ships exactly that. So this module counts
*flips* - a case that passed and now fails, a case that failed and now passes -
and treats the regressions as the thing that has to be justified, not the mean as
the thing that has to be impressive.

**Some cases must never be counted.** Both the flip gate and any stopping rule
are decisions, and a case used to make a decision is a case that has been spent.
The holdout is the frozen answer to that: a reserved set that is measured, and
whose results are read exactly once, after the decision is already made.

Everything here is a pure function of its arguments. Nothing reads the clock, the
filesystem or the model, which is what lets a test pin the boundary instead of
trusting it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Standard deviation of a single pass@1 measurement, in percentage points,
# measured across models and scaffolds. A difference smaller than this is
# indistinguishable from which run you happened to keep.
NOISE_FLOOR_POINTS = 1.5

# Runs per arm needed to resolve a 2-point difference at 80% power. Below this a
# loop is reading noise, and the honest response is a smaller change rather than
# a lower bar.
MIN_RUNS_PER_ARM = 9

# Regressions tolerated before a candidate is refused regardless of its mean. Not
# zero, because a stochastic agent will flip a case by chance; small, because a
# release is not the place to spend it.
DEFAULT_MAX_REGRESSIONS = 1

# A difference this far below baseline is a refusal, not a promotion. An
# improvement loop that can only add is a ratchet, and a ratchet on a noisy
# signal walks the agent downhill.
DEFAULT_MIN_DELTA_POINTS = 0.0

REGRESSION = "regression"
FIX = "fix"

_HOLDOUT_MARKER = 0x5A


@dataclass(frozen=True)
class CaseOutcome:
    """One case, in one run, of one arm."""

    case: str
    passed: bool
    run: int = 0


@dataclass(frozen=True)
class Flip:
    """One case that changed verdict between the two arms."""

    case: str
    direction: str
    detail: str = ""


@dataclass(frozen=True)
class Arm:
    """Every observation made of one configuration, across however many runs."""

    name: str
    outcomes: tuple[CaseOutcome, ...] = ()
    holdout: bool = False

    def runs(self) -> int:
        """Distinct run indices seen, because ten results in one run is one run."""
        return len({outcome.run for outcome in self.outcomes})

    def passed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.passed)

    def total(self) -> int:
        return len(self.outcomes)

    def rate(self) -> float | None:
        """Pass rate as a percentage, or ``None`` when nothing was observed."""
        if not self.outcomes:
            return None
        return 100.0 * self.passed() / self.total()

    def case_verdicts(self) -> dict[str, bool]:
        """The latest verdict per case, so a case run twice is not counted twice."""
        latest: dict[str, bool] = {}
        for outcome in sorted(self.outcomes, key=lambda item: item.run):
            latest[outcome.case] = outcome.passed
        return latest


@dataclass(frozen=True)
class Verdict:
    """Whether a candidate configuration may replace the one it replaces.

    ``promote`` is the only field a caller should act on. ``reason`` is written
    for a human reading a transcript, because a self-changing agent that
    silently refuses a change is indistinguishable from one that is broken.
    """

    promote: bool
    reason: str
    regressions: tuple[Flip, ...] = ()
    fixes: tuple[Flip, ...] = ()
    delta_points: float | None = None
    measurable: bool = True

    @property
    def regression_count(self) -> int:
        return len(self.regressions)


@dataclass
class Holdout:
    """The reserved cases, and the record of what may never be gated on.

    Persisted rather than recomputed, because a holdout that is re-derived from
    a different set of cases each pass is not a holdout - it is a different
    sample with the same name, and the leakage it was meant to prevent comes
    straight back in through the renaming.
    """

    cases: tuple[str, ...] = ()
    spent: bool = False

    def holds(self, case: str) -> bool:
        return case in self.cases

    def gating_cases(self, cases: Iterable[str]) -> list[str]:
        """Only the cases this holdout does not reserve."""
        return [case for case in cases if not self.holds(case)]

    def spend(self) -> None:
        """Mark the holdout as read. Called once, after the decision is made."""
        self.spent = True

    def to_json(self) -> str:
        import json

        return json.dumps({"cases": list(self.cases), "spent": self.spent}, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> Holdout:
        import json

        try:
            payload = json.loads(text or "{}")
        except ValueError:
            return Holdout()
        if not isinstance(payload, dict):
            return Holdout()
        cases = payload.get("cases")
        reserved = tuple(
            case for case in cases if isinstance(case, str) and case
        ) if isinstance(cases, list) else ()
        return Holdout(cases=reserved, spent=bool(payload.get("spent")))


def split_holdout(cases: Sequence[str], fraction: float = 0.2) -> Holdout:
    """Reserve a deterministic slice of the cases, and never move it again.

    Deterministic on purpose. Reserving by shuffling means the reserved set
    depends on a seed, and a seed that is regenerated per pass produces a
    different holdout every time - at which point cases gated on last month are
    gating decisions again and the reservation protects nothing. Hashing the
    case name gives a split that is stable, order-independent, and identical on
    every machine, with no seed to get wrong.
    """
    if not cases or fraction <= 0:
        return Holdout()
    if fraction >= 1:
        return Holdout(cases=tuple(dict.fromkeys(cases)))
    unique = list(dict.fromkeys(cases))
    # Hash each case to a uniform score, then reserve the top slice by score.
    # Ranking on the hash is what makes the split respect ``fraction`` while
    # staying order-independent and seed-free. Selecting on a single hash bit
    # instead - which is what this did first - is stable but also a fixed 50%:
    # every fraction produced the same 23 of 40 cases, so a configured holdout
    # of 20% silently reserved 58% and nothing downstream could tell.
    scored = sorted(unique, key=lambda case: (hashlib.sha256(case.encode("utf-8")).digest(), case))
    count = max(1, round(fraction * len(scored)))
    count = min(count, len(scored) - 1) if len(scored) > 1 else count
    reserved = tuple(sorted(scored[-count:]))
    return Holdout(cases=reserved)


def find_flips(baseline: Arm, candidate: Arm, *, cases: Sequence[str] | None = None) -> tuple[Flip, ...]:
    """Every case whose verdict moved between the two arms.

    Only cases observed in *both* arms count. A case the candidate never ran is
    missing, not fixed, and scoring it as either would let a candidate improve
    its mean by running fewer things.
    """
    before = baseline.case_verdicts()
    after = candidate.case_verdicts()
    shared = sorted(set(before) & set(after))
    if cases is not None:
        allowed = set(cases)
        shared = [case for case in shared if case in allowed]
    latest_before_run = {outcome.case: outcome.run for outcome in sorted(baseline.outcomes, key=lambda item: item.run)}
    latest_after_run = {outcome.case: outcome.run for outcome in sorted(candidate.outcomes, key=lambda item: item.run)}
    flips: list[Flip] = []
    for case in shared:
        if before[case] and not after[case]:
            flips.append(
                Flip(
                    case=case,
                    direction=REGRESSION,
                    detail=f"passed at run {latest_before_run.get(case, 0)}, failed at run {latest_after_run.get(case, 0)}",
                )
            )
        elif not before[case] and after[case]:
            flips.append(
                Flip(
                    case=case,
                    direction=FIX,
                    detail=f"failed at run {latest_before_run.get(case, 0)}, passed at run {latest_after_run.get(case, 0)}",
                )
            )
    return tuple(flips)


def compare_arms(
    baseline: Arm,
    candidate: Arm,
    *,
    max_regressions: int = DEFAULT_MAX_REGRESSIONS,
    min_delta_points: float = DEFAULT_MIN_DELTA_POINTS,
    min_runs: int = MIN_RUNS_PER_ARM,
    cases: Sequence[str] | None = None,
) -> Verdict:
    """Decide whether a candidate configuration earns the change it asked for.

    The order of the checks is the argument. Unmeasurable first, because a
    refusal for "not enough runs" is a different and more useful thing to report
    than a refusal for "too many regressions", and conflating them teaches a
    caller to add runs when the real problem was the regressions. Regressions
    second, because a change that breaks a case it fixed is not an improvement at
    any mean. Mean last, because it is the weakest signal of the three.
    """
    if baseline.holdout or candidate.holdout:
        return Verdict(
            promote=False,
            reason="holdout cases are read once, after the decision, never gated on",
        )

    before_rate = baseline.rate()
    after_rate = candidate.rate()
    if before_rate is None or after_rate is None:
        return Verdict(
            promote=False,
            reason="nothing was measured in one of the arms",
            measurable=False,
        )

    eligible = cases if cases is not None else sorted(
        set(baseline.case_verdicts()) & set(candidate.case_verdicts())
    )
    flips = find_flips(baseline, candidate, cases=eligible)
    regressions = tuple(flip for flip in flips if flip.direction == REGRESSION)
    fixes = tuple(flip for flip in flips if flip.direction == FIX)

    runs = min(baseline.runs(), candidate.runs())
    delta = after_rate - before_rate
    measurable = (
        runs >= min_runs
        and abs(delta) >= NOISE_FLOOR_POINTS
    )

    if runs < min_runs:
        return Verdict(
            promote=False,
            reason=(
                f"{runs} run(s) per arm, {min_runs} needed to tell a real change from "
                f"which run you happened to keep; make the change smaller rather than "
                f"deciding on noise"
            ),
            regressions=regressions,
            fixes=fixes,
            delta_points=delta,
            measurable=False,
        )

    if len(regressions) > max_regressions:
        broken = ", ".join(flip.case for flip in regressions)
        return Verdict(
            promote=False,
            reason=(
                f"{len(regressions)} case(s) that worked now fail, over the budget of "
                f"{max_regressions}: {broken}"
            ),
            regressions=regressions,
            fixes=fixes,
            delta_points=delta,
        )

    if delta < min_delta_points:
        return Verdict(
            promote=False,
            reason=(
                f"the change is {delta:+.1f} points, below the {min_delta_points:+.1f} "
                f"floor; an improvement loop that can only add walks the agent downhill"
            ),
            regressions=regressions,
            fixes=fixes,
            delta_points=delta,
        )

    if not measurable:
        return Verdict(
            promote=False,
            reason=(
                f"{runs} runs per arm and {delta:+.1f} points: measured, but the difference "
                f"is inside the {NOISE_FLOOR_POINTS} point noise floor"
            ),
            regressions=regressions,
            fixes=fixes,
            delta_points=delta,
            measurable=False,
        )

    return Verdict(
        promote=True,
        reason=(
            f"{delta:+.1f} points over {runs} runs per arm, {len(fixes)} fixed, "
            f"{len(regressions)} broken"
        ),
        regressions=regressions,
        fixes=fixes,
        delta_points=delta,
    )


@dataclass(frozen=True)
class HoldoutReading:
    """What the reserved cases said, read once, after the decision was made.

    Reported and never acted on. A holdout whose result feeds back into the
    decision that reserved it is no longer a holdout, it is a second opinion
    obtained with the answer already written down - which is worth having as a
    number and worthless as a gate. So this type has no ``promote`` field at
    all: there is deliberately nothing here for a caller to branch on.
    """

    cases: tuple[str, ...]
    before_rate: float | None
    after_rate: float | None
    surprise: str = ""

    def agrees(self) -> bool:
        """Did the reserved cases move the same way the gated ones did?"""
        if self.before_rate is None or self.after_rate is None:
            return True
        return (self.after_rate - self.before_rate) >= 0


def read_holdout(holdout: Holdout, baseline: Arm, candidate: Arm) -> HoldoutReading:
    """Read the reserved cases once. Does not gate, and marks them spent.

    The interesting field is ``surprise``: when the gated cases improved and the
    reserved ones did not, the change was fitted to the cases that were visible
    while deciding. That is the failure a holdout exists to surface, and it is
    only visible if the reserved cases were never allowed to vote earlier.
    """
    reserved = tuple(case for case in holdout.cases if case in baseline.case_verdicts() and case in candidate.case_verdicts())
    # Scoped to the reserved cases, deliberately. Averaging the whole arm here
    # would report the full suite's improvement as if it were the reserved
    # sample's, which is the leakage the holdout was created to prevent - and
    # worse than no holdout at all, because ``agrees`` would then be answering
    # a question about data that was already visible while deciding.
    before = _rate_of(baseline, reserved)
    after = _rate_of(candidate, reserved)
    surprise = ""
    if before is not None and after is not None and (after - before) < 0:
        surprise = (
            "the gated cases moved up but the reserved cases moved down, so the change "
            "was fitted to the cases that were visible while deciding"
        )
    holdout.spend()
    return HoldoutReading(cases=reserved, before_rate=before, after_rate=after, surprise=surprise)


def _rate_of(arm: Arm, cases: Sequence[str]) -> float | None:
    """Pass rate over exactly these cases, or ``None`` when none were seen."""
    verdicts = arm.case_verdicts()
    seen = [verdicts[case] for case in cases if case in verdicts]
    if not seen:
        return None
    return 100.0 * sum(1 for passed in seen if passed) / len(seen)


@dataclass
class EvidenceLedger:
    """Outcomes per setting, so the next change is compared against a baseline.

    The baseline is not a stored number, it is the outcome history the setting
    had *before* the move. Storing only the post-change number throws away the
    one thing a comparison needs, which is why a change made last week cannot be
    judged this week without re-running it.
    """

    outcomes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record(
        self,
        setting: str,
        *,
        case: str,
        passed: bool,
        run: int,
        value: str,
        setting_before: str = "",
    ) -> None:
        entry = {
            "case": case,
            "passed": bool(passed),
            "run": int(run),
            "value": value,
            "before": setting_before,
        }
        self.outcomes.setdefault(setting, []).append(entry)

    def arm(self, setting: str, *, value: str = "", holdout: bool = False) -> Arm:
        """One arm rebuilt from the ledger for a specific setting value."""
        entries = [entry for entry in self.outcomes.get(setting, []) if entry.get("passed") is not None]
        if value:
            entries = [entry for entry in entries if str(entry.get("value", "")) == str(value)]
        outcomes = tuple(
            CaseOutcome(case=str(entry.get("case", "")), passed=bool(entry.get("passed")), run=int(entry.get("run", 0)))
            for entry in entries
        )
        return Arm(name=f"{setting}={value or 'all'}", outcomes=outcomes, holdout=holdout)

    def baseline_value(self, setting: str, current: str) -> str:
        """The value the setting had before the most recent move, if recorded."""
        entries = self.outcomes.get(setting, [])
        for entry in reversed(entries):
            before = str(entry.get("before", ""))
            if before and before != current:
                return before
        return ""

    def to_json(self) -> str:
        import json

        return json.dumps(self.outcomes, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> EvidenceLedger:
        import json

        try:
            payload = json.loads(text or "{}")
        except ValueError:
            return EvidenceLedger()
        if not isinstance(payload, dict):
            return EvidenceLedger()
        outcomes: dict[str, list[dict[str, Any]]] = {}
        for key, entries in payload.items():
            if not isinstance(key, str) or not isinstance(entries, list):
                continue
            kept = [entry for entry in entries if isinstance(entry, dict)]
            if kept:
                outcomes[key] = kept
        return EvidenceLedger(outcomes=outcomes)
