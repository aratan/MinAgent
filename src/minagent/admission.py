"""Whether a lesson may enter the agent's working context at all.

Once a distilled lesson is in context it is not inert. It shapes the next
decision, that decision produces work, and the work is itself later distilled -
so a single wrong lesson does not stay wrong, it propagates. VaG measured this on
a self-evolving agent and found two things that decide the whole design here.

**Self-evolution is not monotonic.** Past a critical size of the skill pool, each
new skill *degraded* performance rather than adding to it. The pool was not a
library that grows, it was a shared surface every later skill inherits from.

**Contamination is structurally irreversible.** Deleting the skill that caused
the damage recovered only a fraction of the loss, because the descendants had
already inherited the flawed reasoning. This is the sentence the whole module
exists for: *deleting the culprit afterwards is not a repair, it is a partial
refund.* An agent that writes a lesson, watches it misbehave, and then removes
it has already paid most of the cost.

So admission happens before the write, never after. A lesson that fails a critic
is not stored, and nothing downstream ever sees it. There is no queue to review
later and no log to clean up, because a queue of questionable lessons is a queue
of lessons already influencing the next pass.

The three critics are **disjoint on purpose** - structural validity, behavioral
harmlessness, and consistency with what is already known are different questions
with different failure modes, and one critic asked three ways is one critic. The
same work found that selecting for marginal gain, rather than admitting
everything the critics passed, reached higher pass@1 from a pool around five
times smaller: most lessons that survive review contribute nothing, and every
one of them still gets read on every future call.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# A lesson shorter than this is not a rule, and a lesson longer than this is a
# document nobody re-reads. Both are in the prompt on every turn, so both are
# paid for on every turn.
MIN_GUIDELINE_CHARS = 24
MAX_GUIDELINE_CHARS = 400
MAX_TRIGGER_CHARS = 200

# A trigger condition is what makes a lesson retrievable at the right moment.
# Without one, the lesson is either always relevant (and therefore ignored) or
# never (and therefore dead weight in the prompt).
MIN_TRIGGER_CHARS = 12

VALID = "valid"
INVALID = "invalid"
UNSAFE = "unsafe"
REDUNDANT = "redundant"
REJECTED = "rejected"

_ADMISSION_SYSTEM = (
    "You are screening a lesson before it enters an agent's working context, where it "
    "will shape every later decision and be inherited by later lessons derived from "
    "those decisions. Answer with one JSON object and nothing else.\n"
    "Judge only the question you are asked. Do not answer the other two questions: "
    "another reviewer has them and a review that answers three questions is one "
    "review.\n"
)


@dataclass(frozen=True)
class Lesson:
    """One piece of distilled experience, in the shape that makes it usable.

    The three text fields are not interchangeable. ``cause`` is why it happened,
    ``guideline`` is what to do, and ``trigger`` is when the guideline applies.
    Research on experience-derived heuristics found raw trajectories stored as
    examples measurably *worse* than no examples at all, and that the useful form
    is a cause plus a guideline with an explicit condition attached - so a lesson
    missing its trigger is not a smaller lesson, it is the form that does not
    work.
    """

    title: str
    guideline: str
    trigger: str
    cause: str = ""
    evidence: str = ""
    kind: str = "failure"

    def is_wellformed(self) -> bool:
        """Structure only. A lesson that cannot be read is not judged, it is dropped."""
        if not self.guideline or not self.trigger:
            return False
        if not MIN_GUIDELINE_CHARS <= len(self.guideline) <= MAX_GUIDELINE_CHARS:
            return False
        if len(self.trigger) < MIN_TRIGGER_CHARS:
            return False
        return not (len(self.title) > 120 or len(self.trigger) > MAX_TRIGGER_CHARS)

    def describe(self) -> str:
        return (
            f"{self.title}\n"
            f"When: {self.trigger}\n"
            f"Do: {self.guideline}"
            + (f"\nBecause: {self.cause}" if self.cause else "")
        )


@dataclass(frozen=True)
class Criticism:
    """One reviewer's answer to one question."""

    critic: str
    verdict: str
    reason: str = ""


@dataclass(frozen=True)
class Admission:
    """Whether one lesson may be written, and what each reviewer said about it."""

    promote: bool
    reason: str
    criticisms: tuple[Criticism, ...] = ()

    @property
    def rejections(self) -> tuple[Criticism, ...]:
        return tuple(item for item in self.criticisms if item.verdict != VALID)

    def to_json(self) -> str:
        import json

        return json.dumps(
            {
                "promote": self.promote,
                "reason": self.reason,
                "criticisms": [
                    {"critic": item.critic, "verdict": item.verdict, "reason": item.reason}
                    for item in self.criticisms
                ],
            },
            indent=2,
            sort_keys=True,
        )


STRUCTURAL = "structural"
BEHAVIORAL = "behavioral"
CONSISTENCY = "consistency"


def structural_criticism(lesson: Lesson) -> Criticism:
    """Is it well-formed, and does it have the fields retrieval depends on?

    Deterministic on purpose. This is a shape check, and spending a model call
    to discover that a field is empty is a way of learning something the parser
    already knows for certain.
    """
    if not lesson.is_wellformed():
        missing = []
        if not lesson.guideline:
            missing.append("guideline")
        if not lesson.trigger or len(lesson.trigger) < MIN_TRIGGER_CHARS:
            missing.append("trigger")
        if lesson.guideline and not MIN_GUIDELINE_CHARS <= len(lesson.guideline) <= MAX_GUIDELINE_CHARS:
            missing.append("guideline length")
        if not missing:
            missing.append("title or trigger length")
        return Criticism(STRUCTURAL, INVALID, "missing or malformed: " + ", ".join(missing))
    return Criticism(STRUCTURAL, VALID)


def behavioral_criticism(lesson: Lesson) -> Criticism:
    """Would following it on the wrong occasion do damage?

    Kept separate from the other two because a lesson can be perfectly formed,
    perfectly consistent with what is known, and still be a trap: "always retry"
    is true in most sessions and harmful in the one where the failure was a
    cancelled payment. The question is not whether the advice is good, it is
    whether the advice has a failure mode.
    """
    text = f"{lesson.guideline} {lesson.trigger} {lesson.cause}".casefold()
    unbounded = (
        "always", "never", "every time", "no matter what", "must never",
        "whatever happens", "in all cases", "without exception",
    )
    for phrase in unbounded:
        if phrase in text:
            return Criticism(
                BEHAVIORAL,
                UNSAFE,
                f"'{phrase}' has no failure mode: a guideline that cannot be wrong "
                f"cannot be right either",
            )
    if not lesson.cause:
        return Criticism(
            BEHAVIORAL,
            INVALID,
            "no stated cause, so there is no way to tell the case it stops applying to",
        )
    return Criticism(BEHAVIORAL, VALID)


def consistency_criticism(lesson: Lesson, known: Sequence[Lesson] = ()) -> Criticism:
    """Does it contradict, or merely restate, what is already in context?

    Contradiction is a rejection. Restatement is a rejection too, and for a
    different reason: a duplicate lesson is paid for on every future call and
    teaches nothing, which is why the same work reached better accuracy from a
    pool several times smaller by selecting for marginal gain instead of keeping
    everything that survives review.
    """
    guideline = lesson.guideline.casefold().strip()
    for other in known:
        if other.guideline.casefold().strip() == guideline:
            return Criticism(CONSISTENCY, REDUNDANT, f"already known as '{other.title}'")
        overlap = _shared_words(guideline, other.guideline.casefold())
        if overlap and _is_opposed(other.guideline, lesson.guideline):
            return Criticism(
                CONSISTENCY,
                INVALID,
                f"contradicts '{other.title}' on {', '.join(sorted(overlap))}",
            )
    return Criticism(CONSISTENCY, VALID)


def admit(lesson: Lesson, known: Sequence[Lesson] = ()) -> Admission:
    """Screen a lesson with the deterministic critics, before it reaches context.

    The third critic needs a model and so is not run here; ``compose_admission``
    takes its answer and closes the decision. Splitting it this way keeps the
    part that must never be wrong - a malformed or unbounded lesson - out of any
    code path that depends on a request succeeding.

    This returns ``promote=False`` while consistency is still outstanding, and
    that is not a detail. It once returned ``True`` on the grounds that the
    deterministic critics had passed - which meant any caller using the
    convenient function instead of the two-step one got a green light on a
    lesson whose contradiction with what the agent already believed had never
    been checked. The only way to finish is ``compose_admission``, so the
    incomplete result has to read as incomplete.
    """
    criticisms = (structural_criticism(lesson), behavioral_criticism(lesson), consistency_criticism(lesson, known))
    for item in criticisms:
        if item.verdict != VALID:
            return Admission(promote=False, reason=f"{item.critic}: {item.reason}", criticisms=criticisms)
    return Admission(
        promote=False,
        reason="structure and behavior pass, but consistency is still undecided; "
        "pass compose_admission to close this",
        criticisms=criticisms,
    )


def compose_admission(lesson: Lesson, *, consistency: Criticism | None, known: Sequence[Lesson] = ()) -> Admission:
    """Close the decision once the consistency reviewer has answered.

    ``consistency`` of ``None`` means the reviewer could not be reached. That is
    treated as a refusal rather than a pass, because a screen that fails open is
    not a screen - the whole point of a pre-commit gate is that it is the only
    thing standing between a bad lesson and every decision after it.
    """
    criticisms = [structural_criticism(lesson), behavioral_criticism(lesson)]
    if consistency is None:
        criticisms.append(Criticism(CONSISTENCY, REJECTED, "the reviewer could not be reached"))
        return Admission(promote=False, reason="consistency: reviewer unavailable, refusing to admit blind", criticisms=tuple(criticisms))
    criticisms.append(consistency)
    for item in criticisms:
        if item.verdict != VALID:
            return Admission(promote=False, reason=f"{item.critic}: {item.reason}", criticisms=tuple(criticisms))
    return Admission(promote=True, reason="admitted pre-commit by three disjoint critics", criticisms=tuple(criticisms))


def build_consistency_prompt(lesson: Lesson, known: Sequence[Lesson]) -> list[dict[str, str]]:
    """Ask one question: does this lesson contradict or restate what is known?

    Asked alone, with the other two questions deliberately absent. The reviewers
    are disjoint because a single reviewer asked three questions answers the one
    it is most confident about, and a gate made of three confident answers to the
    same question catches one failure mode instead of three.
    """
    if known:  # noqa: SIM108 - the ternary is one unreadable 120-char line
        existing = "\n".join(f"- {item.describe()}" for item in known[:24])
    else:
        existing = "- nothing yet"
    return [
        {"role": "system", "content": _ADMISSION_SYSTEM + (
            '{"verdict": "valid|invalid|redundant", "reason": "why"}\n'
            "Return \"redundant\" if it says something already present, \"invalid\" if it "
            "contradicts a lesson already in context, and \"valid\" only if it adds something "
            "new that does not conflict."
        )},
        {"role": "user", "content": (
            f"Lessons already in context:\n{existing}\n\n"
            f"Candidate lesson:\n{lesson.describe()}"
        )},
    ]


def select_marginal(candidates: Sequence[Lesson], budget: int) -> list[Lesson]:
    """Keep the ones that earn their place, when not all of them do.

    Selection rather than admission alone, because passing review is a low bar.
    Every lesson kept is carried in the prompt on every subsequent call, and the
    same work that reported a five-times-smaller pool at higher accuracy got
    there by dropping lessons the critics had already accepted. ``budget`` of 0
    or less keeps nothing: a caller with no room should say so rather than
    receive the whole set and pay for it quietly.
    """
    if budget <= 0:
        return []
    kept: list[Lesson] = []
    seen: set[str] = set()
    for lesson in candidates:
        fingerprint = " ".join(sorted(set(lesson.guideline.casefold().split())))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append(lesson)
        if len(kept) >= budget:
            break
    return kept


def _shared_words(left: str, right: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "it", "with", "that", "be"}
    return {
        word
        for word in set(left.split()) & set(right.split())
        if len(word) > 3 and word not in stop
    }


_NEGATIONS = ("not ", "no ", "never", "avoid", "don't", "don't", "skip", "stop")


def _is_opposed(left: str, right: str) -> bool:
    """Two guidelines that share a subject but carry opposite polarity."""
    left_negated = any(marker in left for marker in _NEGATIONS)
    right_negated = any(marker in right for marker in _NEGATIONS)
    return left_negated != right_negated


@dataclass
class AdmissionLog:
    """What was screened, and why, so a refusal can be read back later.

    Persisted because a gate nobody can inspect is a gate nobody can trust, and
    because the interesting question about a self-changing agent is not whether
    it refused but whether it refused for the stated reason.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, lesson: Lesson, admission: Admission) -> None:
        self.entries.append({
            "title": lesson.title,
            "promote": admission.promote,
            "reason": admission.reason,
            "criticisms": [
                {"critic": item.critic, "verdict": item.verdict, "reason": item.reason}
                for item in admission.criticisms
            ],
        })

    def refused(self) -> list[dict[str, Any]]:
        return [entry for entry in self.entries if not entry.get("promote")]

    def to_json(self) -> str:
        import json

        return json.dumps(self.entries, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> AdmissionLog:
        import json

        try:
            payload = json.loads(text or "[]")
        except ValueError:
            return AdmissionLog()
        if not isinstance(payload, list):
            return AdmissionLog()
        return AdmissionLog(entries=[entry for entry in payload if isinstance(entry, dict)])


ADMISSION_LOG_NAME = os.path.join(".minagent", "admisiones.json")


def load_admission_log(application_root: str) -> AdmissionLog:
    """The record of past screens, so a refusal can be read back later."""
    if not application_root:
        return AdmissionLog()
    try:
        with open(os.path.join(application_root, ADMISSION_LOG_NAME), encoding="utf-8") as handle:
            return AdmissionLog.from_json(handle.read())
    except OSError:
        return AdmissionLog()


def save_admission_log(application_root: str, log: AdmissionLog) -> str:
    """Persist the record. Refusals are kept, not only the promotions.

    A gate that only remembers what it allowed is indistinguishable from no gate
    at all when something later goes wrong: the question worth answering is which
    lessons were turned away and for which critic's reason.
    """
    if not application_root:
        return ""
    path = os.path.join(application_root, ADMISSION_LOG_NAME)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(log.to_json())
    except OSError:
        return ""
    return path
