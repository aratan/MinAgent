"""Investigate a question against the web and admit what survives as a lesson.

The resident could already reflect on memory, and the web tools already existed
to answer a question the user asked. What was missing was the seam between
them: nothing took a gap in what Ara knows, went and looked for it, and came
back with something she is allowed to believe.

That seam is what this module is, and the shape of it is dictated by a failure
observed elsewhere. A dream-capture system there gathered findings with
confidence and citations, staged them, and left them staged forever:
``recalls: 0``, no promotion, no gate. It looked exactly like learning and
learned nothing. So the last step here is not optional and not advisory. A
finding becomes a lesson only by passing ``compose_admission``, which requires
three critics to agree and treats an unreachable reviewer as a refusal. There
is deliberately no path through this module that writes durable knowledge
without that call.

The second thing borrowed from that failure is shape: findings carry pointers,
not pages. What Ara learned is remembered as where she looked and what she
read there, because a stored paragraph is indistinguishable from a conclusion
she reached herself, and a stored pointer can be re-checked and can expire.

Quarantining instruction-shaped page text happens in :mod:`minagent.provenance`
before any of it reaches the extractor, so no prompt-shaped stranger text ever
becomes a claim to be admitted.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .admission import Admission, Criticism, Lesson, compose_admission
from .provenance import (
    DEFAULT_MIN_TIER,
    TIER_FETCHED,
    TIER_SNIPPET,
    Citation,
    ProvenanceLedger,
    Source,
    quarantine_reason,
    render_pointer,
    supports,
    weakest,
)

#: What the extractor is handed, and what it must answer with.
#:
#: It receives the question and the page spans that survived quarantine, and
#: returns one sentence a person could act on, or an empty string for "these
#: pages did not answer it". Returning nothing is a normal outcome and not a
#: failure: a search that finds nothing is a fact worth recording, and a worker
#: that had to invent a claim to have something to promote would be worse than
#: one that admits it learned nothing.
Extractor = Callable[[str, Sequence[tuple[str, str]]], Awaitable[str]]

#: Asked whether the lesson contradicts what is already known. Returns ``None``
#: when the reviewer cannot be reached, which the gate treats as a refusal.
Consistency = Callable[[Lesson, Sequence[Lesson]], Awaitable[Criticism | None]]


@dataclass(frozen=True)
class ResearchQuestion:
    """One gap worth closing, and what it is allowed to cost."""

    question: str
    why: str = ""
    max_pages: int = 4


@dataclass(frozen=True)
class Finding:
    """A claim, the spans that support it, and how firmly it is held."""

    claim: str
    citations: tuple[Citation, ...]
    confidence: float = 0.0

    def tier(self) -> int:
        return weakest(self.citations)

    def is_promotable(self, *, min_tier: int = DEFAULT_MIN_TIER) -> bool:
        return bool(self.claim) and supports(self.citations, min_tier=min_tier)


@dataclass
class ResearchOutcome:
    """Everything one investigation produced, including what it gave up on."""

    question: ResearchQuestion
    findings: list[Finding] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)
    admission: Admission | None = None
    promoted: bool = False
    reason: str = ""

    def describe(self) -> str:
        head = f"{self.question.question}: {len(self.findings)} finding(s)"
        if self.refused:
            head += f", {len(self.refused)} page(s) refused"
        return f"{head} -> {'promoted' if self.promoted else 'not promoted'} ({self.reason})"


class ResearchWorker:
    """Look something up, screen it, and admit it only if three critics agree.

    The client and the two model-backed callables are injected, so the logic
    that matters -- what gets quarantined, what tier a claim rests on, and
    whether the gate is ever bypassed -- is testable without a network or a
    model, which is the only way it can be tested honestly.
    """

    def __init__(
        self,
        *,
        client: Any,
        extract: Extractor,
        consistency: Consistency,
        known: Sequence[Lesson] = (),
        min_tier: int = DEFAULT_MIN_TIER,
        ledger: ProvenanceLedger | None = None,
    ) -> None:
        self._client = client
        self._extract = extract
        self._consistency = consistency
        self._known = tuple(known)
        self._min_tier = min_tier
        self.ledger = ledger or ProvenanceLedger()

    async def investigate(self, question: ResearchQuestion) -> ResearchOutcome:
        outcome = ResearchOutcome(question=question)
        pages = await self._gather(question, outcome)

        if not pages:
            outcome.reason = "no page survived quarantine"
            return outcome

        spans = [(url, text) for url, text in pages]
        claim = (await self._extract(question.question, spans)).strip()
        if not claim:
            outcome.reason = "the extractor found nothing that answers the question"
            return outcome

        citations = tuple(
            Citation(source=Source(url=url, tier=TIER_FETCHED, retrieved_at=time.time()), span=excerpt)
            for url, text in pages
            for excerpt in _excerpts(text)
        )
        finding = Finding(claim=claim, citations=citations)
        outcome.findings.append(finding)

        if not finding.is_promotable(min_tier=self._min_tier):
            outcome.reason = (
                f"claim rests on tier {finding.tier()}, below the floor {self._min_tier}"
            )
            return outcome

        lesson = self._to_lesson(question, finding)
        verdict = await self._consistency(lesson, self._known)
        outcome.admission = compose_admission(lesson, consistency=verdict, known=self._known)
        outcome.promoted = outcome.admission.promote
        outcome.reason = outcome.admission.reason

        if outcome.promoted:
            self.ledger.record_promotion(lesson.title, render_pointer(citations))
        return outcome

    async def _gather(
        self, question: ResearchQuestion, outcome: ResearchOutcome
    ) -> list[tuple[str, str]]:
        """Search, then fetch, keeping only the text that is safe to read.

        Snippets are recorded at their own weaker tier and then discarded: they
        decide which page is worth fetching, and they are never what a claim is
        built on. Only text that survived quarantine reaches the extractor.
        """
        results = await self._client.search(question.question, max_results=question.max_pages)
        safe: list[tuple[str, str]] = []
        for result in results[: question.max_pages]:
            url = result.get("url", "")
            if not url:
                continue
            title = result.get("title", "")
            # Recorded before the fetch, so a refusal still shows the search
            # that led to the page being refused.
            self.ledger.record(Source(url=url, title=title, tier=TIER_SNIPPET))

            try:
                page = await self._client.fetch(url)
            except Exception as exc:  # a dead link is not a failed investigation
                outcome.reason = outcome.reason or f"could not fetch {url}: {exc}"
                continue

            text = _page_text(page)
            if not text:
                continue

            refusal = quarantine_reason(text)
            if refusal is not None:
                self.ledger.record_refusal(url, refusal)
                outcome.refused.append((url, refusal))
                continue

            self.ledger.record(Source(url=url, title=title, tier=TIER_FETCHED))
            safe.append((url, text))
        return safe

    def _to_lesson(self, question: ResearchQuestion, finding: Finding) -> Lesson:
        return Lesson(
            title=question.question[:120],
            guideline=finding.claim[:400],
            trigger=question.question[:200],
            cause=question.why,
            evidence=render_pointer(finding.citations),
            kind="research",
        )


def _page_text(page: Any) -> str:
    if isinstance(page, dict):
        for key in ("content", "text", "body", "markdown"):
            value = page.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    return page if isinstance(page, str) else ""


def _excerpts(text: str, *, limit: int = 240, count: int = 2) -> list[str]:
    """The spans a citation will point at, kept short so they stay checkable."""
    body = " ".join(text.split())
    if not body:
        return []
    return [body[index * limit : index * limit + limit].strip() for index in range(count)][:count]


# --- the question queue --------------------------------------------------------
#
# Where the questions come from is not a planner, because a planner is a
# component nobody can audit from the outside. Ara writes down what she wants
# to know, in a file in the repository, in the same spirit as IDENTITY.md: the
# queue is a thing she can read, edit, and be wrong about. A research pass with
# nothing queued costs nothing and says so.
#
# The format is deliberately dull, because a queue that is pleasant to write is
# a queue that will be malformed:
#
#     - [ ] What is the real default context length of the local model?
#       why: the compaction thresholds were guessed, not measured
#     - [x] Already answered, kept for the record.
#
# One open question is taken per pass, in file order. Order matters more than
# breadth: the point of a queue is that the oldest unanswered thing is the one
# Ara cared about first, and a pass that answered everything at once would be a
# pass nobody could price.

_OPEN_MARKERS = ("- [ ]", "* [ ]", "- [?]", "- [ ]:")
_DONE_MARKERS = ("- [x]", "* [x]", "- [X]")


def parse_open_questions(text: str, *, max_pages: int = 4) -> list[tuple[str, str]]:
    """Return ``(question, why)`` for each unanswered item, in file order.

    An item runs until the next bullet or a blank line, so a ``why:`` on its own
    line belongs to the question above it and does not become a question of its
    own. Anything without text after the marker is skipped rather than guessed
    at: an empty checkbox is a formatting slip, not a research request.
    """
    found: list[tuple[str, str]] = []
    question: str | None = None
    why: list[str] = []

    def flush() -> None:
        nonlocal question, why
        if question:
            found.append((question, " ".join(why).strip()))
        question, why = None, []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith(_DONE_MARKERS):
            flush()
            continue
        if lowered.startswith(_OPEN_MARKERS):
            flush()
            body = line[line.index("]") + 1 :].strip().lstrip("-*?").strip()
            question = body or None
            continue
        if line.startswith(("why:", "porque:")):
            if question is not None:
                why.append(line.split(":", 1)[1].strip())
            continue
        # A bare continuation line extends the current question rather than
        # starting one, so a wrapped question still reads as one question.
        if question is not None and not why:
            question = f"{question} {line}".strip()
    flush()
    return found


