"""Decide what a session is worth remembering, instead of storing every turn.

The store already captures every turn that used tools and finished without an
error. That is a faithful log and a poor memory: most turns are somebody
asking a question that will not come up again, and a hint block built from
that log is mostly noise. Deciding what to keep happens in three places, in
the order they cost something:

1. **In the turn.** The model calls ``remember`` itself at the moment it
   notices something durable, while it still has the context. This is free and
   it is the layer that gets the judgement right, so the other two only have
   to catch what it missed.
2. **Eureka.** Cheap signals mark a turn as a candidate and only then is the
   model asked whether it is worth keeping. A signal that fires on most turns
   costs a model call on most turns, which is the same as having no signal at
   all, so the thresholds here are deliberately conservative.
3. **A review.** Every ``MEMORY_REFLECTION_INTERVAL`` turns the auto-captured
   log is put in front of the model, which says which entries deserve to stay
   reinforced and which were noise worth forgetting.

Every parser in this module is total: a model that answers with prose, a
fenced code block, or nothing at all must not raise, because a failed
reflection is a missed memory and never a broken turn.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# A turn that failed twice and then worked has the shape of a real discovery:
# something was tried, it did not work, and the reason why is the part worth
# keeping. One failure is an ordinary typo and costs nothing to re-derive.
EUREKA_ERROR_THRESHOLD = 2
# Six distinct steps is where a turn stops being one lookup and starts being
# work whose method is worth repeating.
EUREKA_STEP_THRESHOLD = 6
# The judgement only has to be right about one turn, so the prompt stays small:
# the steps, the outcome, and what is being asked of it.
MAX_TURN_PROMPT_CHARS = 3000
MAX_ENTRIES_PER_REVIEW = 40
MAX_ENTRY_EXCERPT_CHARS = 600
# Generous on purpose. A reasoning model spends this before it writes the
# object: measured against the 9B model this project runs, a cap of 600 was
# spent entirely on thinking and returned an empty answer, and the same
# request answered in 606 tokens. The cap bounds a runaway, it does not
# negotiate the answer.
REFLECTION_MAX_TOKENS = 3000

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_KINDS = ("procedure", "solution", "fact", "preference", "experience", "hypothesis")


@dataclass(frozen=True)
class EurekaSignal:
    """Why a turn is worth the model call that decides whether to keep it."""

    reason: str
    detail: str


@dataclass(frozen=True)
class Verdict:
    """The model's answer to "is there something durable here?"."""

    keep: bool
    kind: str = "solution"
    title: str = ""
    content: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ReviewAction:
    """One entry's fate in a review: reinforced, or forgotten."""

    entry_id: int
    keep: bool
    note: str = ""


def detect_eureka(
    *,
    tool_errors: int,
    steps: Sequence[str],
    turn_succeeded: bool,
) -> EurekaSignal | None:
    """Return why this turn deserves a judgement call, or ``None`` if it does not.

    Only successful turns are considered. A turn that ended in an error has
    nothing settled to remember, and the ordinary auto-capture already skips
    it, so judging it would be a call spent on a turn that is not over yet.
    """
    if not turn_succeeded:
        return None
    if tool_errors >= EUREKA_ERROR_THRESHOLD:
        return EurekaSignal(
            "perseverance",
            f"{tool_errors} tool errors before this turn worked, so the way through is the useful part",
        )
    if len(steps) >= EUREKA_STEP_THRESHOLD:
        return EurekaSignal(
            "long_work",
            f"{len(steps)} steps in one turn, which is more method than a single tool call",
        )
    return None


def build_verdict_prompt(
    *,
    request: str,
    steps: Sequence[str],
    outcome: str,
    signal: EurekaSignal,
) -> list[dict[str, str]]:
    """Ask for a keep-or-drop verdict on one turn.

    The instruction is to answer with nothing but the object, because the
    parser looks for the first balanced ``{...}`` and prose around it is
    discarded rather than fatal.
    """
    steps_text = "\n".join(f"- {step}" for step in steps)
    return [
        {
            "role": "system",
            "content": (
                "You decide what is worth remembering from a finished piece of work. Answer with one JSON "
                "object and nothing else.\n"
                'Keep it: {"keep": true, "kind": "procedure|solution|fact|preference", "title": "short '
                'title", "content": "what to reuse later, concrete and self-contained", "reason": "why"}\n'
                'Drop it: {"keep": false, "reason": "why it is not worth keeping"}\n'
                "Keep only what would still be useful in a later session on its own: a method that worked, "
                "a constraint that was discovered, a preference stated by the user. Drop what is specific to "
                "this request, what the tools could be re-derived from, and anything the outcome does not "
                "actually support."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Candidate turn: {signal.reason}. {signal.detail}\n\n"
                f"The request:\n{request[:1200]}\n\n"
                f"What was done:\n{steps_text[:MAX_TURN_PROMPT_CHARS]}\n\n"
                f"How it ended:\n{outcome[:1200]}"
            ),
        },
    ]


def build_review_prompt(entries: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Put the auto-captured log in front of the model to be culled.

    The entries are the raw log and nothing else, so the decision is about
    their content rather than about how long the conversation was.
    """
    listed = "\n\n".join(
        f"[{entry['id']}] {entry.get('title', '')}\n{str(entry.get('content', ''))[:MAX_ENTRY_EXCERPT_CHARS]}"
        for entry in entries[:MAX_ENTRIES_PER_REVIEW]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are curating a memory store that recorded every task mechanically. Answer with one JSON "
                "object and nothing else.\n"
                "Rule: forget an entry that only records that something happened. Keep an entry that states "
                "something a later session could act on - a method, a constraint, a version, a path, a "
                "preference. The two lists must not overlap and must only use ids you were given.\n"
                '{"keep": [2], "forget": [1, 3], "notes": {"2": "the method, not the task"}}'
            ),
        },
        {"role": "user", "content": f"Entries recorded automatically:\n\n{listed}"},
    ]


def first_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a model answer.

    Shared with the improvement pass, which asks the model the same way and has
    to be just as forgiving: prose around the object is discarded, and anything
    unparseable comes back as ``None`` for the caller to treat as "no answer".
    """
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(candidate)):
            character = candidate[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(candidate[start : index + 1])
                    except ValueError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = candidate.find("{", start + 1)
    return None


def _clean_kind(value: Any) -> str:
    """Accept only a kind the store knows, so a hallucination cannot create one."""
    return str(value).strip().lower() if str(value).strip().lower() in _KINDS else "solution"


def parse_verdict(text: str) -> Verdict:
    """Read a keep-or-drop answer, defaulting to drop.

    The default matters more than the parse: a garbled answer means the model
    is not confident, and storing on a guess is how a memory store fills up
    with things that were never verified.
    """
    payload = first_json_object(text)
    if payload is None:
        return Verdict(keep=False, reason="the answer could not be read as JSON, so nothing was kept")
    if payload.get("keep") is not True:
        return Verdict(keep=False, reason=str(payload.get("reason") or "the model did not keep it"))
    title = str(payload.get("title") or "").strip()
    content = str(payload.get("content") or "").strip()
    if not title or not content:
        return Verdict(keep=False, reason="kept without a title or content, which cannot be recalled by")
    return Verdict(
        keep=True,
        kind=_clean_kind(payload.get("kind")),
        title=title,
        content=content,
        reason=str(payload.get("reason") or ""),
    )


def parse_review(text: str, entries: Sequence[dict[str, Any]]) -> list[ReviewAction]:
    """Read the cull lists and drop anything that names an unknown entry.

    A hallucinated id is removed rather than honoured: forgetting a memory the
    model invented the number for is impossible, and the one it got wrong by
    mistake should survive.
    """
    payload = first_json_object(text)
    if payload is None:
        return []
    known = {int(entry["id"]) for entry in entries}
    actions: list[ReviewAction] = []
    seen: set[int] = set()
    for key, keep in (("keep", True), ("forget", False)):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        raw_notes = payload.get("notes")
        notes: dict[Any, Any] = raw_notes if isinstance(raw_notes, dict) else {}
        for value in values:
            try:
                entry_id = int(value)
            except (TypeError, ValueError):
                continue
            if entry_id not in known or entry_id in seen:
                continue
            seen.add(entry_id)
            actions.append(ReviewAction(entry_id=entry_id, keep=keep, note=str(notes.get(str(entry_id), ""))))
    return actions


def format_review_result(actions: Sequence[ReviewAction]) -> str:
    """One line saying what the review did, for the transcript and for tests."""
    kept = sum(1 for action in actions if action.keep)
    forgotten = len(actions) - kept
    if not actions:
        return "Memory review: nothing to change."
    return f"Memory review: {kept} kept, {forgotten} forgotten."
