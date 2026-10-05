"""Deciding what to ask about when nobody is watching.

The research pass answers the oldest question in ``agente/PREGUNTAS.md``. That
file was a deliberate boundary: the queue is something a person can read, edit
and be wrong about, because an agent that picks its own curiosities spends real
provider credits on questions nobody recorded. This module keeps that property
and adds the autonomy that was asked for, by splitting the two.

**Choosing a question is not the same as answering one.** Choosing writes lines
into the queue, where they can be read, corrected or deleted before any request
is spent on them. Answering is the pass that already exists, and it still only
ever reads the queue. So a night's autonomy produces a shortlist its owner can
read over breakfast, not a batch of facts that arrived already believed.

The shortlist is written through six hats - the fact/feeling/risk/benefit/
alternative/discipline passes Edward de Bono described for forcing a group (or
one person) out of a single habitual frame. Each hat is a different discipline
about the *same* subject, which is what makes the set worth more than one
question asked twice: the white hat asks what is actually established, the black
hat what would break it, the green hat what nobody has tried.

The hats are asked for in **one** request, not six. Six requests per cycle is
six requests nobody is around to audit, and the method is about holding six
disciplines in view, not about spending six times as much to get there. That is
a real deviation from a workshop reading of the method, chosen for a budget,
and it is why a hat that comes back empty is simply not asked about.

Nothing here writes to memory, reaches the network, or calls a model. It builds
a prompt, parses an answer, and edits a text file - so all of it is testable
without either.
"""

from __future__ import annotations

import json
import os
import re

#: How many questions one curation pass may add to the queue.
MAX_CURATED_QUESTIONS = 3

#: Below this many open questions, a cycle is allowed to curate. Refusing to
#: top the queue up past this is what stops a nightly loop from growing a backlog
#: nobody will ever read: the queue is a working note, not an archive.
MIN_OPEN_QUESTIONS = 2

#: Beyond this, a curated question is not worth a queue line.
MAX_QUESTION_CHARS = 160

#: The six disciplines, each with the question it is allowed to ask. Order is the
#: order De Bono sets them in, and it matters: the hats that widen come after the
#: ones that ground, so a cycle spends its certainty before it spends its hope.
HATS: tuple[tuple[str, str, str], ...] = (
    (
        "white",
        "Only what is already established. Cite what is known, what the evidence says, and what is unknown.",
        "What is actually known about {subject}, and what is still unverified?",
    ),
    (
        "red",
        "Feelings and instinct, without defending them. Name the hunch and the worry as feelings.",
        "What does the gut say could go wrong with {subject}, and why does it feel urgent?",
    ),
    (
        "black",
        "Caution. The strongest case against, the failure mode, the thing that breaks in production.",
        "What would make {subject} fail, and which failure would be worst?",
    ),
    (
        "yellow",
        "Benefit. What is worth doing here, and what would be gained by getting it right?",
        "What is the upside of {subject} that a cautious plan would leave on the table?",
    ),
    (
        "green",
        "Alternatives. What has not been tried, and what could be done differently.",
        "What alternative approach to {subject} has not been tried yet?",
    ),
    (
        "blue",
        "Control. What is the next concrete step, what is it worth, and what would decide it.",
        "What is the cheapest next step on {subject}, and what result would count as an answer?",
    ),
)

#: What the pass is asked for, and how it is asked to answer. A list of bare
#: questions invites a list of bare answers; the shape is the difference between
#: a queue and a pile.
ANSWER_SHAPE = (
    "Reply with a JSON array and nothing else. Each element is an object with three keys:\n"
    '  {"hat": one of the six names above, "question": a single technical question, '
    '"why": one sentence on what answering it would change}\n'
    f"At most {MAX_CURATED_QUESTIONS} elements. Write a question only where you have a reason to "
    "want it answered; an empty hat is a better answer than a vague question. Every question must "
    "be answerable by looking things up, must name the specific thing it is about rather than "
    '"the system" or "the approach", and must stand on its own without this conversation. '
    "Never leave a template placeholder - a word in curly braces - in a question."
)

_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# The queue's own markers, imported by text rather than by import so this module
# stays independent of the research pass it feeds.
_OPEN_MARKERS = ("- [ ]", "* [ ]", "- [?]", "- [ ]:")
_DONE_MARKERS = ("- [x]", "* [x]", "- [x]")


def build_prompt(*, subject: str, user_model: str, lessons: str, already_asked: str) -> str:
    """The one request a curation pass makes.

    Everything it is given is something the agent already has: what the person
    is like, what has been learned, and what has already been asked. Nothing
    here asks the model to invent an interest - an invented interest is the
    failure this whole module exists to prevent.
    """
    # The subject is filled into every hat, never shown as a placeholder: a
    # `{subject}` left in the text is a word the model copies into its answer,
    # and a queue full of questions about a literal brace is worse than no
    # queue at all, because it looks like work.
    topic = subject.strip() or "whatever the material below points at, named by you in one phrase"
    hats = "\n".join(
        f"- {name}: {discipline} Ask: {template.format(subject=topic)}" for name, discipline, template in HATS
    )
    sections = [
        "Propose research questions for later. You are not answering them now.",
        "",
        f"Subject: {topic}",
        "",
        "Six hats, one discipline each:",
        hats,
        "",
        f"What is known about the person you work for:\n{user_model or '(nothing written yet)'}",
        "",
        f"What has been learned so far:\n{lessons or '(nothing stored yet)'}",
        "",
        f"Already asked and answered, do not repeat:\n{already_asked or '(nothing yet)'}",
        "",
        ANSWER_SHAPE,
    ]
    return "\n".join(sections)


def parse_questions(text: str) -> list[tuple[str, str, str]]:
    """Return ``(hat, question, why)`` for each usable element of the reply.

    Anything unusable is dropped rather than repaired: a question with no reason
    behind it is the thing that spends credits without learning, so a malformed
    entry costs the cycle one question instead of producing a vague one.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    match = _JSON_ARRAY.search(text)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    known = {name for name, _, _ in HATS}
    found: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        hat = str(item.get("hat", "")).strip().lower()
        question = _clean(str(item.get("question", "")))
        why = _clean(str(item.get("why", "")))
        if hat not in known or not question or not why:
            continue
        if _PLACEHOLDER.search(question) or _PLACEHOLDER.search(why):
            # A question the model filled in by copying the template is not a
            # question, and answering it would spend a lookup to learn nothing.
            continue
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append((hat, question, why))
        if len(found) >= MAX_CURATED_QUESTIONS:
            break
    return found


def _clean(value: str) -> str:
    """One field, flattened and bounded, so a line in the queue stays one line."""
    text = _CONTROL.sub(" ", value).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) > MAX_QUESTION_CHARS:
        text = text[: MAX_QUESTION_CHARS - 1].rstrip() + "…"
    return text


def has_material(subject: str, user_model: str, lessons: str) -> bool:
    """Whether there is anything real to be curious about.

    The hats are good at questions and better at making them sound reasonable,
    which is the problem: given nothing, they produce a confident set about a
    subject nobody has. Seen by running this for real - the queue filled with
    questions about zero-knowledge identity systems. So the question is asked of
    the material rather than answered by it.
    """
    return bool(subject.strip() or user_model.strip() or lessons.strip())


def open_question_count(text: str) -> int:
    """How many questions are already queued."""
    return sum(1 for line in text.splitlines() if line.strip().startswith(_OPEN_MARKERS))


def needs_curation(text: str, minimum: int = MIN_OPEN_QUESTIONS) -> bool:
    """Whether a cycle is allowed to add to the queue.

    Curating a queue that already has questions would bury the ones a person
    wrote down, which are the ones they meant.
    """
    return open_question_count(text) < max(1, minimum)


def render_entry(question: str, why: str) -> str:
    """One queue line in the format the research pass already reads."""
    return f"- [ ] {question}\n  why: {why}"


def append_questions(path: str, questions: list[tuple[str, str, str]]) -> int:
    """Append curated questions to the queue file, skipping what is already there.

    The file is the thing a person reads, so this never rewrites what is in it:
    it appends, and it refuses to add a question whose text is already queued.
    """
    if not questions:
        return 0
    try:
        with open(path, encoding="utf-8") as handle:
            existing = handle.read()
    except FileNotFoundError:
        existing = ""
    except OSError:
        return 0

    pending = open_question_count(existing)
    room = max(0, MIN_OPEN_QUESTIONS + MAX_CURATED_QUESTIONS - pending)
    added: list[str] = []
    queued_lower = {line.strip().lower() for line in existing.splitlines()}
    for hat, question, why in questions:
        del hat  # The hat named the question; the queue line does not carry it.
        if not added and room <= 0:
            break
        if f"- [ ] {question}".lower() in queued_lower:
            continue
        added.append(render_entry(question, why))
        queued_lower.add(f"- [ ] {question}".lower())
        room -= 1
    if not added:
        return 0

    header = "" if existing.endswith("\n") or not existing else "\n"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{header}{'\n'.join(added)}\n")
    except OSError:
        return 0
    return len(added)


def already_asked(text: str, limit: int = 12) -> str:
    """The answered questions, so a curation pass does not re-ask them."""
    answered: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.lower().startswith(_DONE_MARKERS):
            continue
        body = stripped[stripped.index("]") + 1 :].strip().lstrip("-*? ").strip()
        note = ""
        following = lines[index + 1].strip() if index + 1 < len(lines) else ""
        # The verdict the research pass writes under an answered question is an
        # em dash and some text. A line starting with a bullet is the next
        # question instead, and swallowing it would hand the model a question it
        # must believe was already answered.
        if following and not following.lower().startswith(_OPEN_MARKERS + _DONE_MARKERS):
            note = following
        answered.append(f"- {body} {note}".strip())
    return "\n".join(answered[-limit:])


def describe(questions: list[tuple[str, str, str]]) -> str:
    """One line per curated question, for the cycle record."""
    return "; ".join(f"[{hat}] {question}" for hat, question, _ in questions)
