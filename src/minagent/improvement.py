"""Reflect on a finished session and turn what it learned into improvements.

The memory layers decide what is worth keeping. This is the step after that:
what does the kept knowledge *imply*? A session where the same refusal
happened four times is not four facts, it is a hypothesis about a setting
that is too tight. Two turns that each rediscovered the same constraint are
not two memories, they are a pattern worth writing down before the next
session pays for it again.

Two destinations, and the difference between them is the whole design:

* **Proposals.** Every hypothesis reaches a document the user reads, with the
  evidence that supports it and the way to check whether it was right. Nothing
  here needs to be trusted to be useful.
* **Adjustments.** A hypothesis may also ask for one of a small, named set of
  settings to move, and only those move on their own. Everything is a number
  in a ``.env`` file with a minimum, a maximum and a cooldown, so a wrong
  hypothesis costs one wrong number rather than a broken agent. Code is never
  in the set: a model that rewrites its own source has no way to notice it made
  it worse.

The rule that makes the second half safe is evidence. A hypothesis with no
evidence gets written down and applied to nothing, because a conclusion with
nothing behind it is a guess and a guess should never move a setting.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .reflection import first_json_object

MAX_HYPOTHESES = 12
MAX_PROMPT_ENTRIES = 24
MAX_EXCERPT_CHARS = 500
# A trigger is a retrieval condition, not an essay. Long enough for a real
# precondition ("the user has corrected me twice in one turn"), short enough that
# someone scanning the prompt for when to apply a lesson can actually find the
# condition instead of inferring it.
MAX_TRIGGER_CHARS = 160
# One nudge per setting per cooldown. Without it a hypothesis that is wrong in
# one direction gets corrected by the next one in the other, and the setting
# oscillates forever while both hypotheses look supported.
ADJUSTMENT_COOLDOWN_SECONDS = 24 * 60 * 60
ADJUSTMENT_LOG_NAME = os.path.join(".minagent", "ajustes.json")

TARGETS = ("agent", "project")
KINDS = ("insight", "improvement")


@dataclass(frozen=True)
class SettingRule:
    """What a setting is allowed to be, and how far one nudge may move it.

    The bounds are not decoration. They are the difference between a bad
    hypothesis producing a worse setting and a bad hypothesis producing a
    setting the agent can still run on.
    """

    name: str
    minimum: int
    maximum: int
    step: int = 1
    why: str = ""


# The only settings the agent may move on its own. Each is a number with a
# sane range, each is about how it paces itself rather than what it can do, and
# each is written to .env so a restart keeps it and a human can see it.
SAFE_SETTINGS: dict[str, SettingRule] = {
    "MEMORY_REFLECTION_INTERVAL": SettingRule(
        name="MEMORY_REFLECTION_INTERVAL",
        minimum=3,
        maximum=120,
        why="How often the auto-captured log is culled. Lower when the store fills with noise.",
    ),
    "COMPUTE_QUEUE_LIMIT": SettingRule(
        name="COMPUTE_QUEUE_LIMIT",
        minimum=1,
        maximum=32,
        why="Heavy jobs that may wait. Lower when renders back up faster than they finish.",
    ),
    "COMPUTE_JOB_TIMEOUT_SECONDS": SettingRule(
        name="COMPUTE_JOB_TIMEOUT_SECONDS",
        minimum=120,
        maximum=7200,
        why="How long a heavy job may run before it is stopped. Raise when a real render hits the limit.",
    ),
    "COMPUTE_VOICE_TIMEOUT_SECONDS": SettingRule(
        name="COMPUTE_VOICE_TIMEOUT_SECONDS",
        minimum=30,
        maximum=600,
        why="The speaking budget, which is shorter than a render's. Raise when speech is cut off.",
    ),
}


@dataclass(frozen=True)
class Hypothesis:
    """One extrapolation, with the evidence that produced it.

    ``falsifier`` and ``trigger`` are the two fields that make this a
    scientific claim rather than a hunch, and both were missing when this
    dataclass was written. A hypothesis that cannot say what would prove it
    wrong is not weaker for admitting it - it is unfalsifiable, which means
    nothing can ever count as refuting it and so nothing ever will. A hypothesis
    with no trigger condition is worse: it cannot be retrieved at the moment it
    applies, so it is either always on and therefore ignored, or never on and
    therefore dead weight in every prompt that carries it.

    The literature is blunt about why these are not optional. The single most
    transferable recommendation in a recent critique of agentic AI-scientist
    systems was a centralised preregistration record: write the hypothesis down
    *with* its falsifier before the experiment, because a claim written after
    the result is a claim that can accommodate any outcome. And work on
    experience-derived heuristics found that the useful shape is a cause plus a
    guideline with an explicit condition attached - not a trajectory, not a
    summary, and not a paragraph.
    """

    title: str
    statement: str
    evidence: str
    expected: str
    verify: str
    target: str = "agent"
    kind: str = "improvement"
    setting: str = ""
    value: str = ""
    reason: str = ""
    falsifier: str = ""
    trigger: str = ""


@dataclass(frozen=True)
class Adjustment:
    """A setting move the agent is making on its own, with its before and after."""

    name: str
    previous: str
    proposed: str
    reason: str


@dataclass
class AdjustmentLog:
    """What was already moved, and when, so a nudge is not repeated."""

    entries: dict[str, float] = field(default_factory=dict)

    def recently_changed(self, name: str, now: float) -> bool:
        previous = self.entries.get(name)
        return previous is not None and now - previous < ADJUSTMENT_COOLDOWN_SECONDS

    def record(self, name: str, now: float) -> None:
        self.entries[name] = now

    def to_json(self) -> str:
        return json.dumps(self.entries, indent=2, sort_keys=True)

    @staticmethod
    def from_json(text: str) -> AdjustmentLog:
        try:
            payload = json.loads(text or "{}")
        except ValueError:
            # A corrupt log must not stop the agent from running; it only means
            # the cooldown is not known, so the worst case is one extra nudge.
            return AdjustmentLog()
        if not isinstance(payload, dict):
            return AdjustmentLog()
        entries: dict[str, float] = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
                entries[key] = float(value)
        return AdjustmentLog(entries=entries)


def load_adjustment_log(application_root: str) -> AdjustmentLog:
    """Read the cooldown file, treating anything unreadable as empty."""
    if not application_root:
        return AdjustmentLog()
    try:
        with open(os.path.join(application_root, ADJUSTMENT_LOG_NAME), encoding="utf-8") as handle:
            return AdjustmentLog.from_json(handle.read())
    except OSError:
        return AdjustmentLog()


def save_adjustment_log(application_root: str, log: AdjustmentLog) -> str:
    """Persist the cooldown so it survives the restart that follows the change."""
    if not application_root:
        return ""
    path = os.path.join(application_root, ADJUSTMENT_LOG_NAME)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(log.to_json())
    except OSError:
        return ""
    return path


def build_session_prompt(
    *,
    knowledge: Sequence[dict[str, Any]],
    log: Sequence[dict[str, Any]],
    stats: dict[str, str],
) -> list[dict[str, str]]:
    """Ask what this session teaches beyond what it recorded.

    The material is deliberately small: what was judged worth keeping, what is
    still sitting in the automatic log, and the handful of counters that show
    where the friction was. A session produces a lot of text and almost none of
    it is about how the agent itself should work differently.
    """

    def listed(entries: Sequence[dict[str, Any]]) -> str:
        return "\n".join(
            f"- {entry.get('title', '')}: {str(entry.get('content', ''))[:MAX_EXCERPT_CHARS]}"
            for entry in list(entries)[:MAX_PROMPT_ENTRIES]
        )

    counters = "\n".join(f"- {key}: {value}" for key, value in sorted(stats.items()))
    return [
        {
            "role": "system",
            "content": (
                "You are reflecting on a finished working session in order to improve how the agent works "
                "next time. Answer with one JSON object and nothing else.\n"
                'Write a hypothesis only when the material supports one. A hypothesis is an extrapolation: '
                "a pattern across several items, not a restatement of one of them.\n"
                "Fill in every field of every hypothesis. If you cannot name the specific items from the "
                "material that support one, do not write it: a hypothesis whose evidence field is empty or "
                "missing is dropped, and it is dropped for a reason.\n"
                '{"hypotheses": [{"title": "short name", "kind": "insight|improvement", "target": '
                '"agent|project", "statement": "what this session shows", "evidence": "the specific items '
                'that support it", "expected": "what would be better next time", "verify": "how to check '
                'it was right", "falsifier": "what would show this is wrong", "trigger": "when this '
                'applies", "setting": "a setting name or empty", "value": "a proposed value or '
                'empty", "reason": "why that value"}]}\n'
                "The target decides who acts on it. Use \"agent\" for anything about how the agent itself "
                "paces, schedules, retries, refuses or loads things - the GPU budget, the job queue, the "
                "timeouts, the voice, the memory it keeps - and \"project\" only for facts about the code, "
                "the models or the data the user is working with.\n"
                "Two fields decide whether a hypothesis is worth keeping.\n"
                "Falsifier: name the observation that would prove you wrong. A hypothesis you cannot refute "
                "is not a cautious hypothesis, it is an unfalsifiable one, and an unfalsifiable hypothesis "
                "can never be retired - it will sit in the prompt forever being agreed with. Write the "
                "falsifier before the evidence, not after: a claim written once the result is known can "
                "absorb any outcome, and that is the difference between a prediction and a rationalisation. "
                "\"Verify\" asks how to confirm it; \"falsifier\" asks how to kill it. They are not the same "
                "field and a hypothesis that only fills in the first one has not been tested.\n"
                "Trigger: name the condition under which this applies, in terms someone could notice in the "
                "moment. \"When the user corrects the agent twice in one turn\" is a trigger. \"When it is "
                "useful\" and \"in difficult situations\" are not - they describe no moment, so the lesson "
                "can never be applied on purpose and only gets applied by accident.\n"
                "A setting may only be proposed for target \"agent\", and only from this list, each with the "
                "evidence that would justify changing it. A setting proposed without a falsifier will still "
                "be tried and measured, and will be reverted if it makes things worse, but it can never be "
                "kept afterwards - so without a falsifier you are proposing work, not a change:\n"
                + "\n".join(
                    f"- {rule.name}: {rule.why} Allowed range {_current_display(rule)}."
                    for rule in sorted(SAFE_SETTINGS.values(), key=lambda item: item.name)
                )
                + "\nLeave setting and value empty unless the material names a specific number that should "
                "change, and never propose a value at either end of a range. An empty list is a correct "
                "answer when nothing generalises. Refusing to write a falsifier is also a correct answer: a "
                "hypothesis you cannot falsify is not ready to be proposed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"What happened this session:\n{counters or '- nothing was counted'}\n\n"
                f"Judged knowledge:\n{listed(knowledge) or '- none'}\n\n"
                f"Automatic log, not yet judged:\n{listed(log) or '- none'}"
            ),
        },
    ]


def parse_hypotheses(text: str) -> list[Hypothesis]:
    """Read the reflection, dropping anything that is not actually a hypothesis.

    Every field the agent will act on has to be present and non-empty, and the
    setting is checked against the allowlist here rather than at the point of
    use, so a proposed name the project never defined cannot reach the file.
    """
    payload = first_json_object(text)
    if payload is None:
        return []
    raw = payload.get("hypotheses")
    if not isinstance(raw, list):
        return []
    hypotheses: list[Hypothesis] = []
    for entry in raw[:MAX_HYPOTHESES]:
        if not isinstance(entry, dict):
            continue
        statement = _text(entry.get("statement"))
        evidence = _text(entry.get("evidence"))
        # The evidence floor is the whole safety story: no evidence, no
        # proposal, and certainly no setting change.
        if not statement or not evidence:
            continue
        target = _text(entry.get("target")).casefold()
        kind = _text(entry.get("kind")).casefold()
        setting = _text(entry.get("setting")).upper()
        value = _text(entry.get("value"))
        if setting not in SAFE_SETTINGS:
            setting, value = "", ""
        hypotheses.append(
            Hypothesis(
                title=_text(entry.get("title"))[:120] or statement[:60],
                statement=statement,
                evidence=evidence,
                expected=_text(entry.get("expected")),
                verify=_text(entry.get("verify")),
                target=target if target in TARGETS else "agent",
                kind=kind if kind in KINDS else "improvement",
                setting=setting,
                value=value,
                reason=_text(entry.get("reason")),
                falsifier=_text(entry.get("falsifier")),
                trigger=_text(entry.get("trigger"))[:MAX_TRIGGER_CHARS],
            )
        )
    return hypotheses


def can_promote(hypothesis: Hypothesis) -> bool:
    """Whether a hypothesis is allowed to become a lasting change.

    A setting may be *tried* without a falsifier, because trying it is safe: the
    trial in :mod:`minagent.measure` watches a window and reverts anything that
    makes things worse. What is not safe is *keeping* a change nobody can refute,
    because then there is no statement that could ever have come back wrong, and
    the only way out is someone editing the file by hand.

    So the falsifier gates promotion, not application. The first version of this
    rule dropped the setting outright, which read as the safer choice and was not:
    it quietly disabled the revert path, because no trial was ever recorded for
    a change that had been discarded. A guard that removes the thing it guards is
    not a guard.
    """
    return bool(hypothesis.falsifier.strip())


def plan_adjustments(
    hypotheses: Sequence[Hypothesis],
    current: dict[str, str],
    log: AdjustmentLog,
    now: float,
) -> list[Adjustment]:
    """Turn the safe hypotheses into setting moves that respect the bounds.

    Refused here rather than at the point of change: a setting outside its
    range, a value that is not a number, a target that is not the agent, a
    setting already moved inside the cooldown, and a second move of the same
    setting in the same pass.
    """
    planned: list[Adjustment] = []
    seen: set[str] = set()
    for hypothesis in hypotheses:
        name = hypothesis.setting
        if not name or name in seen or name not in SAFE_SETTINGS:
            continue
        seen.add(name)
        if hypothesis.target != "agent" or not hypothesis.evidence:
            continue
        # An insight is something noticed; an improvement is a prescription. A
        # number is a prescription, and the distinction does real work here: a
        # model asked to reflect on a session where a render hit the limit
        # observed "renders take longer than the timeout" - an insight - and
        # then proposed a shorter timeout anyway, which is the opposite of what
        # its own evidence said. The bounds would have contained that, not
        # prevented it.
        if hypothesis.kind != "improvement":
            continue
        if log.recently_changed(name, now):
            continue
        rule = SAFE_SETTINGS[name]
        proposed = _as_int(hypothesis.value)
        if proposed is None:
            continue
        previous = _as_int(current.get(name))
        if previous is None or proposed == previous:
            # Without a known starting point there is no way to tell a nudge
            # from a jump, so the setting is left where the user put it.
            continue
        if not rule.minimum <= proposed <= rule.maximum:
            continue
        if proposed == rule.minimum:
            # Pinning a setting to its floor is a reaction, not a considered
            # value: it is what a model reaches for when it wants the problem
            # to go away rather than to be smaller.
            continue
        if abs(proposed - previous) > (rule.maximum - rule.minimum) // 2:
            # A move of more than half the allowed span is not a nudge, it is a
            # different setting wearing the same name.
            continue
        planned.append(
            Adjustment(
                name=name,
                previous=str(previous),
                proposed=str(proposed),
                reason=hypothesis.reason or hypothesis.title,
            )
        )
    return planned


def apply_adjustments(application_root: str, adjustments: Sequence[Adjustment]) -> tuple[str, str]:
    """Write the moves into the project ``.env`` and return what changed.

    Returns the file and the list of ``NAME=old -> new`` moves. A write that
    fails is reported, never raised: a reflection that cannot be applied is
    still a reflection the user can read.
    """
    if not application_root or not adjustments:
        return "", ""
    path = os.path.join(application_root, ".env")
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except FileNotFoundError:
        lines = []
    except OSError:
        return "", ""

    applied: list[str] = []
    for adjustment in adjustments:
        replacement = f"{adjustment.name}={adjustment.proposed}"
        pattern = re.compile(rf"^\s*{re.escape(adjustment.name)}\s*=")
        for index, line in enumerate(lines):
            if pattern.match(line):
                applied.append(f"{adjustment.name}: {adjustment.previous} -> {adjustment.proposed}")
                lines[index] = replacement
                break
        else:
            lines.append(replacement)
            applied.append(f"{adjustment.name}: unset -> {adjustment.proposed}")
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError:
        return "", ""
    return path, "\n".join(applied)


def format_document_section(hypotheses: Sequence[Hypothesis], applied: Sequence[str] = ()) -> str:
    """One markdown section per reflection, ready to append to the document."""
    if not hypotheses:
        return ""
    stamp = time_stamp()
    lines = [f"## Reflexión del {stamp}", ""]
    for hypothesis in hypotheses:
        marker = "AJA" if hypothesis.kind == "insight" else "Mejora"
        lines.append(f"### {marker} · {hypothesis.title}")
        lines.append(f"*Ámbito:* {hypothesis.target}")
        lines.append(f"*{hypothesis.statement}*")
        lines.append(f"- **Evidencia:** {hypothesis.evidence}")
        if hypothesis.expected:
            lines.append(f"- **Efecto esperado:** {hypothesis.expected}")
        if hypothesis.verify:
            lines.append(f"- **Cómo comprobarla:** {hypothesis.verify}")
        if hypothesis.setting and hypothesis.value:
            rule = SAFE_SETTINGS[hypothesis.setting]
            lines.append(
                f"- **Ajuste propuesto:** `{hypothesis.setting}={hypothesis.value}` "
                f"(permitido {rule.minimum}-{rule.maximum}: {rule.why})"
            )
        lines.append("")
    if applied:
        lines.append("### Ajustes aplicados")
        lines.extend(f"- {entry}" for entry in applied)
        lines.append("")
    return "\n".join(lines)


DOCUMENT_NAME = "MEJORAS.md"
DOCUMENT_HEADER = """# MEJORAS

Lo que el agente ha deducido de sus propias sesiones, y lo que ha cambiado por
ello. Cada reflexión anade una seccion al final con la evidencia que la sostiene
y como comprobarla, para que una hipotesis equivocada se pueda leer y borrar
como lo que es.

Este archivo lo escribe el agente. Editarlo a mano funciona: la siguiente
reflexion lo anade al final y no toca lo que ya esta escrito.
"""


def append_document(application_root: str, section: str) -> str:
    """Add one reflection to the document, creating it with a header if needed.

    Appended rather than rewritten, and never truncated: a proposal the user
    has not read yet is not one the agent may quietly delete on the next pass.
    """
    if not application_root or not section.strip():
        return ""
    path = os.path.join(application_root, DOCUMENT_NAME)
    existing = ""
    try:
        with open(path, encoding="utf-8") as handle:
            existing = handle.read()
    except FileNotFoundError:
        existing = ""
    except OSError:
        return ""
    try:
        with open(path, "a", encoding="utf-8") as handle:
            if not existing.strip():
                # The header explains what the file is and that a human may edit
                # it, which is the only thing that makes it a document rather
                # than a log.
                handle.write(DOCUMENT_HEADER)
                if not DOCUMENT_HEADER.endswith("\n"):
                    handle.write("\n")
            else:
                handle.write("\n" if not existing.endswith("\n") else "")
            handle.write(section.strip() + "\n")
    except OSError:
        return ""
    return path


def read_document(application_root: str, limit: int = 120) -> str:
    """The last few lines of the document, which is what ``/mejoras`` shows."""
    if not application_root:
        return ""
    try:
        with open(os.path.join(application_root, DOCUMENT_NAME), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def format_adjustment_report(
    applied: Sequence[str], proposed: Sequence[str], count: int = 0
) -> str:
    """One line for the transcript saying what was concluded and what moved.

    The hypothesis count comes first on purpose. A session that taught something
    and changed nothing must not report the same as a session that taught
    nothing, or the useful case reports as silence.
    """
    parts: list[str] = [f"{count} hipótesis"] if count else []
    if applied:
        parts.append("autoajustó " + ", ".join(applied))
    if proposed:
        parts.append("propuso " + ", ".join(proposed))
    return "; ".join(parts)


def _current_display(rule: SettingRule) -> str:
    """A range to aim at, when the live value is not in the prompt already."""
    return f"between {rule.minimum} and {rule.maximum}"


def _text(value: Any) -> str:
    """Read one field as a line of text, whether the model sent a string or a list.

    A model asked for evidence naturally answers with a list of the items it
    found - which is what it did here, unprompted, with the best hypothesis of
    the session in it. A parser that only accepts strings throws that away and
    reports that the session taught nothing, which is the worst way to be
    wrong here.
    """
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return " ".join(str(value).split())
    if isinstance(value, (list, tuple)):
        parts = [_text(item) for item in value]
        return "; ".join(part for part in parts if part)
    return ""


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def time_stamp() -> str:
    """A readable local timestamp for the document headings."""
    from datetime import datetime

    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
