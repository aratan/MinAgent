"""Long work the agent plans for itself: a flow of tool steps, checkpointed.

The turn loop already runs many tool calls in a row, so a task of ten steps
needs nothing new. A task that takes twenty minutes needs three things the turn
loop cannot give it, and none of them is about being slower:

1. **The plan outlives the turn.** The flow keeps going after the context was
   compacted, after ``Esc``, and after the process was restarted. So it lives in
   ``.minagent/flows.json`` and every step is checkpointed the moment it
   finishes, which is the only place in this codebase where a half-finished job
   is worth writing to disk.
2. **One step consumes the next one's input.** Reading a file and then editing
   it, generating a video and then publishing it. The runner resolves
   ``{{steps.2.output}}`` into the next step's arguments, so the plan is written
   once, before the results exist.
3. **The agent stays in charge.** This is deliberately not a plan the model
   hands over and forgets: the runner stops and returns control at every
   decision point, at every failure, and every ``FLOW_BATCH_STEPS``, so the
   model can correct a step that did not work instead of watching the rest of a
   wrong plan run to the end.

What a flow is not: it is not a way to skip the tools the session was gated on.
Every step goes through the same ``execute_tool`` the turn loop uses, so the
terminal still asks in ``TERMINAL_MODE=ask`` and every MCP server still runs
under its own approval rules. The flow adds a plan and a checkpoint, nothing
else.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .context import compress_for_context
from .errors import AgentError

FLOWS_FILE_NAME = os.path.join(".minagent", "flows.json")

FLOW_STORE_VERSION = 1

MAX_FLOW_STEPS = 40
"""The longest plan the model may write in one call.

Forty tool calls is far past what one turn should hold, and a plan longer than
this is a sign the model is trying to guess a task it should ask about instead.
"""

MAX_FLOW_ARG_CHARS = 40000
MAX_OBJECTIVE_CHARS = 2000
MAX_NOTE_CHARS = 300
MAX_CONDITION_CHARS = 200
MAX_ARG_DEPTH = 8

STEP_OUTPUT_CHARS = 4000
"""What one step's result keeps inline in the flow record.

The record is the only copy of a step's result once the turn that ran it is
gone, so a result larger than this is not shortened: it goes to the same
off-window archive ``recall_tool_output`` reads, and the record keeps a
head-and-tail preview naming the reference. Anything that is dropped from the
record this way can be read back exactly, which is the difference between a
flow that can be inspected a week later and one that cannot.
"""

STEP_ERROR_CHARS = 2000
"""The inline budget for a failure message, archived on the same terms."""

MAX_REPORT_OUTPUTS = 3
"""How many recent step outputs the tool result carries in full."""

DEFAULT_BATCH_STEPS = 8
"""How many steps run before control comes back to the model unasked.

The runner exists to save round trips, and the number it saves them by is
exactly this. Eight is a balance measured against what a plan is for: long
enough that a mechanical sequence of reads and writes costs one request rather
than eight, short enough that a plan going wrong is caught while the wrong step
is still the one that just ran.
"""

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_ABANDONED = "abandoned"

ACTIVE_STATUSES = (STATUS_RUNNING, STATUS_PAUSED, STATUS_FAILED)

STEP_DONE = "done"
STEP_FAILED = "failed"
STEP_SKIPPED = "skipped"
STEP_BLOCKED = "blocked"

RECORD_MARK = {STEP_DONE: "ok", STEP_FAILED: "fail", STEP_SKIPPED: "skip", STEP_BLOCKED: "wait"}

PLACEHOLDER = re.compile(r"\{\{\s*steps\.(\d+)\.(output|status|error)\s*\}\}")

_CONDITION = re.compile(
    r"^\s*(?P<negate>!)?\s*steps\s*\.\s*(?P<index>\d+)\s*\.\s*"
    r"(?P<field>ok|failed|status|output|error)"
    r"(?:\s*(?P<operator>==|!=|not\s+contains|contains)\s*(?P<value>.*?))?\s*$",
    re.IGNORECASE,
)

_TRUTHY = ("true", "yes", "1")
_FALSY = ("false", "no", "0", "")

TERMINAL_TOOL_NAME = "run_terminal"
_EXIT_CODE = re.compile(r"^Exit code: (\d+)")

MAX_FLOWS_PER_WORKSPACE = 20

RUNNER_TOOL_NAMES = ("run_flow", "flow_continue")
"""The tools a step may not name: a flow that starts a flow never ends."""


class FlowBlocked(Exception):
    """A step needs the output of a step that has not run yet.

    Not an error the model should retry: it is a plan that guessed. The message
    names the step whose output is missing so the correction is to insert a
    decision point before it, not to run the same call again.
    """


@dataclass(frozen=True, slots=True)
class Condition:
    """One ``when`` clause, read into something that can be answered.

    The grammar is deliberately tiny - a step index, a field, and at most a
    comparison - because the model writing it is a small local model and a
    general expression language would be one more thing to get wrong in a plan
    that is already hard to keep straight. Anything a single clause cannot say
    is what ``decide`` exists for.
    """

    text: str
    index: int
    """The zero-based record this clause reads, which is step N+1 in the plan."""

    field: str
    """``ok``, ``failed``, ``status``, ``output`` or ``error``."""

    negate: bool
    operator: str
    value: str
    error: str = ""
    """Set when the clause could not be read, which pauses the flow rather than
    running a step whose guard nobody understands."""

    def expects_failure(self) -> bool:
        """Is this clause only ever true when the step it reads did not work?

        That combination is the one the model writes and cannot get right: a
        guard like ``!steps.2.ok`` is the obvious way to say "if the tests
        failed", but a failed step stops the flow, so the branch it guards is
        the one step the flow can never reach. Detecting it while the plan is
        still being written is the only point where the fix is one word.
        """
        if self.field == "ok":
            return self.negate
        if self.field == "failed":
            return True
        if self.field == "status":
            return self.value.casefold() == "failed"
        if self.field == "error":
            return self.operator == "truthy" and not self.negate
        return False


CONDITION_GRAMMAR = (
    'A condition is steps.N.ok, !steps.N.ok, steps.N.status == done, steps.N.output contains "text", '
    'steps.N.output (meaning "it produced something"), or !steps.N.error. N counts steps from 1. '
    '"always", or no "when" at all, runs the step.'
)


@dataclass(frozen=True, slots=True)
class Step:
    """One tool call the agent planned, before its arguments are known."""

    tool: str
    args: dict[str, Any]
    note: str = ""
    decide: bool = False
    """Stop here and let the model look at what this step produced.

    This is the point of the whole file: a plan that keeps running through a
    result it did not expect is a plan that is already failing, and the only
    way to find out is to stop and look.
    """

    optional: bool = False
    """A step that may fail without stopping the flow.

    For the steps whose failure is a legitimate answer - a file that may not
    exist, a search that may find nothing. Without it, every optional step is a
    branch the model has to write out itself.
    """

    when: str = ""
    """The guard, as written. Empty means the step always runs."""

    condition: Condition | None = None


@dataclass(slots=True)
class StepRecord:
    """What actually happened to one step, and what it returned."""

    index: int
    tool: str
    note: str = ""
    status: str = STEP_DONE
    output: str = ""
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "tool": self.tool,
            "note": self.note,
            "status": self.status,
            "output": self.output,
            "error": self.error,
        }

    @staticmethod
    def from_json(payload: Any) -> StepRecord | None:
        if not isinstance(payload, dict):
            return None
        status = str(payload.get("status") or STEP_DONE)
        if status not in RECORD_MARK:
            status = STEP_DONE
        return StepRecord(
            index=_as_index(payload.get("index")),
            tool=str(payload.get("tool") or ""),
            note=str(payload.get("note") or ""),
            status=status,
            output=str(payload.get("output") or ""),
            error=str(payload.get("error") or ""),
        )


@dataclass(slots=True)
class Flow:
    """A plan, how far it got, and where it stopped."""

    id: str
    objective: str
    steps: list[Step] = field(default_factory=list)
    cursor: int = 0
    status: str = STATUS_RUNNING
    records: list[StepRecord] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    workspace: str = ""
    archived: list[str] = field(default_factory=list)
    """Archive references for step outputs too large to keep in the record.

    Held separately from the records because a record keeps a preview that
    already names them, and the model has to be able to find those names again
    once the preview that carried them has scrolled out of the report.
    """

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    @property
    def remaining(self) -> int:
        return max(0, len(self.steps) - self.cursor)

    def record_for(self, index: int) -> StepRecord | None:
        for record in self.records:
            if record.index == index:
                return record
        return None

    def describe_next(self) -> str:
        """The next step in one line, for the prompt section and ``/flow``."""
        if self.cursor >= len(self.steps):
            return "nothing left to run"
        step = self.steps[self.cursor]
        marker = "decide here" if step.decide else ""
        note = f" · {step.note}" if step.note else ""
        guard = f" when {step.when}" if step.when else ""
        return f"{self.cursor + 1}. {step.tool}({', '.join(sorted(step.args)) or 'no arguments'}){guard}{note} {marker}".strip()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "status": self.status,
            "cursor": self.cursor,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": [
                {
                    "tool": step.tool,
                    "args": step.args,
                    "note": step.note,
                    "decide": step.decide,
                    "optional": step.optional,
                    "when": step.when,
                }
                for step in self.steps
            ],
            "records": [record.to_json() for record in self.records],
            "archived": list(self.archived),
        }

    @staticmethod
    def from_json(payload: Any) -> Flow | None:
        if not isinstance(payload, dict) or not isinstance(payload.get("objective"), str):
            return None
        steps = [step for step in (_step_from_json(item) for item in payload.get("steps") or []) if step]
        status = str(payload.get("status") or STATUS_PAUSED)
        if status not in (STATUS_RUNNING, STATUS_PAUSED, STATUS_DONE, STATUS_FAILED, STATUS_ABANDONED):
            status = STATUS_PAUSED
        cursor = _as_index(payload.get("cursor"))
        records = [record for record in (StepRecord.from_json(item) for item in payload.get("records") or []) if record]
        archived = payload.get("archived")
        return Flow(
            id=str(payload.get("id") or "1"),
            objective=str(payload["objective"]),
            steps=steps,
            cursor=min(cursor, len(steps)),
            status=status,
            records=records,
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
            workspace=str(payload.get("workspace") or ""),
            archived=[str(item) for item in archived if isinstance(item, str)] if isinstance(archived, list) else [],
        )


StepExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
FlowSaver = Callable[[Flow], None]
CancelledCheck = Callable[[], bool]
# Decides whether a result that returned counts as done or failed, given the tool
# and the text it produced. The default only knows about the terminal's exit code.
StepJudge = Callable[[str, str], str]
# Stores an oversized result off-window and returns its reference, or None when it
# could not be stored. The app passes the same archive ``recall_tool_output``
# reads, so a step's full output stays recoverable like any other tool result.
ArchiveStore = Callable[[str], str | None]
# Called as ``on_step(index, step, arguments)`` before each step runs, so the
# terminal shows the step rather than a silent pause.
StepAnnouncer = Callable[[int, Step, dict[str, Any]], None]


# ------------------------------------------------------------------ validation


def _as_index(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _as_text(value: Any, limit: int, label: str) -> str:
    if not isinstance(value, str):
        raise AgentError(f"{label} must be text.")
    text = value.strip()
    if len(text) > limit:
        raise AgentError(f"{label} is too long ({len(text)} characters; the limit is {limit}).")
    return text


def _as_flag(value: Any, label: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "on"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "off", ""):
        return False
    raise AgentError(f"{label} must be true or false.")


def _validate_arguments(value: Any, label: str, depth: int = 0) -> Any:
    """Check the argument tree before anything is executed from it.

    A flow argument is data the model wrote and the runner hands to a tool, so
    it is bounded here rather than at the tool: a runaway structure fails as a
    readable message instead of as a stack trace twenty minutes into a plan.
    """
    if depth > MAX_ARG_DEPTH:
        raise AgentError(f"{label} nests deeper than {MAX_ARG_DEPTH} levels.")
    if isinstance(value, str):
        if len(value) > MAX_FLOW_ARG_CHARS:
            raise AgentError(
                f"{label} is {len(value)} characters long; a step argument is limited to {MAX_FLOW_ARG_CHARS}."
            )
        return value
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise AgentError(f"{label} has an argument name that is not text.")
            cleaned[key] = _validate_arguments(item, f"{label}.{key}", depth + 1)
        return cleaned
    if isinstance(value, list):
        return [_validate_arguments(item, f"{label}[{position}]", depth + 1) for position, item in enumerate(value)]
    raise AgentError(f"{label} must be text, a number, a list, or an object.")


def parse_steps(
    raw: Any,
    *,
    known_tools: set[str],
    forbidden: Sequence[str] = RUNNER_TOOL_NAMES,
    label: str = "steps",
) -> list[Step]:
    """Read the model's plan, or say exactly what is wrong with it.

    ``known_tools`` is every tool the session can reach, loaded or not: a flow
    step naming a tool whose capability is not loaded is a normal thing to
    write, and the runner loads it the same way the turn loop does. What is
    rejected is a name that is not a tool at all, because that is the failure
    mode a small model walks into, and it should be told before twenty steps run,
    and the runner itself, which has no end and no place to checkpoint.
    """
    if not isinstance(raw, list) or not raw:
        raise AgentError(f"{label} must be a non-empty list of steps.")
    if len(raw) > MAX_FLOW_STEPS:
        raise AgentError(f"A flow may hold at most {MAX_FLOW_STEPS} steps; this one has {len(raw)}.")
    blocked_names = set(forbidden)
    steps: list[Step] = []
    for position, item in enumerate(raw):
        number = position + 1
        if isinstance(item, str):
            # A model that writes "1. read_file(path)" instead of an object is
            # saved by taking the tool name and telling it what to fix.
            raise AgentError(
                f"Step {number} is text, not an object. Each step needs tool and args, for example "
                f'{{"tool": "read_file", "args": {{"path": "README.md"}}}}. Got {item[:80]!r}.'
            )
        if not isinstance(item, dict):
            raise AgentError(f"Step {number} must be an object with tool and args.")
        tool = _as_text(item.get("tool"), MAX_NOTE_CHARS, f"Step {number} tool")
        if not tool:
            raise AgentError(f"Step {number} has no tool name.")
        if tool in blocked_names:
            raise AgentError(
                f"Step {number} calls {tool}, which is the flow runner itself. A flow runs tools, "
                f"not itself; control comes back to you between steps."
            )
        if tool not in known_tools:
            raise AgentError(
                f"Step {number} calls {tool}, which is not a tool. Use a tool name exactly as it is "
                f"written in the capability index."
            )
        raw_args = item.get("args", {})
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            raise AgentError(f"Step {number} args must be an object of named arguments, not a list or text.")
        when = _as_text(item.get("when", ""), MAX_CONDITION_CHARS, f"Step {number} when")
        condition: Condition | None = None
        if when:
            try:
                condition = parse_condition(when)
            except AgentError as unreadable:
                raise AgentError(f"Step {number}: {unreadable.message}") from unreadable
            # Both wrong orders are refused here rather than at run time: a self
            # reference is a loop, and a guard that reads a later step is a plan
            # written from the end backwards.
            if condition.index == position:
                raise AgentError(
                    f'Step {number} is conditional on "{when}", which is itself. A guard reads an '
                    f"earlier step."
                )
            if condition.index > position:
                raise AgentError(
                    f'Step {number} is conditional on "{when}", but step {condition.index + 1} comes '
                    f"later and has not run yet. A guard reads a step before it."
                )
            if condition.expects_failure() and not steps[condition.index].optional:
                raise AgentError(
                    f'Step {number} reads a failure of step {condition.index + 1} with "{when}", but '
                    f"that step is not optional: when it fails the flow stops there, so this step would "
                    f'never run. Mark step {condition.index + 1} "optional": true to let a failure '
                    f"pass through to the branch, or mark the step before it \"decide\": true and write "
                    f"this step after reading the result."
                )
        steps.append(
            Step(
                tool=tool,
                args=_validate_arguments(raw_args, f"Step {number} args"),
                note=_as_text(item.get("note", ""), MAX_NOTE_CHARS, f"Step {number} note"),
                decide=_as_flag(item.get("decide"), f"Step {number} decide"),
                optional=_as_flag(item.get("optional"), f"Step {number} optional"),
                when=when,
                condition=condition,
            )
        )
    return steps


def make_flow(
    *,
    flow_id: str,
    objective: str,
    steps: Sequence[Step],
    workspace: str,
    now: str | None = None,
) -> Flow:
    stamp = now or time.strftime("%Y-%m-%d %H:%M:%S")
    return Flow(
        id=flow_id,
        objective=objective,
        steps=list(steps),
        cursor=0,
        status=STATUS_RUNNING,
        created_at=stamp,
        updated_at=stamp,
        workspace=workspace,
    )


# ----------------------------------------------------------------- conditions


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_condition(text: str) -> Condition:
    """Read one ``when`` clause, or say what a readable one looks like.

    The error is a grammar rather than a position in a regular expression: the
    caller is a model that has just written the clause, so the useful reply is
    one it can act on, and a clause it cannot parse must never run a step
    silently.
    """
    raw = text.strip()
    if raw.lower() in ("", "always", "true"):
        return Condition(text=raw, index=-1, field="", negate=False, operator="always", value="")
    match = _CONDITION.match(raw)
    if match is None:
        raise AgentError(f'A "when" clause cannot be read: {raw!r}. {CONDITION_GRAMMAR}')
    operator = (match.group("operator") or "").lower()
    if operator == "not contains":
        operator = "not contains"
    field = match.group("field").lower()
    value = _unquote(match.group("value") or "")
    if field in ("ok", "failed"):
        # A boolean field compared against text is a clause nobody can answer.
        if operator in ("contains", "not contains"):
            raise AgentError(
                f'steps.{match.group("index")}.{field} is true or false, so it cannot be compared with '
                f'"contains". Use steps.N.ok or steps.N.status == done instead. {CONDITION_GRAMMAR}'
            )
        if operator in ("==", "!=") and value.lower() not in _TRUTHY + _FALSY:
            raise AgentError(
                f'steps.{match.group("index")}.{field} is true or false, not {value!r}. Write '
                f"steps.N.ok or !steps.N.ok. {CONDITION_GRAMMAR}"
            )
        operator = "is" if operator == "==" else "is not" if operator == "!=" else "is"
    elif not operator:
        # No comparison means "it produced something", except for a status, where
        # the only sensible reading of a bare "steps.N.status" is "it worked".
        operator = "==" if field == "status" else "truthy"
        value = "done" if field == "status" else ""
    return Condition(
        text=raw,
        index=int(match.group("index")) - 1,
        field=field,
        negate=bool(match.group("negate")),
        operator=operator,
        value=value,
    )


def evaluate_condition(condition: Condition, flow: Flow) -> bool:
    """Answer a guard, or refuse to guess.

    A clause that reads a step which has not run is a plan written in the wrong
    order, and running the step anyway is how a flow deletes a file it meant to
    inspect. It pauses instead, naming the step that has to come first.
    """
    if condition.error:
        raise FlowBlocked(condition.error)
    record = flow.record_for(condition.index)
    if record is None or condition.index >= flow.cursor:
        raise FlowBlocked(
            f'A step is conditional on "{condition.text}" but step {condition.index + 1} has not run. '
            f"A guard reads a step before it, so move the condition's step earlier or mark the last "
            f"step before it with \"decide\": true and write this step once you have seen that result."
        )
    if condition.field == "ok":
        holds = record.status == STEP_DONE
    elif condition.field == "failed":
        holds = record.status == STEP_FAILED
    elif condition.field == "status":
        holds = record.status.casefold() == condition.value.casefold()
    elif condition.field == "error":
        holds = bool(record.error)
    else:
        holds = bool(record.output)
    if condition.operator in ("is not", "!="):
        holds = not holds
    elif condition.operator == "contains":
        holds = condition.value.casefold() in record.output.casefold()
    elif condition.operator == "not contains":
        holds = condition.value.casefold() not in record.output.casefold()
    return not holds if condition.negate else holds


# --------------------------------------------------------------- substitution


def resolve_arguments(value: Any, flow: Flow) -> Any:
    """Fill ``{{steps.N.output}}`` from the steps that have already run.

    Raises ``FlowBlocked`` rather than substituting an empty string, because an
    empty string here is a silent wrong answer: the step would run with a path
    that is not there and fail somewhere further from the cause.
    """
    if isinstance(value, str):
        return _resolve_text(value, flow)
    if isinstance(value, dict):
        return {key: resolve_arguments(item, flow) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_arguments(item, flow) for item in value]
    return value


def _resolve_text(text: str, flow: Flow) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        field = match.group(2)
        record = flow.record_for(index)
        if record is None or index >= flow.cursor:
            raise FlowBlocked(
                f"A step refers to {{{{steps.{index}.{field}}}}} but step {index + 1} has not run. "
                f"Mark the last step before it with \"decide\": true, run the flow, and write the "
                f"next steps once you have read that output."
            )
        if field == "status":
            return record.status
        if field == "error":
            return record.error
        return record.output

    return PLACEHOLDER.sub(replace, text)


# ------------------------------------------------------------------ the runner


def _result_text(result: Any) -> str:
    if isinstance(result, dict):
        if isinstance(result.get("tool_text"), str):
            return result["tool_text"]
        if result.get("image") or result.get("images"):
            names = [str(image.get("path", "?")) for image in result.get("images") or []]
            if result.get("image"):
                names.append(str((result.get("image") or {}).get("path", "?")))
            return "Image result: " + ", ".join(name for name in names if name)
        return json.dumps(result, ensure_ascii=False, default=str)
    return str(result)


def _keep(text: str, limit: int, archive: ArchiveStore | None, what: str) -> tuple[str, str | None]:
    """Compress a result and, if it is still too big, archive it rather than cut it.

    The record keeps a head-and-tail preview either way, but the two cases are
    not the same: with an archive the omitted text is one ``recall_tool_output``
    call away, and without one it is gone for good. That difference is why the
    note says which one happened instead of always talking about the limit.
    """
    text = compress_for_context(text).strip()
    if len(text) <= limit:
        return text, None
    head = limit * 3 // 4
    tail = limit - head
    omitted = len(text) - limit
    reference = archive(text) if archive is not None else None
    if reference is None:
        return (
            text[:head]
            + f"\n\n[{what} shortened for the flow record: {omitted} of {len(text)} characters omitted "
            f"and could not be archived, so they are lost. Read the file the step wrote, or run the "
            f"step again with a narrower command.]\n\n"
            + text[-tail:]
        ), None
    return (
        text[:head]
        + f"\n\n[{what} shortened for the flow record: {omitted} of {len(text)} characters omitted. "
        f'Nothing was lost: call recall_tool_output with id="{reference}" and an offset/limit to read '
        f"any part of it.]\n\n"
        + text[-tail:]
    ), reference


def judge_result(tool: str, text: str) -> str:
    """Did this step succeed, for a call that returned rather than raised?

    A command that exits 1 did not work, whatever the tool reports, and a flow
    that calls it ``ok`` makes ``when: "!steps.2.ok"`` quietly mean the opposite
    of what the model meant by it. The turn loop shows the same exit code to the
    user, so the flow would otherwise be the one place a failure reads as a
    success. Narrow on purpose: only the terminal has an exit code to read, and
    every other tool signals failure by raising.
    """
    if tool == TERMINAL_TOOL_NAME:
        match = _EXIT_CODE.match(text)
        if match is not None and match.group(1) != "0":
            return STEP_FAILED
    return STEP_DONE


def _error_text(error: BaseException) -> str:
    message = getattr(error, "message", None)
    text = message if isinstance(message, str) and message else str(error) or error.__class__.__name__
    if getattr(error, "may_have_changed", False):
        text += " The target may have changed despite this error; inspect it before relying on it."
    return text


async def advance_flow(
    flow: Flow,
    execute: StepExecutor,
    *,
    save: FlowSaver,
    archive: ArchiveStore | None = None,
    judge: StepJudge | None = None,
    is_cancelled: CancelledCheck | None = None,
    on_step: StepAnnouncer | None = None,
    batch_limit: int = DEFAULT_BATCH_STEPS,
    now: str | None = None,
) -> None:
    """Run the flow until it needs the model, and checkpoint every step.

    Control comes back for exactly four reasons, and each of them is a place
    where a plan that kept going would have been wrong: a step marked
    ``decide``, a step that failed, a step whose input does not exist yet, and
    the batch limit. Everything else runs without a round trip.
    """
    stamp = now or time.strftime("%Y-%m-%d %H:%M:%S")
    flow.status = STATUS_RUNNING
    ran = 0
    while flow.cursor < len(flow.steps):
        if is_cancelled is not None and is_cancelled():
            flow.status = STATUS_PAUSED
            flow.updated_at = stamp
            save(flow)
            return
        if ran >= max(1, batch_limit):
            flow.status = STATUS_PAUSED
            flow.updated_at = stamp
            save(flow)
            return
        index = flow.cursor
        step = flow.steps[index]
        if step.condition is not None:
            try:
                met = evaluate_condition(step.condition, flow)
            except FlowBlocked as blocked:
                flow.records.append(
                    StepRecord(index=index, tool=step.tool, note=step.note, status=STEP_BLOCKED, error=str(blocked))
                )
                flow.status = STATUS_PAUSED
                flow.updated_at = stamp
                save(flow)
                return
            if not met:
                # A step no guard let through costs nothing, so it neither runs a
                # tool nor spends a slot of the batch: a plan made mostly of
                # conditions should reach the end, not stop early for having
                # skipped a lot.
                flow.records.append(
                    StepRecord(
                        index=index,
                        tool=step.tool,
                        note=step.note,
                        status=STEP_SKIPPED,
                        error=f"Skipped: {step.condition.text} was not true.",
                    )
                )
                flow.cursor = index + 1
                flow.updated_at = stamp
                save(flow)
                continue
        try:
            arguments = resolve_arguments(step.args, flow)
        except FlowBlocked as blocked:
            flow.records.append(
                StepRecord(index=index, tool=step.tool, note=step.note, status=STEP_BLOCKED, error=str(blocked))
            )
            flow.status = STATUS_PAUSED
            flow.updated_at = stamp
            save(flow)
            return
        if on_step is not None:
            on_step(index, step, arguments)
        try:
            result = await execute(step.tool, arguments)
        except Exception as error:  # noqa: BLE001 - any tool failure is a step failure
            failure, reference = _keep(_error_text(error), STEP_ERROR_CHARS, archive, "error")
            if reference:
                flow.archived.append(reference)
            flow.records.append(
                StepRecord(index=index, tool=step.tool, note=step.note, status=STEP_FAILED, error=failure)
            )
            if step.optional:
                flow.cursor = index + 1
                flow.updated_at = stamp
                ran += 1
                save(flow)
                continue
            flow.status = STATUS_FAILED
            flow.updated_at = stamp
            save(flow)
            return
        text = _result_text(result)
        denied = text.startswith("Permission denied") or text.startswith("MCP call denied")
        if denied:
            # A refusal is not a failure the model may route around: the user
            # said no, so the flow stops here and says so.
            flow.records.append(
                StepRecord(
                    index=index,
                    tool=step.tool,
                    note=step.note,
                    status=STEP_FAILED,
                    output=_keep(text, STEP_ERROR_CHARS, archive, "output")[0],
                    error="The call was denied, so it did not run.",
                )
            )
            flow.status = STATUS_FAILED
            flow.updated_at = stamp
            save(flow)
            return
        kept, reference = _keep(text, STEP_OUTPUT_CHARS, archive, "output")
        if reference:
            flow.archived.append(reference)
        failed_result = (judge or judge_result)(step.tool, text) == STEP_FAILED
        flow.records.append(
            StepRecord(
                index=index,
                tool=step.tool,
                note=step.note,
                status=STEP_FAILED if failed_result else STEP_DONE,
                output=kept,
                error="The command reported a non-zero exit code." if failed_result else "",
            )
        )
        if failed_result:
            # The same rule as an exception: the flow stops so the model can see
            # it, unless the step was declared optional.
            if not step.optional:
                flow.status = STATUS_FAILED
                flow.updated_at = stamp
                save(flow)
                return
            flow.cursor = index + 1
            flow.updated_at = stamp
            ran += 1
            save(flow)
            continue
        flow.cursor = index + 1
        flow.updated_at = stamp
        ran += 1
        save(flow)
        if step.decide:
            # A decision point on the last step is not a pause: there is nothing
            # after it to decide about, and a flow reported as paused with an
            # empty remainder is one the model will try to continue.
            flow.status = STATUS_PAUSED if flow.cursor < len(flow.steps) else STATUS_DONE
            flow.updated_at = stamp
            save(flow)
            return
    flow.status = STATUS_DONE
    flow.updated_at = stamp
    save(flow)


# ----------------------------------------------------------------- persistence


def _store_path(application_root: str) -> str:
    return os.path.join(application_root, FLOWS_FILE_NAME)


def load_flows(application_root: str, workspace: str = "") -> list[Flow]:
    """Every flow stored for this workspace, newest identifier last.

    A corrupt file is treated as no file rather than as a crash: the flows are a
    convenience for a task that is already underway, and refusing to start
    because one could not be parsed would be a worse trade.
    """
    if not application_root:
        return []
    try:
        with open(_store_path(application_root), encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict) or not isinstance(payload.get("flows"), list):
        return []
    flows: list[Flow] = []
    for item in payload["flows"]:
        flow = Flow.from_json(item)
        if flow is None:
            continue
        if workspace and flow.workspace and os.path.realpath(flow.workspace) != os.path.realpath(workspace):
            continue
        flows.append(flow)
    return flows


def active_flow(application_root: str, workspace: str = "") -> Flow | None:
    """The unfinished flow for this workspace, if there is one.

    At most one is active at a time: two half-finished plans in one workspace is
    not a feature, it is a plan the agent lost track of.
    """
    for flow in reversed(load_flows(application_root, workspace)):
        if flow.active:
            return flow
    return None


def next_flow_id(flows: Sequence[Flow]) -> str:
    highest = 0
    for flow in flows:
        if flow.id.isdigit():
            highest = max(highest, int(flow.id))
    return str(highest + 1)


def save_flow(application_root: str, flow: Flow) -> str:
    """Write the flow back, checkpoint by checkpoint.

    Atomic, because this file is written after every single step of a plan that
    may run for half an hour, and a half-written store is a lost plan.
    """
    if not application_root:
        return ""
    stored = load_flows(application_root)
    kept = [item for item in stored if item.id != flow.id]
    kept.append(flow)
    _prune(kept, flow.workspace)
    path = _store_path(application_root)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"version": FLOW_STORE_VERSION, "flows": [item.to_json() for item in kept]}, handle, indent=2)
        os.replace(temporary, path)
    except OSError:
        return ""
    return path


def abandon_flow(application_root: str, flow: Flow) -> Flow:
    flow.status = STATUS_ABANDONED
    flow.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    save_flow(application_root, flow)
    return flow


def forget_flow(application_root: str, flow: Flow) -> bool:
    """Remove a flow from the store entirely."""
    if not application_root:
        return False
    stored = load_flows(application_root)
    kept = [item for item in stored if item.id != flow.id]
    if len(kept) == len(stored):
        return False
    path = _store_path(application_root)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"version": FLOW_STORE_VERSION, "flows": [item.to_json() for item in kept]}, handle, indent=2)
    except OSError:
        return False
    return True


def _prune(flows: list[Flow], workspace: str) -> None:
    """Bound the store, and drop the outputs of the oldest finished flows.

    Twenty flows is far more than a person keeps in progress, and the cap is on
    the whole file rather than on one workspace so a long-lived store cannot
    grow without bound either way.
    """
    while len(flows) > MAX_FLOWS_PER_WORKSPACE:
        finished = next(
            (index for index, item in enumerate(flows) if item.status in (STATUS_DONE, STATUS_ABANDONED)),
            None,
        )
        position = finished if finished is not None else 0
        dropped = flows.pop(position)
        if dropped.status in (STATUS_DONE, STATUS_ABANDONED):
            for record in dropped.records:
                record.output = ""


# ------------------------------------------------------------------- reporting


def _summary(flow: Flow) -> str:
    failed = sum(1 for record in flow.records if record.status in (STEP_FAILED, STEP_BLOCKED))
    skipped = sum(1 for record in flow.records if record.status == STEP_SKIPPED)
    parts = [f"{flow.cursor}/{len(flow.steps)} steps"]
    if failed:
        parts.append(f"{failed} not done")
    if skipped:
        parts.append(f"{skipped} skipped")
    return ", ".join(parts) if len(parts) == 1 else f"{parts[0]} ({'; '.join(parts[1:])})"


def format_flow_report(flow: Flow, *, outputs: int = MAX_REPORT_OUTPUTS) -> str:
    """The state of the flow as the model reads it after a call.

    Every line is a step that ran, with what it returned where that matters,
    because a decision point is only decidable if the results are in front of
    the model rather than in a file it was not told about.
    """
    lines = [f"Flow {flow.id} · {flow.status} · {_summary(flow)}", f'Objective: {flow.objective}']
    recent = flow.records[-max(1, outputs):] if outputs else []
    shown = {record.index for record in recent}
    for index, step in enumerate(flow.steps):
        record = flow.record_for(index)
        number = f"{index + 1}."
        if record is None:
            marker = "next" if index == flow.cursor else "pending"
            note = f" · {step.note}" if step.note else ""
            flags = "".join(
                flag for flag, on in (("decide", step.decide), ("optional", step.optional)) if on
            )
            detail = f" · {flags}" if flags else ""
            guard = f" when {step.when}" if step.when else ""
            lines.append(
                f"  {number} {marker}: {step.tool}({', '.join(sorted(step.args)) or 'no arguments'})"
                f"{guard}{note}{detail}"
            )
            continue
        detail = f" · {record.note}" if record.note else ""
        guard = f" when {step.when}" if step.when else ""
        lines.append(f"  {number} {RECORD_MARK.get(record.status, '?'):>4}: {record.tool}{guard}{detail}")
        if record.status == STEP_DONE and record.output and index in shown:
            lines.append(f"        {record.output}")
        elif record.error:
            lines.append(f"        {record.error}")
    if flow.status in (STATUS_RUNNING, STATUS_PAUSED, STATUS_FAILED):
        lines.append(f"Next: {flow.describe_next()}")
    if flow.archived:
        listed = ", ".join(f'id="{reference}"' for reference in flow.archived[:MAX_REPORT_OUTPUTS])
        rest = len(flow.archived) - MAX_REPORT_OUTPUTS
        lines.append(
            f"Full step outputs kept out of the window, readable with recall_tool_output: {listed}"
            + (f", and {rest} more" if rest > 0 else "")
        )
    return "\n".join(lines)


def format_decision(flow: Flow) -> str:
    """What the model is expected to do now that the runner stopped.

    The choices are named rather than implied, because the failure mode of a
    plan that halted is the model restarting the whole thing instead of fixing
    the step that broke, which repeats every side effect that already worked.
    """
    if flow.status == STATUS_FAILED:
        problem = flow.record_for(flow.cursor - 1)
        reason = problem.error if problem is not None and problem.error else "a step did not complete"
        return (
            f"The flow stopped at step {flow.cursor + 1} ({reason}). Fix that step rather than starting "
            f"the flow again: the steps above already ran and their outputs are in the report. Call "
            f'flow_continue with action "replace_remaining" and the corrected steps, "skip_step" to pass '
            f"it, or \"abort\" to stop for good."
        )
    if flow.status == STATUS_DONE:
        return "Every step ran. Report what the flow produced; do not call flow_continue." + _recall_note(flow)
    if flow.status == STATUS_ABANDONED:
        return "This flow was abandoned. Start a new one with run_flow if there is more to do."
    return (
        "Read the results, then continue: flow_continue with action \"continue\" runs the next steps, "
        '"replace_remaining" writes them again with what you now know, "skip_step" passes the next one, '
        'or "abort" stops. Nothing above runs twice.'
    ) + _recall_note(flow)


def _recall_note(flow: Flow) -> str:
    if not flow.archived:
        return ""
    return (
        f" {len(flow.archived)} step output(s) are archived whole: read them with "
        f'recall_tool_output(id="{flow.archived[-1]}") rather than assuming the preview was all of it.'
    )


def format_flow_result(flow: Flow, *, resumed: bool = False) -> str:
    """The full tool result: what to do next, then what ran and what it said.

    The decision comes first on purpose. It is the line that has to be read
    before the model decides what to call, and a model that restarts a flow
    instead of fixing the step that failed repeats every side effect that
    already worked - so the one sentence that prevents it cannot sit below a
    table of results that invites skimming past it.
    """
    header = "Flow resumed" if resumed else "Flow started"
    return f"{format_decision(flow)}\n\n{header}: {format_flow_report(flow)}"


def format_prompt_section(flow: Flow | None) -> str:
    """The prompt line that keeps a half-finished flow visible.

    Compact on purpose. A long task is exactly the one where the window is
    under pressure, so this section names the objective, where it stopped and
    what is next, and nothing else: the detail is one ``flow_continue`` away.
    """
    if flow is None or not flow.active:
        return ""
    return (
        f"Active flow {flow.id} ({flow.status}, {_summary(flow)}): {flow.objective}\n"
        f"Next step: {flow.describe_next()}. flow_continue resumes it; run_flow starts a new one and "
        f"replaces this. The steps that already ran are not repeated."
    )


# ------------------------------------------------------------------- guidance

FLOWS_GUIDANCE = """\
A flow is for work too long for one turn: many steps, one feeding the next, surviving a compaction or a
restart. For one or two calls just make them.

Write the plan in run_flow as steps of {tool, args, note, decide, optional, when}:
- tool: a tool name exactly as the index writes it. args: that tool's named arguments.
- {{steps.N.output}} anywhere in an argument becomes what step N returned, so one step can read, edit
  and check without you seeing any of it.
- decide: true stops the flow there and shows you the results. Put it wherever the next step depends on
  something you have not seen.
- optional: true lets a step fail without stopping the flow. A run_terminal step that exits non-zero
  counts as failed, so a step you want to branch on must be optional.
- when runs the step only if an earlier one says so: steps.1.ok, !steps.2.ok, steps.1.status == done,
  steps.1.output contains "x", steps.1.output. N is an earlier step. One condition; for two, decide.
- note: one short line, shown in the report.

Control comes back at every decide, failure, blocked guard and few steps, so you can correct the plan:
flow_continue with continue, skip_step, replace_remaining or abort. Never start a flow again to fix a
failed step - the steps above already ran. An oversized step output is archived whole: read it with
recall_tool_output.

A flow runs the tools you would call yourself, under the same approval rules."""


def create_flow_tools() -> list[dict[str, Any]]:
    """The two tools a plan needs: write one, and drive it forward.

    The step object is spelled out once per tool, so every word in it is paid
    for twice in every request that carries the capability. That is why these
    descriptions are one line each and the rules live in the guidance: the
    schema says what the fields are, the guidance says what they mean, and
    neither repeats the other.
    """
    step_shape = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "description": "Tool name, as the index writes it"},
            "args": {"type": "object", "description": "Its named arguments; {{steps.N.output}} expands here"},
            "note": {"type": "string", "description": "One line: what this step is for"},
            "decide": {"type": "boolean", "description": "Stop after this step and show the results"},
            "optional": {"type": "boolean", "description": "A failure here does not stop the flow"},
            "when": {
                "type": "string",
                "description": 'Run only if an earlier step agrees, e.g. steps.1.ok or steps.1.output contains "x"',
            },
        },
        "required": ["tool"],
    }
    # flow_continue repeats the step object, and a request carries both tools, so
    # the field descriptions are paid for twice. It keeps the structure - the
    # model sends valid steps either way - and drops the prose, which the
    # guidance and run_flow's own schema already state once between them.
    step_shape_bare = {
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "args": {"type": "object"},
            "note": {"type": "string"},
            "decide": {"type": "boolean"},
            "optional": {"type": "boolean"},
            "when": {"type": "string"},
        },
        "required": ["tool"],
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "run_flow",
                "description": (
                    "Write a multi-step plan and run it until it needs a decision, fails, or runs out of "
                    "batch. For work too long for one turn, not for one or two calls. Replaces any "
                    "unfinished flow."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string", "description": "What the whole flow achieves, in one line"},
                        "steps": {"type": "array", "items": step_shape, "description": "In order. At most 40."},
                    },
                    "required": ["objective", "steps"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "flow_continue",
                "description": (
                    "Drive a paused or failed flow forward. Use replace_remaining to correct a step rather "
                    "than starting the flow again: what already ran does not run twice."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["continue", "skip_step", "replace_remaining", "abort"],
                            "description": (
                                "continue runs the next steps; skip_step passes the one in front; "
                                "replace_remaining writes them again with the corrected plan; abort stops"
                            ),
                        },
                        "steps": {
                            "type": "array",
                            "items": step_shape_bare,
                            "description": "The corrected steps, shaped as in run_flow. Only with replace_remaining.",
                        },
                    },
                    "required": ["action"],
                },
            },
        },
    ]


def format_flow_panel(flows: Sequence[Flow]) -> str:
    """The ``/flow`` listing: what exists, and where it stopped."""
    if not flows:
        return "No flow has been written for this workspace yet."
    lines = []
    for flow in flows:
        lines.append(f"Flow {flow.id} · {flow.status} · {_summary(flow)}")
        lines.append(f"  {flow.objective}")
        if flow.active:
            lines.append(f"  Next: {flow.describe_next()}")
    return "\n".join(lines)


def _step_from_json(payload: Any) -> Step | None:
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool")
    args = payload.get("args")
    if not isinstance(tool, str) or not tool:
        return None
    when = str(payload.get("when") or "")
    condition: Condition | None = None
    if when:
        try:
            condition = parse_condition(when)
        except AgentError:
            # A stored plan that was edited by hand can hold a clause nobody can
            # read. It is kept as an error rather than dropped, because dropping
            # it would run the step unguarded, which is the one outcome a guard
            # exists to prevent.
            condition = Condition(
                text=when,
                index=-1,
                field="",
                negate=False,
                operator="error",
                value="",
                error=f'The stored condition "{when}" cannot be read. {CONDITION_GRAMMAR}',
            )
    return Step(
        tool=tool,
        args=args if isinstance(args, dict) else {},
        note=str(payload.get("note") or ""),
        decide=bool(payload.get("decide")),
        optional=bool(payload.get("optional")),
        when=when,
        condition=condition,
    )
