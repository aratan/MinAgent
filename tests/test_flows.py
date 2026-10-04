"""Tests for the long plans the agent writes and runs for itself.

The runner is the part that can go quietly wrong, so these tests pin the
properties that make a flow safe to leave running: a step is checkpointed the
moment it finishes, control comes back exactly where a plan that kept going
would have been wrong, a step that consumed an output it could not have is
blocked rather than fed an empty string, and a flow step goes through the same
gated funnel a tool call does.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from minagent.app import MinAgent
from minagent.capabilities import LOAD_CAPABILITY_TOOL_NAME
from minagent.context import estimate_text_tokens
from minagent.errors import AgentError
from minagent.flows import (
    FLOWS_GUIDANCE,
    STATUS_ABANDONED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PAUSED,
    STEP_BLOCKED,
    STEP_DONE,
    STEP_FAILED,
    STEP_SKIPPED,
    Flow,
    FlowBlocked,
    StepRecord,
    active_flow,
    advance_flow,
    create_flow_tools,
    evaluate_condition,
    format_decision,
    format_flow_report,
    format_prompt_section,
    load_flows,
    make_flow,
    parse_condition,
    parse_steps,
    resolve_arguments,
    save_flow,
)
from minagent.jsutil import json_stringify
from minagent.tool_archive import ToolArchive
from minagent.workspace import WorkspaceAccess

TOOLS = {"read_file", "write_file", "run_terminal", "edit_file", "list_directory", "create_directory"}


class _FakeOutput:
    """A stdout stand-in that records what was written."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, value: str) -> None:
        self.chunks.append(value)

    def isatty(self) -> bool:
        return False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def _app(tmp_path, **settings: Any) -> MinAgent:
    app = MinAgent(stdout=_FakeOutput())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "Test"
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    for name, value in settings.items():
        setattr(app, name, value)
    app.ensure_flow_tools()
    app.rebuild_capabilities()
    return app


def _flow(tmp_path, *specs: tuple[str, dict[str, Any], dict[str, Any]], **kwargs: Any) -> Flow:
    """A flow built from (tool, args, flags) triples, already checkpointed."""
    flow = make_flow(
        flow_id=kwargs.get("flow_id", "1"),
        objective=kwargs.get("objective", "Test objective"),
        steps=parse_steps(
            [
                {"tool": tool, "args": args, **flags}
                for tool, args, flags in specs
            ],
            known_tools=TOOLS,
        ),
        workspace=str(tmp_path),
    )
    save_flow(str(tmp_path), flow)
    return flow


def _archive_recorder() -> tuple[Any, dict[str, str]]:
    """An archive stand-in that keeps what it was given, to read back later."""
    store: dict[str, str] = {}
    counter = [0]

    def archive(text: str) -> str | None:
        counter[0] += 1
        reference = f"arch{counter[0]}"
        store[reference] = text
        return reference

    return archive, store


def _recorder(results: dict[str, Any], calls: list[tuple[str, dict[str, Any]]]):
    """An executor that answers from a table and records what it was asked."""

    async def execute(tool: str, arguments: dict[str, Any]) -> Any:
        calls.append((tool, arguments))
        answer = results.get(tool, "ok")
        if isinstance(answer, Exception):
            raise answer
        return answer

    return execute


async def _run(flow: Flow, tmp_path, execute, **kwargs: Any) -> None:
    """Drive a flow to its next stop with the disk checkpoint in place."""
    await advance_flow(flow, execute, save=lambda item: save_flow(str(tmp_path), item), **kwargs)


# ---------------------------------------------------------------- validation


def test_a_plan_is_read_step_by_step_with_its_flags():
    steps = parse_steps(
        [
            {"tool": "read_file", "args": {"path": "a.txt"}, "note": "look"},
            {"tool": "write_file", "args": {"path": "b.txt"}, "decide": True, "optional": True},
        ],
        known_tools=TOOLS,
    )
    assert [(step.tool, step.note) for step in steps] == [("read_file", "look"), ("write_file", "")]
    assert (steps[0].decide, steps[0].optional) == (False, False)
    assert (steps[1].decide, steps[1].optional) == (True, True)


def test_a_step_written_as_text_is_corrected_with_an_example():
    """A small model writes "1. read_file(path)" often enough to be worth naming."""
    with pytest.raises(AgentError, match=r'Step 1 is text, not an object.*"tool": "read_file"'):
        parse_steps(["read_file(path)"], known_tools=TOOLS)


def test_a_tool_that_does_not_exist_is_refused_before_the_plan_runs():
    with pytest.raises(AgentError, match="Step 2 calls teleport, which is not a tool"):
        parse_steps(
            [{"tool": "read_file"}, {"tool": "teleport"}],
            known_tools=TOOLS,
        )


def test_a_flow_cannot_start_another_flow():
    """No defined end and no defined checkpoint: the runner is not a step tool."""
    with pytest.raises(AgentError, match="which is the flow runner itself"):
        parse_steps([{"tool": "run_flow"}], known_tools=TOOLS | {"run_flow"})


def test_a_plan_is_bounded():
    with pytest.raises(AgentError, match="at most 40 steps"):
        parse_steps([{"tool": "read_file"}] * 41, known_tools=TOOLS)


def test_arguments_are_checked_before_anything_is_executed_from_them():
    with pytest.raises(AgentError, match="args must be an object"):
        parse_steps([{"tool": "write_file", "args": ["a.txt"]}], known_tools=TOOLS)
    with pytest.raises(AgentError, match="must be text, a number, a list, or an object"):
        parse_steps([{"tool": "write_file", "args": {"path": object()}}], known_tools=TOOLS)
    with pytest.raises(AgentError, match="limited to 40000"):
        parse_steps([{"tool": "write_file", "args": {"content": "x" * 40001}}], known_tools=TOOLS)


def test_an_empty_plan_is_refused_rather_than_treated_as_finished():
    with pytest.raises(AgentError, match="non-empty list"):
        parse_steps([], known_tools=TOOLS)


# --------------------------------------------------------------- substitution


def test_a_step_consumes_the_output_of_the_step_before_it():
    flow = make_flow(
        flow_id="1",
        objective="copy",
        steps=parse_steps(
            [
                {"tool": "read_file", "args": {"path": "a.txt"}},
                {"tool": "write_file", "args": {"path": "b.txt", "content": "{{steps.0.output}}"}},
            ],
            known_tools=TOOLS,
        ),
        workspace=".",
    )
    flow.records.append(StepRecord(index=0, tool="read_file", status=STEP_DONE, output="hello"))
    flow.cursor = 1
    assert resolve_arguments(flow.steps[1].args, flow) == {"path": "b.txt", "content": "hello"}


async def test_substitution_reaches_inside_a_nested_argument():
    flow = make_flow(
        flow_id="1",
        objective="nested",
        steps=parse_steps(
            [{"tool": "run_terminal", "args": {"command": "pytest {{steps.0.output}}"}}],
            known_tools=TOOLS,
        ),
        workspace=".",
    )
    flow.records.append(StepRecord(index=0, tool="read_file", status=STEP_DONE, output="ok"))
    flow.cursor = 1
    assert resolve_arguments({"a": ["{{steps.0.status}}"]}, flow) == {"a": ["done"]}
    assert resolve_arguments({"a": ["{{steps.0.output}}"]}, flow) == {"a": ["ok"]}


def test_an_output_that_does_not_exist_blocks_the_step_instead_of_filling_it_with_nothing():
    """An empty string here fails twenty steps later, somewhere else entirely."""
    flow = make_flow(
        flow_id="1",
        objective="guess",
        steps=parse_steps(
            [{"tool": "write_file", "args": {"content": "{{steps.4.output}}"}}], known_tools=TOOLS
        ),
        workspace=".",
    )
    with pytest.raises(FlowBlocked, match="step 5 has not run"):
        resolve_arguments(flow.steps[0].args, flow)


# ------------------------------------------------------------------- running


async def test_steps_run_in_order_until_the_plan_is_done(tmp_path):
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(
        tmp_path,
        ("read_file", {"path": "a"}, {}),
        ("write_file", {"path": "b"}, {}),
        ("run_terminal", {"command": "pytest"}, {}),
    )
    await _run(flow, tmp_path, _recorder({"read_file": "one", "write_file": "two"}, calls))
    assert [call[0] for call in calls] == ["read_file", "write_file", "run_terminal"]
    assert flow.status == STATUS_DONE
    assert flow.cursor == 3


async def test_a_decision_point_hands_control_back_with_the_results_in_hand(tmp_path):
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(
        tmp_path,
        ("read_file", {"path": "a"}, {"decide": True}),
        ("edit_file", {"path": "b"}, {}),
    )
    await _run(flow, tmp_path, _recorder({"read_file": "the contents"}, calls))
    assert [call[0] for call in calls] == ["read_file"], "the step after decide must not run"
    assert flow.status == STATUS_PAUSED
    report = format_flow_report(flow)
    assert "the contents" in report, "a decision is only decidable if the result is in front of it"
    assert "Next: 2." in report
    assert '"replace_remaining"' in format_decision(flow)
    assert "Nothing above runs twice" in format_decision(flow)


async def test_a_decision_point_on_the_last_step_finishes_rather_than_pausing(tmp_path):
    """A paused flow with an empty remainder is one the model tries to continue."""
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {"decide": True}))
    await _run(flow, tmp_path, _recorder({"read_file": "one"}, []))
    assert flow.status == STATUS_DONE
    assert "Every step ran" in format_decision(flow)


async def test_every_step_is_checkpointed_as_it_finishes(tmp_path):
    """The checkpoint is what makes a plan survive a crash rather than merely resume."""
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {"decide": True}), ("read_file", {"path": "b"}, {}))
    seen: list[int] = []
    await _run(
        flow,
        tmp_path,
        _recorder({"read_file": "one"}, calls),
        on_step=lambda index, step, arguments: seen.append(index),
        is_cancelled=lambda: len(seen) > 0,
    )
    stored = load_flows(str(tmp_path))[0]
    assert stored.cursor == 1
    assert stored.records[0].status == STEP_DONE
    assert stored.status == STATUS_PAUSED, "a cancelled flow is paused, never lost"


async def test_the_batch_limit_is_how_often_the_runner_answers_to_somebody_else(tmp_path):
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(tmp_path, *(("read_file", {"path": str(index)}, {}) for index in range(6)))
    await _run(flow, tmp_path, _recorder({}, calls), batch_limit=2)
    assert len(calls) == 2
    assert flow.status == STATUS_PAUSED
    assert flow.cursor == 2, "the steps that ran are not run again"


async def test_a_failed_step_stops_the_flow_and_names_itself(tmp_path):
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {}), ("write_file", {"path": "b"}, {}))
    await _run(flow, tmp_path, _recorder({"write_file": AgentError("no such file: b")}, calls))
    assert [call[0] for call in calls] == ["read_file", "write_file"]
    assert flow.status == STATUS_FAILED
    assert "no such file: b" in flow.records[1].error
    decision = format_decision(flow)
    assert "Fix that step rather than starting the flow again" in decision
    assert "already ran" in decision


async def test_an_optional_step_may_fail_without_stopping_the_plan(tmp_path):
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(
        tmp_path,
        ("read_file", {"path": "missing"}, {"optional": True}),
        ("write_file", {"path": "b"}, {}),
    )
    await _run(flow, tmp_path, _recorder({"read_file": AgentError("no such file"), "write_file": "ok"}, calls))
    assert [call[0] for call in calls] == ["read_file", "write_file"]
    assert flow.status == STATUS_DONE
    assert flow.records[0].status == STEP_FAILED
    assert flow.cursor == 2


async def test_a_denied_call_stops_the_flow_rather_than_being_routed_around(tmp_path):
    """The user said no. The runner is not allowed to try the next step anyway."""
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(tmp_path, ("run_terminal", {"command": "rm -rf /"}, {}), ("run_terminal", {"command": "x"}, {}))
    await _run(flow, tmp_path, _recorder({"run_terminal": "Permission denied by the user"}, calls))
    assert len(calls) == 1
    assert flow.status == STATUS_FAILED
    assert "denied" in flow.records[0].error


async def test_a_step_blocked_on_a_missing_output_pauses_rather_than_fails(tmp_path):
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(tmp_path, ("write_file", {"path": "b", "content": "{{steps.9.output}}"}, {}))
    await _run(flow, tmp_path, _recorder({}, calls))
    assert calls == []
    assert flow.status == STATUS_PAUSED
    assert flow.records[0].status == STEP_BLOCKED
    assert "decide" in flow.records[0].error


async def test_an_unexpected_exception_fails_the_step_instead_of_the_session(tmp_path):
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {}))
    await _run(flow, tmp_path, _recorder({"read_file": RuntimeError("boom")}, []))
    assert flow.status == STATUS_FAILED
    assert "boom" in flow.records[0].error


# ---------------------------------------------------------------- persistence


async def test_a_flow_survives_a_restart(tmp_path):
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {"decide": True}), ("write_file", {"path": "b"}, {}))
    await _run(flow, tmp_path, _recorder({"read_file": "one"}, calls))
    # A new session reads the file rather than holding a reference to the object.
    restored = active_flow(str(tmp_path), str(tmp_path))
    assert restored is not None
    assert restored.id == flow.id
    assert restored.cursor == 1
    assert restored.records[0].output == "one"
    assert restored.steps[1].tool == "write_file"


def test_a_flow_of_another_workspace_is_not_picked_up(tmp_path):
    other = tmp_path / "otro"
    other.mkdir()
    _flow(tmp_path, ("read_file", {"path": "a"}, {"decide": True}))
    assert active_flow(str(tmp_path), str(other)) is None


def test_a_corrupt_store_is_ignored_rather_than_refusing_to_start(tmp_path):
    """The flows are a convenience for work already underway, not a dependency."""

    path = tmp_path / ".minagent" / "flows.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_flows(str(tmp_path)) == []
    assert active_flow(str(tmp_path)) is None


def test_only_one_flow_is_active_per_workspace(tmp_path):
    first = _flow(tmp_path, ("read_file", {"path": "a"}, {"decide": True}))
    second = _flow(
        tmp_path,
        ("read_file", {"path": "b"}, {"decide": True}),
        flow_id="2",
    )
    assert [item.id for item in load_flows(str(tmp_path))] == ["1", "2"]
    assert active_flow(str(tmp_path)) is not None
    assert active_flow(str(tmp_path)).id == "2"
    assert first.id == "1" and second.id == "2"


async def test_a_finished_flow_is_not_offered_again(tmp_path):
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {}))
    await _run(flow, tmp_path, _recorder({}, []))
    assert flow.status == STATUS_DONE
    assert active_flow(str(tmp_path)) is None


async def test_the_prompt_line_names_the_objective_and_the_next_step(tmp_path):
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {"decide": True}), ("write_file", {"path": "b"}, {}))
    section = format_prompt_section(flow)
    assert "Test objective" in section
    assert "Next step: 1. read_file(path)" in section
    await _run(flow, tmp_path, _recorder({"read_file": "one"}, []))
    paused = format_prompt_section(flow)
    assert "paused" in paused
    assert "Next step: 2. write_file" in paused
    assert format_prompt_section(None) == ""


# ------------------------------------------------------------- the app session


async def test_the_model_writes_a_plan_and_it_runs(tmp_path):
    (tmp_path / "origen.txt").write_text("contenido\n", encoding="utf-8")
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=8)
    result = await app.execute_tool(
        "run_flow",
        {
            "objective": "Copy the file and report it",
            "steps": [
                {"tool": "read_file", "args": {"path": "origen.txt"}, "note": "leer"},
                {"tool": "write_file", "args": {"path": "salida.txt", "content": "{{steps.0.output}}"}},
            ],
        },
    )
    assert (tmp_path / "salida.txt").read_text(encoding="utf-8").strip() == "contenido"
    assert app.flow is not None and app.flow.status == STATUS_DONE
    assert "Flow started" in result


async def test_a_flow_step_loads_the_capability_it_needs(tmp_path):
    """A step naming a tool of an unloaded capability is ordinary, not an error."""
    (tmp_path / "a.txt").write_text("hola\n", encoding="utf-8")
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=8)
    assert "read_file" not in [tool["function"]["name"] for tool in app.tools]
    await app.execute_tool(
        "run_flow",
        {"objective": "read it", "steps": [{"tool": "read_file", "args": {"path": "a.txt"}}]},
    )
    assert app.capabilities is not None and "files.read" in app.capabilities.loaded


async def test_a_failing_step_hands_the_model_the_choice_to_correct_it(tmp_path):
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=8)
    report = await app.execute_tool(
        "run_flow",
        {
            "objective": "read a file that is not there",
            "steps": [{"tool": "read_file", "args": {"path": "nada.txt"}}],
        },
    )
    assert app.flow is not None and app.flow.status == STATUS_FAILED
    assert "replace_remaining" in report
    (tmp_path / "notas.txt").write_text("hola\n", encoding="utf-8")
    fixed = await app.execute_tool(
        "flow_continue",
        {
            "action": "replace_remaining",
            "steps": [{"tool": "read_file", "args": {"path": "notas.txt"}}],
        },
    )
    assert "hola" in fixed
    assert "Flow resumed" in fixed
    assert app.flow is not None and app.flow.status == STATUS_DONE
    # The corrected step replaced the broken one rather than being judged by it.
    assert [record.status for record in app.flow.records] == [STEP_DONE]


async def test_correcting_a_flow_does_not_run_what_already_ran(tmp_path):
    (tmp_path / "origen.txt").write_text("contenido\n", encoding="utf-8")
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=1)
    await app.execute_tool(
        "run_flow",
        {
            "objective": "two steps",
            "steps": [{"tool": "read_file", "args": {"path": "origen.txt"}}, {"tool": "list_directory"}],
        },
    )
    assert app.flow is not None and app.flow.status == STATUS_PAUSED
    assert app.flow.cursor == 1
    # A model that restarts the flow would read the file again; replacing the
    # remainder does not, which is the point of keeping the record.
    await app.execute_tool(
        "flow_continue",
        {"action": "replace_remaining", "steps": [{"tool": "create_directory", "args": {"path": "salida"}}]},
    )
    assert app.flow is not None
    assert [record.tool for record in app.flow.records] == ["read_file", "create_directory"]


async def test_resuming_a_flow_that_does_not_exist_is_refused(tmp_path):
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=8)
    with pytest.raises(AgentError, match="no unfinished flow"):
        await app.execute_tool("flow_continue", {"action": "continue"})


async def test_a_new_flow_replaces_an_unfinished_one_and_says_so(tmp_path):
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=1)
    await app.execute_tool(
        "run_flow",
        {"objective": "first", "steps": [{"tool": "list_directory"}, {"tool": "list_directory"}]},
    )
    report = await app.execute_tool("run_flow", {"objective": "second", "steps": [{"tool": "list_directory"}]})
    assert "was unfinished and has been abandoned" in report
    stored = load_flows(str(tmp_path))
    assert [item.status for item in stored] == [STATUS_ABANDONED, STATUS_DONE]


async def test_an_unknown_action_is_refused_with_the_real_ones(tmp_path):
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=8)
    await app.execute_tool(
        "run_flow",
        {"objective": "one", "steps": [{"tool": "list_directory"}, {"tool": "list_directory", "decide": True}]},
    )
    with pytest.raises(AgentError, match='action must be "continue"'):
        await app.execute_tool("flow_continue", {"action": "restart"})
    # Wrong on a finished flow too, so it is refused there as well.
    await app.execute_tool("flow_continue", {"action": "continue"})
    with pytest.raises(AgentError, match='action must be "continue"'):
        await app.execute_tool("flow_continue", {"action": "restart"})


async def test_an_unfinished_flow_is_visible_to_the_model_in_the_prompt(tmp_path):
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=1)
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.refresh_system_prompt()
    await app.execute_tool(
        "run_flow",
        {
            "objective": "Build the report",
            "steps": [{"tool": "list_directory"}, {"tool": "list_directory", "decide": True}],
        },
    )
    app.refresh_system_prompt()
    assert "Active flow" in app.messages[0]["content"]
    assert "Build the report" in app.messages[0]["content"]


async def test_the_flow_tools_belong_to_a_capability(tmp_path):
    """The invariant every registered tool must keep, flows included."""
    app = _app(tmp_path, flows_enabled=True)
    assert app.capabilities is not None
    claimed = {name for entry in app.capabilities.entries for name in entry.tool_names}
    assert {"run_flow", "flow_continue"} <= claimed
    assert "flows" in [entry.name for entry in app.capabilities.entries]
    assert LOAD_CAPABILITY_TOOL_NAME not in claimed
    # Loaded on demand like everything else: the index line names them.
    assert "run_flow" in cast(dict[str, str], app.capability_index_section())["content"]


async def test_the_flow_command_lists_and_drives_a_plan_without_the_model(tmp_path):
    (tmp_path / "a.txt").write_text("hola\n", encoding="utf-8")
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=1)
    await app.execute_tool(
        "run_flow",
        {
            "objective": "read two files",
            "steps": [{"tool": "read_file", "args": {"path": "a.txt"}}, {"tool": "list_directory"}],
        },
    )
    assert app.flow is not None and app.flow.cursor == 1
    app.print("")
    await app.handle_flow_command("1 continue")
    assert app.flow is not None and app.flow.status == STATUS_DONE
    assert "Flow 1 · done" in app._stdout.text


async def test_the_flow_command_can_abandon_and_remove_a_plan(tmp_path):
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=1)
    await app.execute_tool(
        "run_flow",
        {"objective": "one", "steps": [{"tool": "list_directory"}, {"tool": "list_directory", "decide": True}]},
    )
    assert app.flow is not None and app.flow.active
    await app.handle_flow_command("abort")
    assert app.flow is None
    assert [item.status for item in load_flows(str(tmp_path))] == [STATUS_ABANDONED]
    await app.handle_flow_command("1 forget")
    assert load_flows(str(tmp_path)) == []


async def test_the_flow_command_says_so_when_flows_are_off(tmp_path):
    app = _app(tmp_path, flows_enabled=False)
    with pytest.raises(AgentError, match="FLOWS_ENABLED=on"):
        await app.handle_flow_command("")


def test_a_flow_step_is_counted_as_work_the_turn_did(tmp_path):
    """The evidence guard reads this list, so a planned write must land in it."""
    flow = make_flow(
        flow_id="1",
        objective="write",
        steps=parse_steps([{"tool": "write_file", "args": {"path": "b.txt", "content": "x"}}], known_tools=TOOLS),
        workspace=".",
    )
    assert flow.steps[0].tool == "write_file"
    assert isinstance(flow.records, list)
    assert STEP_SKIPPED not in {record.status for record in flow.records}


def test_the_step_record_survives_the_json_the_store_writes(tmp_path):
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {"decide": True, "note": "leer"}))
    restored = Flow.from_json(flow.to_json())
    assert restored is not None
    assert restored.steps[0].note == "leer"
    assert restored.steps[0].decide is True
    assert restored.objective == "Test objective"


# ----------------------------------------------------------------- conditions


def test_a_condition_is_read_into_something_answerable():
    """The whole grammar, because a guard nobody can parse must not run a step."""
    assert parse_condition("always").operator == "always"
    assert parse_condition("steps.1.ok").field == "ok"
    assert parse_condition("!steps.2.ok").negate is True
    assert parse_condition("steps.3.status == done").operator == "=="
    assert parse_condition("steps.3.status").value == "done", "a bare status means it worked"
    assert parse_condition('steps.1.output contains "error"').operator == "contains"
    assert parse_condition('steps.1.output not contains "error"').operator == "not contains"
    assert parse_condition("steps.1.output").operator == "truthy"
    assert parse_condition('steps.1.output contains "x"').value == "x", "quotes are not part of the text"


def test_an_unreadable_condition_is_answered_with_the_grammar():
    with pytest.raises(AgentError, match="A condition is steps.N.ok"):
        parse_condition("steps.two.ok")
    with pytest.raises(AgentError, match="is true or false, not 'maybe'"):
        parse_condition("steps.1.ok == maybe")


def test_a_boolean_field_cannot_be_compared_as_text():
    with pytest.raises(AgentError, match="cannot be compared with"):
        parse_condition('steps.1.ok contains "yes"')


def test_a_guard_reads_a_step_before_it():
    with pytest.raises(AgentError, match="which is itself"):
        parse_steps([{"tool": "read_file"}, {"tool": "write_file", "when": "steps.2.ok"}], known_tools=TOOLS)
    with pytest.raises(AgentError, match="comes later and has not run yet"):
        parse_steps([{"tool": "read_file", "when": "steps.4.ok"}, {"tool": "write_file"}], known_tools=TOOLS)


async def test_a_step_runs_only_when_its_guard_holds(tmp_path):
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(
        tmp_path,
        ("read_file", {"path": "a"}, {"optional": True}),
        ("write_file", {"path": "b"}, {"when": 'steps.1.output contains "OK"'}),
        ("run_terminal", {"command": "x"}, {"when": "steps.1.output"}),
        ("edit_file", {"path": "c"}, {}),
    )
    async def execute(tool: str, arguments: dict[str, Any]) -> Any:
        calls.append((tool, arguments))
        return ""

    await _run(flow, tmp_path, execute)
    assert [call[0] for call in calls] == ["read_file", "edit_file"], "an empty output fails both guards"
    assert flow.records[1].status == STEP_SKIPPED
    assert 'steps.1.output contains "OK" was not true' in flow.records[1].error
    assert flow.records[2].status == STEP_SKIPPED
    assert flow.status == STATUS_DONE
    report = format_flow_report(flow)
    assert 'when steps.1.output contains "OK"' in report
    assert "when steps.1.output" in report


async def test_a_guard_that_only_a_failure_can_satisfy_is_refused_before_the_flow_runs():
    """The one composition the model writes and cannot get right on its own.

    A failed step stops the flow, so a branch guarded on that failure is the
    single step the flow could never reach. Caught while the plan is written,
    where the fix is one word; caught later it is a finished plan and a
    puzzled model.
    """
    with pytest.raises(AgentError, match='Mark step 1 "optional": true'):
        parse_steps(
            [
                {"tool": "run_terminal", "args": {"command": "pytest"}},
                {"tool": "write_file", "args": {"path": "fix.md"}, "when": "!steps.1.ok"},
            ],
            known_tools=TOOLS,
        )
    with pytest.raises(AgentError, match="reads a failure of step 1"):
        parse_steps(
            [
                {"tool": "run_terminal", "args": {"command": "pytest"}},
                {"tool": "write_file", "args": {"path": "fix.md"}, "when": "steps.1.status == failed"},
            ],
            known_tools=TOOLS,
        )
    # The same guard is fine once the step it reads may fail without stopping.
    allowed = parse_steps(
        [
            {"tool": "run_terminal", "args": {"command": "pytest"}, "optional": True},
            {"tool": "write_file", "args": {"path": "fix.md"}, "when": "!steps.1.ok"},
        ],
        known_tools=TOOLS,
    )
    assert allowed[1].condition is not None


async def test_a_guard_reads_a_failure_it_was_written_against(tmp_path):
    """The common case: the test fails, so the fix runs. Needs optional on the
    step it branches on, because a plain failure stops the flow by design."""
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(
        tmp_path,
        ("run_terminal", {"command": "pytest"}, {"optional": True}),
        ("write_file", {"path": "fix.md"}, {"when": "!steps.1.ok"}),
        ("write_file", {"path": "ok.md"}, {"when": "steps.1.ok"}),
    )
    results = {"run_terminal": AgentError("3 failed"), "write_file": "written"}
    await _run(flow, tmp_path, _recorder(results, calls))
    assert [call[0] for call in calls] == ["run_terminal", "write_file"]
    assert calls[1][1]["path"] == "fix.md", "the fix runs because the tests failed"
    assert flow.records[2].status == STEP_SKIPPED, "and the success branch does not"
    assert flow.status == STATUS_DONE


async def test_a_failure_the_model_did_not_plan_for_still_stops_the_flow(tmp_path):
    """No guard was written for it, so there is nothing to reach."""
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(
        tmp_path,
        ("run_terminal", {"command": "pytest"}, {}),
        ("read_file", {"path": "b"}, {}),
    )
    await _run(flow, tmp_path, _recorder({"run_terminal": AgentError("3 failed")}, calls))
    assert [call[0] for call in calls] == ["run_terminal"]
    assert flow.status == STATUS_FAILED
    assert "replace_remaining" in format_decision(flow)


async def test_a_step_a_guard_skipped_does_not_spend_the_batch(tmp_path):
    """A plan made mostly of conditions should reach the end, not stop early."""
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(
        tmp_path,
        ("read_file", {"path": "a"}, {"optional": True}),
        *(
            ("read_file", {"path": f"f{index}"}, {"when": "steps.1.ok"})
            for index in range(6)
        ),
        ("read_file", {"path": "last"}, {}),
    )
    async def execute(tool: str, arguments: dict[str, Any]) -> Any:
        calls.append((tool, arguments))
        if arguments["path"] == "a":
            raise AgentError("nada")
        return "uno"

    await _run(flow, tmp_path, execute, batch_limit=2)
    assert [call[1]["path"] for call in calls] == ["a", "last"]
    assert flow.cursor == 8, "the six skipped steps passed without stopping the flow"
    assert flow.status == STATUS_DONE, "only the two real steps spent a batch slot"
    assert [record.status for record in flow.records[1:7]] == [STEP_SKIPPED] * 6


def test_a_guard_on_a_step_that_has_not_run_is_refused_rather_than_guessed():
    flow = make_flow(
        flow_id="1",
        objective="early",
        steps=parse_steps([{"tool": "read_file", "args": {"path": "a"}}], known_tools=TOOLS),
        workspace=".",
    )
    with pytest.raises(FlowBlocked, match="step 3 has not run"):
        evaluate_condition(parse_condition("steps.3.ok"), flow)


def test_a_condition_from_a_hand_edited_store_pauses_instead_of_running_unguarded(tmp_path):
    """Dropping an unreadable clause would run the step with no guard at all."""
    stored = {
        "id": "1",
        "objective": "edited by hand",
        "status": "paused",
        "cursor": 0,
        "steps": [{"tool": "delete_file", "args": {"path": "a"}, "when": "steps.two.ok"}],
        "records": [],
    }
    save_flow(str(tmp_path), Flow.from_json(stored))  # type: ignore[arg-type]
    restored = active_flow(str(tmp_path))
    assert restored is not None
    with pytest.raises(FlowBlocked, match="cannot be read"):
        evaluate_condition(restored.steps[0].condition, restored)


def test_a_condition_survives_the_store(tmp_path):
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {}), ("write_file", {"path": "b"}, {"when": "steps.1.ok"}))
    restored = Flow.from_json(flow.to_json())
    assert restored is not None
    assert restored.steps[1].when == "steps.1.ok"
    assert restored.steps[1].condition is not None
    assert restored.steps[1].condition.index == 0


# --------------------------------------------------- outputs kept whole


async def test_an_oversized_step_output_is_archived_and_readable_whole(tmp_path):
    """A record that only keeps a preview is a result nobody can go back to."""
    calls: list[tuple[str, dict[str, Any]]] = []
    archive, store = _archive_recorder()
    payload = "".join(f"line {index} of the output\n" for index in range(1000))
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {"decide": True}))
    await _run(flow, tmp_path, _recorder({"read_file": payload}, calls), archive=archive)
    record = flow.records[0]
    assert len(record.output) < len(payload)
    assert 'id="arch1"' in record.output, "the preview has to name the reference"
    assert store["arch1"] == payload.strip(), "what was archived is the whole result"
    assert flow.archived == ["arch1"]
    assert 'id="arch1"' in format_flow_report(flow)
    assert "recall_tool_output" in format_decision(flow)


async def test_a_short_output_is_left_alone(tmp_path):
    archive, store = _archive_recorder()
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {}))
    await _run(flow, tmp_path, _recorder({"read_file": "corta"}, []), archive=archive)
    assert flow.records[0].output == "corta"
    assert flow.archived == []
    assert store == {}


async def test_text_that_could_not_be_archived_says_so_rather_than_pretending(tmp_path):
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {}))
    payload = "x" * 9000
    await _run(flow, tmp_path, _recorder({"read_file": payload}, []), archive=lambda _text: None)
    assert "could not be archived" in flow.records[0].output
    assert "lost" in flow.records[0].output
    assert flow.archived == []


async def test_the_archived_output_is_recoverable_through_the_real_archive(tmp_path):
    """The whole chain: a real terminal step, a real archive, recall_tool_output."""
    app = _app(tmp_path, flows_enabled=True, terminal_mode="auto", flow_batch_steps=4)
    app.terminal_command_shell = "/bin/sh"
    app.terminal_timeout_seconds = 60
    app.tool_archive = ToolArchive(str(tmp_path))
    app.register_tool_schemas([__import__("minagent.tools", fromlist=["x"]).build_terminal_tool()])
    result = await app.execute_tool(
        "run_flow",
        {
            "objective": "print a lot",
            "steps": [{"tool": "run_terminal", "args": {"command": "seq 1 9000"}, "decide": True}],
        },
    )
    assert app.flow is not None and app.flow.archived
    reference = app.flow.archived[0]
    assert f'id="{reference}"' in result
    recalled = app.tool_archive.read(reference, limit=40000)
    assert "characters 0-40000" in recalled
    tail = app.tool_archive.read(reference, offset=40000, limit=40000)
    assert "characters 40000-" in tail
    assert "9000" in tail, "the tail is inside the archive, not just the preview"


async def test_the_archived_references_survive_a_restart(tmp_path):
    archive, _store = _archive_recorder()
    flow = _flow(tmp_path, ("read_file", {"path": "a"}, {}))
    await _run(flow, tmp_path, _recorder({"read_file": "y" * 9000}, []), archive=archive)
    restored = load_flows(str(tmp_path))[0]
    assert restored.archived == ["arch1"], "a finished flow still owes the model its references"


# ------------------------------------------------ what counts as a failure


async def test_a_command_that_exits_non_zero_is_a_failed_step(tmp_path):
    """pytest exits 1 without raising, and "ok" has to mean what the model meant."""
    calls: list[tuple[str, dict[str, Any]]] = []
    flow = _flow(
        tmp_path,
        ("run_terminal", {"command": "pytest"}, {"optional": True}),
        ("run_terminal", {"command": "pytest -x tests"}, {"when": "!steps.1.ok"}),
    )
    async def execute(tool: str, arguments: dict[str, Any]) -> Any:
        calls.append((tool, arguments))
        return "Exit code: 1\n3 failed" if len(calls) == 1 else "Exit code: 0\nall green"

    await _run(flow, tmp_path, execute)
    assert [call[1]["command"] for call in calls] == ["pytest", "pytest -x tests"]
    assert flow.records[0].status == STEP_FAILED
    assert "non-zero exit code" in flow.records[0].error
    assert flow.status == STATUS_DONE


async def test_a_non_zero_exit_stops_the_flow_unless_the_step_is_optional(tmp_path):
    flow = _flow(tmp_path, ("run_terminal", {"command": "grep q file"}, {}), ("list_directory", {}, {}))
    await _run(flow, tmp_path, _recorder({"run_terminal": "Exit code: 2\nno such file"}, []))
    assert flow.status == STATUS_FAILED
    assert [record.tool for record in flow.records] == ["run_terminal"], "the next step never ran"


async def test_a_successful_command_is_untouched_by_that_rule(tmp_path):
    flow = _flow(tmp_path, ("run_terminal", {"command": "ls"}, {}))
    await _run(flow, tmp_path, _recorder({"run_terminal": "Exit code: 0\nREADME.md"}, []))
    assert flow.records[0].status == STEP_DONE
    assert flow.records[0].error == ""


# ------------------------------------------------ where the correction goes


async def test_the_correction_comes_before_the_results(tmp_path):
    """The sentence that stops a restart cannot sit under a table of output.

    A model that restarts a flow repeats every side effect that already worked,
    so the line that prevents it is read before anything else in the result.
    """
    app = _app(tmp_path, flows_enabled=True, flow_batch_steps=8)
    report = await app.execute_tool(
        "run_flow",
        {"objective": "read a file that is not there", "steps": [{"tool": "read_file", "args": {"path": "nada.txt"}}]},
    )
    decision_at = report.index("Fix that step rather than starting the flow again")
    report_at = report.index("Flow started:")
    assert decision_at < report_at
    assert "the steps above already ran" in report, "and what to do is named, not implied"


# ------------------------------------------------------ what it costs to load


def test_the_flow_capability_has_a_context_budget():
    """It was the most expensive thing in the system and nothing said so.

    Guidance and both schemas go out with every request that carries the
    capability, so a phrase written twice is paid for twice. Measured, not
    guessed: the budget is what it costs now, and going over it means the prose
    belongs in the guidance once rather than in both schemas.
    """
    cost = estimate_text_tokens(FLOWS_GUIDANCE) + estimate_text_tokens(json_stringify(create_flow_tools()))
    assert cost <= 1200, f"the flows capability costs {cost} tokens to load, over the 1200 budget"
    assert estimate_text_tokens(FLOWS_GUIDANCE) <= 550, "the guidance has to keep the rules and lose the prose"


def test_the_step_fields_are_described_once_and_not_twice():
    """Both tools take steps, so both would otherwise carry the same prose."""
    run_flow, flow_continue = create_flow_tools()
    assert "as the index writes it" in json_stringify(run_flow)
    replacement = flow_continue["function"]["parameters"]["properties"]["steps"]["items"]
    assert all("description" not in field for field in replacement["properties"].values())
    assert set(replacement["properties"]) == set(run_flow["function"]["parameters"]["properties"]["steps"]["items"]["properties"])


def test_the_budget_leaves_room_for_the_window_it_runs_on(tmp_path):
    """The comparison that matters: a loaded capability against the fixed prompt."""
    app = _app(tmp_path, flows_enabled=True, terminal_mode="auto")
    app.load_capabilities(["flows"])
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.refresh_system_prompt()
    index = app.capability_index_section()
    assert index is not None
    assert estimate_text_tokens(index["content"]) < estimate_text_tokens(json_stringify(app.tools))
