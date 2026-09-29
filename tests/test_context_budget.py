"""Tests for the context governor: what it gives up, in what order, and when.

The point of the cascade is that the window is finite and the agent still has
to be able to work, so these tests check both halves of that: that pressure
actually frees tokens, and that what it frees is always recoverable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from minagent.app import MinAgent
from minagent.context_budget import (
    SHED_IDLE_CAPABILITIES,
    SHED_INDEX_SUMMARIES,
    SHED_MEMORY_HINTS,
    SHED_OLD_TOOL_RESULTS,
    SHED_STEPS,
    ContextPolicy,
)
from minagent.tool_archive import ToolArchive
from minagent.workspace import WorkspaceAccess


class _FakeOutput:
    """A stdout stand-in that records what was written."""

    def __init__(self, columns: int = 100) -> None:
        self.columns = columns
        self.chunks: list[str] = []

    def write(self, value: str) -> None:
        self.chunks.append(value)

    def isatty(self) -> bool:
        return False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def _app(tmp_path: Path, window: int = 3000, **settings: Any) -> MinAgent:
    """A session on a small window, with the prompt composed."""
    app = MinAgent(stdout=_FakeOutput())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "Test"
    app.context_window = window
    app.compaction_reserve_tokens = 0
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    app.tool_archive = ToolArchive(str(tmp_path))
    for name, value in settings.items():
        setattr(app, name, value)
    app.rebuild_capabilities()
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.refresh_system_prompt()
    return app


def _fill_to(app: MinAgent, used_tokens: int) -> None:
    """Grow the conversation with tool results until it passes a given size.

    Measuring the target instead of hardcoding a number of turns keeps the test
    about the policy rather than about how big the fixed prompt happens to be.
    Coarse on purpose: it overshoots by up to one turn.
    """
    index = 0
    while app.estimate_current_context_tokens() < used_tokens:
        app.messages.append({"role": "user", "content": f"pregunta {index} " + "x" * 400})
        app.messages.append({"role": "assistant", "content": f"respuesta {index} " + "y" * 400})
        app.messages.append(
            {"role": "tool", "tool_call_id": f"c{index}", "content": f"resultado {index} " + "z" * 2000}
        )
        index += 1


def _grow_to(app: MinAgent, used_tokens: int) -> None:
    """Add exactly one message, sized so the context lands on a precise count.

    The pressure bands are narrow, so overshooting by a whole turn would skip
    past the step this test is about.
    """
    app.messages.append({"role": "user", "content": "x" * 4096})
    for _ in range(16):
        current = app.estimate_current_context_tokens()
        if abs(used_tokens - current) <= 1:
            return
        content = app.messages[-1]["content"]
        # The estimator counts roughly three characters per token.
        resized = content[: max(1, len(content) + (used_tokens - current) * 3)]
        app.messages[-1]["content"] = resized
    raise AssertionError(f"could not land the context on {used_tokens} tokens")


def _sent_tools(app: MinAgent) -> list[str]:
    return [tool["function"]["name"] for tool in app.tools]


# ------------------------------------------------------------------- the policy


def test_a_full_window_sheds_nothing():
    policy = ContextPolicy()
    assert policy.steps_to_shed(700, 1000, 0) == 0


def test_the_high_watermark_starts_the_cascade_one_step_at_a_time():
    policy = ContextPolicy()
    assert policy.steps_to_shed(750, 1000, 0) == 1
    assert policy.steps_to_shed(760, 1000, 1) == 1
    assert policy.steps_to_shed(850, 1000, 1) == 2
    assert policy.steps_to_shed(1000, 1000, 0) == len(SHED_STEPS)


def test_pressure_already_applied_is_never_reduced():
    policy = ContextPolicy()
    assert policy.steps_to_shed(780, 1000, 3) == 3


def test_restoring_waits_for_the_low_watermark():
    """Otherwise a conversation at the threshold would shed and restore every turn."""
    policy = ContextPolicy()
    assert not policy.should_restore(700, 1000)
    assert not policy.should_restore(600, 1000)
    assert policy.should_restore(400, 1000)


def test_an_unknown_window_is_no_pressure():
    assert ContextPolicy().pressure(5000, 0) == 0.0


def test_the_watermarks_must_leave_room_to_restore():
    with pytest.raises(ValueError, match="low < high"):
        ContextPolicy(high_watermark=0.4, low_watermark=0.6)
    with pytest.raises(ValueError):
        ContextPolicy(high_watermark=1.0, low_watermark=1.0)


# ----------------------------------------------------------------- the cascade


def test_a_crowded_session_gives_up_the_index_summaries_first(tmp_path):
    app = _app(tmp_path, memory_enabled=True, web_search_enabled=True)
    app.ensure_web_search_tools()
    _grow_to(app, 2350)
    full = app.estimate_current_context_tokens()
    assert app.regulate_context() == [SHED_INDEX_SUMMARIES]
    assert app._shed_steps == [SHED_INDEX_SUMMARIES]
    assert "Read a workspace file" not in app.messages[0]["content"]
    # The summaries go; both the capability name and the callable tool stay.
    # The name is what load_capability takes, the tool is what the model calls,
    # and a shed index that dropped either would strand the model.
    assert "[files.read; on demand]" in app.messages[0]["content"], "the names must survive to be loadable"
    assert "- read_file, list_directory" in app.messages[0]["content"], "the callable must survive to be usable"
    assert app.estimate_current_context_tokens() < full


def test_pressure_deeper_sheds_memory_hints_and_idle_capabilities(tmp_path):
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    app.memory_hint_context = "Deploy checklist: run make deploy"
    app.load_capabilities(["web"])
    app.note_capability_use("web_search")
    _fill_to(app, 2500)
    shed = app.regulate_context()
    assert SHED_INDEX_SUMMARIES in shed
    assert SHED_MEMORY_HINTS in shed
    assert app.memory_hint_context == ""
    assert "web_search" in _sent_tools(app), "a capability in use stays"


def test_a_task_in_flight_never_loses_the_tool_it_is_using(tmp_path):
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    app.load_capabilities(["web", "files.write"])
    app.note_capability_use("web_search")
    _fill_to(app, 2900)
    app.regulate_context()
    assert "web_search" in _sent_tools(app)
    assert "write_file" not in _sent_tools(app), "what it never touched is what goes"


def test_deep_pressure_unloads_capabilities_the_turn_never_used(tmp_path):
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    app.load_capabilities(["web", "files.write"])
    _fill_to(app, 2900)
    shed = app.regulate_context()
    assert SHED_IDLE_CAPABILITIES in shed
    assert "web_search" not in _sent_tools(app)
    assert "load_capability('web')" in app._unloaded_tool_hint(), "it must still be loadable"


def test_the_shed_is_reported_so_the_user_sees_why_the_prompt_changed(tmp_path):
    app = _app(tmp_path)
    _grow_to(app, 2350)
    app.regulate_context()
    assert "Context trimmed" in app._stdout.text
    assert SHED_INDEX_SUMMARIES in app._stdout.text


def test_old_tool_results_become_archive_references(tmp_path):
    app = _app(tmp_path)
    _fill_to(app, 2950)
    before = app.estimate_current_context_tokens()
    shed = app.regulate_context()
    assert SHED_OLD_TOOL_RESULTS in shed
    assert app.estimate_current_context_tokens() < before
    stubbed = [m for m in app.messages if m.get("role") == "tool" and "archived" in str(m.get("content", ""))]
    assert stubbed, "the old results must stay recoverable by reference"


def test_a_step_with_nothing_to_give_does_not_hide_the_steps_after_it(tmp_path):
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    app.load_capabilities(["web"])
    _fill_to(app, 2950)
    shed = app.regulate_context()
    # No memory hints in this session, so that step frees nothing: it is not
    # reported, and it is not counted as given up either.
    assert SHED_MEMORY_HINTS not in shed
    assert SHED_OLD_TOOL_RESULTS in shed
    assert app._shed_steps == [SHED_OLD_TOOL_RESULTS, SHED_IDLE_CAPABILITIES, SHED_INDEX_SUMMARIES]
    assert app._shed_cursor == len(SHED_STEPS), "the empty step must not be retried next turn"


def test_an_empty_step_is_not_retried_on_the_next_turn(tmp_path):
    app = _app(tmp_path)
    _fill_to(app, 2900)
    app.regulate_context()
    first = list(app._shed_steps)
    app.regulate_context()
    assert app._shed_steps == first, "already given up, and already past"


def test_everything_comes_back_once_the_window_is_free_again(tmp_path):
    app = _app(tmp_path)
    _grow_to(app, 2350)
    app.regulate_context()
    assert app._shed_steps
    del app.messages[1:]
    app.regulate_context()
    assert app._shed_steps == []
    assert "Read a workspace file" in app.messages[0]["content"]


def test_an_uncrowded_session_is_left_alone(tmp_path):
    app = _app(tmp_path)
    assert app.regulate_context() == []
    assert app._shed_steps == []
    assert app._stdout.text == ""


# ------------------------------------------------- is the meter telling the truth?


def test_no_request_means_nothing_to_compare_against(tmp_path):
    app = _app(tmp_path)
    assert app.prompt_calibration()["samples"] == 0
    app.print_prompt_token_breakdown()
    assert "no request sent yet" in app._stdout.text


def test_the_meter_is_compared_against_what_the_endpoint_reported(tmp_path):
    app = _app(tmp_path)
    app.record_prompt_calibration(1000, 900)
    app.record_prompt_calibration(2000, 1800)
    calibration = app.prompt_calibration()
    assert calibration["samples"] == 2
    assert calibration["last_ratio"] == pytest.approx(0.9)
    assert calibration["mean_ratio"] == pytest.approx(0.9)
    assert calibration["estimated"] == 2000
    assert calibration["reported"] == 1800


def test_the_last_pair_and_the_mean_are_reported_separately(tmp_path):
    """One request compared with itself is a fact; the mean is a different one."""
    app = _app(tmp_path)
    app.record_prompt_calibration(1000, 1000)
    app.record_prompt_calibration(1000, 500)
    calibration = app.prompt_calibration()
    assert calibration["last_ratio"] == pytest.approx(0.5)
    assert calibration["mean_ratio"] == pytest.approx(0.75)


def test_a_nothing_useful_sample_is_ignored(tmp_path):
    app = _app(tmp_path)
    app.record_prompt_calibration(0, 900)
    app.record_prompt_calibration(1000, 0)
    assert app.prompt_calibration()["samples"] == 0


def test_only_the_last_few_samples_are_kept(tmp_path):
    app = _app(tmp_path)
    for index in range(30):
        app.record_prompt_calibration(1000 + index, 1000 + index)
    assert len(app._prompt_calibration) <= 12


def test_a_meter_that_runs_low_is_called_out_with_its_consequence(tmp_path):
    app = _app(tmp_path)
    app.record_prompt_calibration(1000, 1300)
    app.print_prompt_token_breakdown()
    text = " ".join(app._stdout.text.split())
    assert "the meter runs 30% low" in text
    assert "really trims at about" in text, "the governor acts on the meter, so its bias matters"


def test_a_meter_that_runs_high_only_trims_early_which_is_safe(tmp_path):
    app = _app(tmp_path)
    app.record_prompt_calibration(2000, 1500)
    app.print_prompt_token_breakdown()
    text = " ".join(app._stdout.text.split())
    assert "the meter runs 25% high" in text
    assert "really trims at about" not in text


def test_a_meter_that_is_close_enough_is_not_dramatised(tmp_path):
    app = _app(tmp_path)
    app.record_prompt_calibration(1000, 1010)
    app.print_prompt_token_breakdown()
    text = " ".join(app._stdout.text.split())
    assert "the meter runs 1% low" in text
    assert "really trims at about" not in text


def test_the_mean_is_shown_once_there_is_more_than_one_request(tmp_path):
    app = _app(tmp_path)
    app.record_prompt_calibration(1000, 1000)
    app.print_prompt_token_breakdown()
    assert "over 1 requests" not in " ".join(app._stdout.text.split())
    app.record_prompt_calibration(1000, 990)
    app.print_prompt_token_breakdown()
    assert "mean 1% high over 2 requests" in " ".join(app._stdout.text.split())


async def test_a_turn_records_what_the_endpoint_really_read(tmp_path):
    """The estimate is taken before the request; the usage after it is the truth."""
    app = _app(tmp_path)
    responses = [
        {"payload": {"usage": {"prompt_tokens": 4321}}, "message": {"content": "hola", "tool_calls": []}},
    ]

    async def fake(messages, options=None):
        return responses.pop(0)

    app.call_chat_completions = fake  # type: ignore[method-assign]
    app.messages.append({"role": "user", "content": "dime algo"})
    await app.request_assistant_turn(None)
    calibration = app.prompt_calibration()
    assert calibration["samples"] == 1
    assert calibration["reported"] == 4321
    assert calibration["estimated"] > 0
