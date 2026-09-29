"""Tests for the capability catalog and the on-demand loading it drives.

The catalog is what keeps the tool schemas out of every request, so these tests
check the property that makes that safe: a tool is either claimed by a
capability, or it is unreachable, and the model is always told how to get it.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from minagent.app import MinAgent, build_terminal_tool
from minagent.capabilities import (
    DEFAULT_CAPABILITY_IDLE_TURNS,
    LOAD_CAPABILITY_TOOL_NAME,
    Capability,
    CapabilityCatalog,
    build_builtin_capabilities,
    build_mcp_capabilities,
)
from minagent.context import estimate_text_tokens
from minagent.errors import AgentError
from minagent.jsutil import json_stringify
from minagent.web_search import WebSearchClient
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


def _app(tmp_path, **settings: Any) -> MinAgent:
    """An app with the features a test asks for, and nothing else."""
    app = MinAgent(stdout=_FakeOutput())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "Test"
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    for name, value in settings.items():
        setattr(app, name, value)
    app.rebuild_capabilities()
    return app


def _catalog() -> CapabilityCatalog:
    return CapabilityCatalog(
        entries=(
            Capability(name="files.read", summary="Read a file", tool_names=("read_file", "list_directory")),
            Capability(name="files.write", summary="Write a file", tool_names=("write_file", "edit_file")),
            Capability(
                name="web",
                summary="Search the web",
                tool_names=("web_search",),
                guidance="Be careful.",
                eager=True,
            ),
        )
    )


def _sent_tools(app: MinAgent) -> list[str]:
    return [tool["function"]["name"] for tool in app.tools]


# ------------------------------------------------------------------ the catalog


def test_only_eager_capabilities_start_loaded():
    catalog = _catalog()
    assert catalog.loaded == {"web"}
    assert catalog.loaded_tool_names() == ["web_search"]
    assert [entry.name for entry in catalog.unloaded_entries()] == ["files.read", "files.write"]


def test_loading_reports_only_what_was_not_loaded_yet():
    catalog = _catalog()
    assert [entry.name for entry in catalog.load(["files.read", "web"])] == ["files.read"]
    assert [entry.name for entry in catalog.load(["web"])] == []
    assert catalog.loaded_tool_names() == ["read_file", "list_directory", "web_search"]


def test_unknown_names_are_reported_rather_than_ignored():
    catalog = _catalog()
    assert catalog.unknown_names(["web", "nope", " files.write "]) == ["nope"]


def test_a_tool_is_mapped_back_to_the_capability_that_owns_it():
    catalog = _catalog()
    entry = catalog.capability_for_tool("web_search")
    assert entry is not None and entry.name == "web"
    assert catalog.capability_for_tool("run_terminal") is None


def test_guidance_is_only_published_for_what_is_loaded():
    catalog = _catalog()
    assert [section["name"] for section in catalog.loaded_guidance()] == ["Capability: web"]
    catalog.load(["files.read"])
    assert [section["name"] for section in catalog.loaded_guidance()] == ["Capability: web"]


def test_an_unused_capability_is_dropped_after_the_idle_turns():
    catalog = _catalog()
    catalog.load(["files.read"])
    assert catalog.unload_unused([], DEFAULT_CAPABILITY_IDLE_TURNS) == []
    assert [entry.name for entry in catalog.unload_unused([], DEFAULT_CAPABILITY_IDLE_TURNS)] == ["files.read"]
    assert "files.read" not in catalog.loaded
    # The always-loaded set is never aged out, or the agent could not recover.
    assert catalog.unload_unused([], DEFAULT_CAPABILITY_IDLE_TURNS) == []


def test_a_capability_the_agent_uses_stays_loaded_however_long_the_task_runs():
    catalog = _catalog()
    catalog.load(["files.read"])
    for _ in range(5):
        assert catalog.unload_unused(["files.read"], DEFAULT_CAPABILITY_IDLE_TURNS) == []
    assert "files.read" in catalog.loaded
    assert [entry.name for entry in catalog.unload_unused([], 1)] == ["files.read"]


def test_an_idle_limit_of_zero_turns_aging_off():
    catalog = _catalog()
    catalog.load(["files.read"])
    for _ in range(5):
        assert catalog.unload_unused([], 0) == []
    assert "files.read" in catalog.loaded


def test_resetting_returns_to_the_always_loaded_set():
    catalog = _catalog()
    catalog.load(["files.read"])
    catalog.reset_loaded()
    assert catalog.loaded == {"web"}
    assert catalog.idle_turns == {"web": 0}


def test_the_index_is_one_line_per_capability_and_names_the_loader():
    catalog = _catalog()
    catalog.load(["files.read"])
    index = catalog.render_index()
    lines = index.splitlines()
    assert len(lines) == 1 + len(catalog.entries)
    assert LOAD_CAPABILITY_TOOL_NAME in lines[0]
    assert "- read_file, list_directory  [files.read: Read a file; loaded]" in index
    assert "- web_search  [web: Search the web; always]" in index
    assert "- write_file, edit_file  [files.write: Write a file; on demand]" in index


def test_the_index_stops_naming_every_tool_of_a_wide_capability():
    catalog = _catalog()
    catalog.entries = (
        Capability(
            name="mcp.wide",
            summary="Tools from a server with many tools",
            tool_names=tuple(f"mcp_0_{index}_tool" for index in range(9)),
        ),
    )
    line = catalog.render_index().splitlines()[1]
    assert "mcp_0_0_tool, mcp_0_1_tool, mcp_0_2_tool" in line
    assert "+6 more" in line
    assert "mcp_0_8_tool" not in line


def test_the_index_says_how_each_tool_is_called(tmp_path):
    """So the model can use a tool without first spending a request loading it."""
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    index = app.capability_index_section()
    assert index is not None
    assert "web_search(query)" in index["content"]
    assert "web_fetch(url)" in index["content"]


def test_the_index_can_be_rendered_as_names_only():
    catalog = _catalog()
    catalog.load(["files.read"])
    compact = catalog.render_index(compact=True)
    assert "Read a file" not in compact
    assert "- read_file, list_directory  [files.read; loaded]" in compact
    assert "- web_search  [web; always]" in compact


def test_the_index_leads_with_the_callable_tool_not_the_capability_name():
    """A small model copies the first name on the line, so that must be a tool.

    Leading with the capability name made it call ``files.write``, which is not
    a tool, and the turn ended with a write that never happened.
    """
    catalog = _catalog()
    line = next(row for row in catalog.render_index(hints={"read_file": "read_file(path)"}).splitlines() if "files.read" in row)
    assert line.startswith("- read_file(path), list_directory")
    assert line.index("read_file") < line.index("files.read")


def test_the_compact_index_still_names_the_tools_it_offers():
    """Under pressure the summaries go, but the callable names must stay.

    An index that cannot be acted on is not a saving: the model cannot call what
    it cannot see, and the shed only happens when the window is already full.
    """
    catalog = _catalog()
    catalog.load(["files.read"])
    compact = catalog.render_index(compact=True)
    assert "read_file" in compact
    assert "files.read" in compact


def test_the_loader_does_not_repeat_the_catalogue():
    """The index is the one copy of the names; repeating them here is pure cost."""
    catalog = _catalog()
    tool = catalog.load_capability_tool()
    assert tool["function"]["name"] == LOAD_CAPABILITY_TOOL_NAME
    assert "enum" not in tool["function"]["parameters"]["properties"]["capabilities"]["items"]
    for entry in catalog.entries:
        assert entry.name not in tool["function"]["description"]


def test_a_name_the_index_does_not_have_comes_back_with_the_real_ones(tmp_path):
    app = _app(tmp_path)
    report = app.load_capabilities(["files.read", "telepathy"])
    assert "Unknown capabilities: telepathy" in report
    assert "files.read: loaded" in report


def test_mcp_guidance_is_bounded_and_carries_the_untrusted_data_warning():
    entries = build_mcp_capabilities(
        {"mcp_0_0_echo": {"server_name": "one", "remote_tool_name": "echo"}},
        [{"server_name": "one", "instructions": "Echo politely.\x00"}],
    )
    assert [entry.name for entry in entries] == ["mcp.one", "mcp.manage"]
    assert entries[0].tool_names == ("mcp_0_0_echo",)
    assert "Echo politely." in entries[0].guidance and "\x00" not in entries[0].guidance
    assert "untrusted data" in entries[0].guidance
    # With no server connected the authoring tool still has to be reachable, or
    # the agent could not create one.
    assert [entry.name for entry in build_mcp_capabilities({}, [])] == ["mcp.manage"]
    assert build_mcp_capabilities({}, [], authoring_enabled=False) == []


def test_builtin_capabilities_follow_the_enabled_features():
    off = build_builtin_capabilities(terminal_mode="off")
    assert [entry.name for entry in off] == [
        "files.read",
        "files.write",
        "files.delete",
        "tool_output.recall",
        "files.download",
    ]
    on = build_builtin_capabilities(
        terminal_mode="ask",
        terminal_environment="System: Linux.",
        skill_context="Skills: one.",
        memory_enabled=True,
        web_search_enabled=True,
    )
    by_name = {entry.name: entry for entry in on}
    assert set(by_name) == {
        "files.read",
        "files.write",
        "files.delete",
        "tool_output.recall",
        "files.download",
        "terminal",
        "skills",
        "memory",
        "web",
    }
    assert "user approval is required" in by_name["terminal"].guidance
    assert "System: Linux." in by_name["terminal"].guidance
    assert by_name["skills"].guidance == "Skills: one."
    assert "recall" in by_name["memory"].tool_names
    assert by_name["web"].tool_names == ("web_search", "web_fetch")


# ------------------------------------------------------------- the app session


def test_the_session_starts_with_only_what_it_cannot_do_without(tmp_path):
    """The loader and the tool that reads back an archived result; nothing else."""
    app = _app(tmp_path)
    assert _sent_tools(app) == ["recall_tool_output", LOAD_CAPABILITY_TOOL_NAME]
    assert "read_file" not in _sent_tools(app)
    assert "load_capability('files.read')" in app._unloaded_tool_hint()


def test_every_registered_tool_belongs_to_a_capability(tmp_path):
    """A tool no capability claims could never be loaded, so it would vanish."""
    app = _app(
        tmp_path,
        terminal_mode="auto",
        skills_enabled=True,
        memory_enabled=True,
        web_search_enabled=True,
        mcp_enabled=True,
    )
    app.register_tool_schemas([build_terminal_tool()])
    app.ensure_skill_tools()
    app.ensure_mcp_tools()
    app.ensure_memory_tools()
    app.ensure_web_search_tools()
    app.ensure_download_tools()
    catalog = app.capabilities
    assert catalog is not None
    claimed = {name for entry in catalog.entries for name in entry.tool_names}
    # The loader is not stored with the schemas: it is composed from the catalog
    # itself, which is why nothing else may go unclaimed.
    assert set(app._tool_schemas) == claimed
    assert LOAD_CAPABILITY_TOOL_NAME in _sent_tools(app)


def test_loading_publishes_the_tools_and_the_guidance_of_a_capability(tmp_path):
    app = _app(tmp_path, terminal_mode="auto")
    app.register_tool_schemas([build_terminal_tool()])
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.refresh_system_prompt()

    report = app.load_capabilities(["terminal"])
    assert "terminal: loaded" in report
    assert "run_terminal" in _sent_tools(app)
    assert "run_terminal" in app.messages[0]["content"]
    assert "load_capability('terminal')" not in app._unloaded_tool_hint()

    again = app.load_capabilities(["terminal"])
    assert "terminal: already loaded" in again


def test_loading_reports_a_name_the_index_does_not_have(tmp_path):
    app = _app(tmp_path)
    report = app.load_capabilities(["telepathy"])
    assert "Unknown capabilities: telepathy" in report
    assert "files.write" in report, "the reply should list what it could have loaded"


def test_the_loader_takes_one_name_or_several(tmp_path):
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    assert "web: loaded" in app.load_capabilities("web")
    report = app.load_capabilities(["files.write", "web"])
    assert "files.write: loaded" in report and "web: already loaded" in report
    assert {"web_search", "write_file"} <= set(_sent_tools(app))


def test_loading_nothing_is_reported_rather_than_silently_accepted(tmp_path):
    app = _app(tmp_path)
    assert app.load_capabilities([]) == "Name at least one capability from the index."
    assert app.load_capabilities(["  "]) == "Name at least one capability from the index."


class _StubSearchClient:
    """A web client that answers nothing, because the call never gets that far."""

    async def search(self, query: str, limit: int) -> list[dict[str, str]]:
        return []


async def test_calling_a_tool_of_an_unloaded_capability_loads_it_and_runs(tmp_path):
    """Refusing costs a whole request to repeat what the index already said."""
    app = _app(tmp_path, web_search_enabled=True)
    app.web_search_client = cast(WebSearchClient, _StubSearchClient())
    app.ensure_web_search_tools()
    assert "web_search" not in _sent_tools(app)
    result = await app.execute_tool("web_search", {"query": "anything"})
    assert "web was loaded on demand" in result
    assert "web_search" in _sent_tools(app)
    assert app.capabilities is not None and "web" in app.capabilities.loaded


async def test_calling_a_capability_by_name_loads_it_and_names_its_tools(tmp_path):
    app = _app(tmp_path)
    assert "read_file" not in _sent_tools(app)
    result = await app.execute_tool("files.read", {})
    assert "files.read is a capability, not a tool" in result
    assert "read_file" in result
    assert "read_file" in _sent_tools(app), "the next call must work straight away"


async def test_a_tool_no_capability_owns_is_still_an_error(tmp_path):
    app = _app(tmp_path)
    with pytest.raises(AgentError, match="Tool is not available: teleport"):
        await app.execute_tool("teleport", {})


async def test_using_a_tool_marks_its_capability_as_used(tmp_path):
    (tmp_path / "notes.txt").write_text("hola\n")
    app = _app(tmp_path)
    app.load_capabilities(["files.read"])
    await app.execute_tool("read_file", {"path": "notes.txt"})
    assert app._capabilities_used_this_turn == {"files.read"}
    assert app.age_capabilities() == []


def test_a_missing_capability_note_names_what_to_load(tmp_path):
    app = _app(tmp_path)
    note = app.missing_capability_note()
    assert "load_capability('files.write') for write_file, edit_file, create_directory" in note
    assert "load_capability" in app.plan_without_action_note()


def test_the_index_marks_what_is_loaded_as_it_changes(tmp_path):
    app = _app(tmp_path)
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.refresh_system_prompt()
    # The index marks each capability as always loaded, loaded, or on demand.
    assert "; on demand]" in app.messages[0]["content"]
    assert "; always]" in app.messages[0]["content"]
    app.load_capabilities(["files.write"])
    assert "write_file(path, content), edit_file(path, old_text, new_text)" in app.messages[0]["content"]
    assert "; loaded]" in app.messages[0]["content"]
    assert "; on demand]" in app.messages[0]["content"]


def test_a_capability_the_agent_ignores_is_unloaded(tmp_path):
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    app.load_capabilities(["web"])
    assert "web_search" in _sent_tools(app)

    for _ in range(DEFAULT_CAPABILITY_IDLE_TURNS):
        app.note_capability_use("web_search")
        assert app.age_capabilities() == [], "a used capability must not be dropped"
    assert "web_search" in _sent_tools(app)

    for _ in range(DEFAULT_CAPABILITY_IDLE_TURNS - 1):
        app.age_capabilities()
    assert app.age_capabilities() == ["web"]
    assert "web_search" not in _sent_tools(app)
    assert app.capabilities is not None
    assert "tool_output.recall" in app.capabilities.loaded, "the always-loaded set survives"


def test_aging_reports_what_left_in_the_terminal(tmp_path):
    app = _app(tmp_path)
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.refresh_system_prompt()
    app.load_capabilities(["files.write"])
    for _ in range(DEFAULT_CAPABILITY_IDLE_TURNS - 1):
        app.age_capabilities()
    app.report_capability_aging()
    assert "Capability unloaded" in app._stdout.text
    assert "files.write" in app._stdout.text


def test_rebuilding_the_catalog_keeps_what_is_loaded_and_how_idle_it_is(tmp_path):
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    app.load_capabilities(["web"])
    app.age_capabilities()
    assert app.capabilities is not None and app.capabilities.idle_turns["web"] == 1
    app.rebuild_capabilities()
    assert app.capabilities is not None
    assert "web" in app.capabilities.loaded
    assert app.capabilities.idle_turns["web"] == 1, "a skill appearing must not reset the clock"
    assert "web_search" in _sent_tools(app)


def test_a_new_conversation_starts_from_the_always_loaded_set(tmp_path):
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    app.load_capabilities(["web", "files.read"])
    app._steps_this_turn = ["read_file"]
    app._reset_conversation_state()
    assert app.capabilities is not None and app.capabilities.loaded == {"tool_output.recall"}
    assert "web_search" not in _sent_tools(app)
    assert app._steps_this_turn == []


def test_the_index_is_cheaper_than_the_schemas_it_replaces(tmp_path):
    """The whole point of loading on demand: the index must be the smaller half."""
    app = _app(tmp_path, web_search_enabled=True)
    app.ensure_web_search_tools()
    index = app.capability_index_section()
    assert index is not None
    eager = estimate_text_tokens(json_stringify(app.tools))
    assert estimate_text_tokens(index["content"]) < eager
