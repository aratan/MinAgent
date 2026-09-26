"""Core behaviour tests: workspace safety, attachments, streaming, and the editor."""

from __future__ import annotations

import asyncio
import json
import os
import pty
import re
import sys
import time
import tty
from datetime import datetime
from typing import Any

import pytest

from minagent.app import (
    _ANNOUNCED_ACTION,
    _MISSING_CAPABILITY_REQUEST,
    UI_COLORS,
    MinAgent,
    build_terminal_tool,
)
from minagent.attachments import prepare_user_message
from minagent.config import Config, load_configuration, parse_directory_entry_limit
from minagent.context import chunk_summary_transcript, compress_for_context, estimate_text_tokens
from minagent.editor import (
    AUTOCOMPLETE_PANEL_ROWS,
    Key,
    PasteState,
    build_autocomplete_state,
    format_autocomplete_panel,
    handle_autocomplete_keypress,
    handle_control_j_input,
    handle_pasted_input,
)
from minagent.errors import AgentError
from minagent.init_project import collect_project_essentials
from minagent.line_editor import LineEditor
from minagent.markdown_terminal import create_terminal_rendering
from minagent.openai import read_streaming_response
from minagent.request_cache import RequestCache
from minagent.secrets import approval_preview
from minagent.skills import create_skill_tools, execute_skill_tool, slugify_skill_name
from minagent.terminal_command import run_terminal_command
from minagent.terminal_text import (
    terminal_columns,
    terminal_text_width,
    truncate_terminal_text,
    wrap_styled_segments,
)
from minagent.tool_archive import MAX_ARCHIVE_RECALL_CHARS, MAX_ARCHIVED_CHARS, ToolArchive
from minagent.workspace import WorkspaceAccess


class _TurnStopped(Exception):
    """Raised by a fake endpoint to end a turn as soon as its options are captured."""


READ_STOP_MARKER = re.compile(
    r"\n\n\[Read stopped at the output limit\. Continue with offset=1, column=(\d+)\.\]$"
)


class FakeOutput:
    """A stdout stand-in that records what was written."""

    def __init__(self, columns: int = 80) -> None:
        self.columns = columns
        self.chunks: list[str] = []

    def write(self, value: str) -> None:
        self.chunks.append(value)

    def isatty(self) -> bool:
        return False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


class FakeEditor:
    """A minimal editor stand-in for the paste and completion tests."""

    def __init__(self, line: str = "", cursor: int = 0) -> None:
        self.line = line
        self.cursor = cursor
        self.prev_rows = 0
        self.is_completion_enabled = True

    def prompt(self, preserve: bool = False) -> None:
        return None

    def mark_multiline(self) -> None:
        return None


def make_rendering(columns: int):
    """Build a terminal renderer that records output without colouring."""
    output = FakeOutput(columns)
    buffer: list[str] = []

    def ui_text(value: Any, *args: Any) -> str:
        return str(value)

    def ui_print(value: str) -> None:
        buffer.append(f"{value}\n")

    def print_(value: str) -> None:
        buffer.append(f"{value}\n")

    rendering = create_terminal_rendering(
        stdout=output,
        get_use_color=lambda: False,
        ui_colors={"assistantBackground": [0, 0, 0]},
        ui_text=ui_text,
        ui_print=ui_print,
        print=print_,
    )
    return rendering, output, buffer


class FakeStreamingResponse:
    """A minimal ``aiter_bytes`` stand-in for the SSE parser."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events

    async def aiter_bytes(self):
        for event in self.events:
            yield f"data: {json.dumps(event)}\n\n".encode()
        yield b"data: [DONE]\n\n"


# ---------------------------------------------------------------- workspace


async def test_read_file_can_continue_within_a_long_unicode_line(tmp_path):
    content = "\U0001F600abc" * 12_000
    (tmp_path / "long.txt").write_text(content, encoding="utf-8")
    access = WorkspaceAccess(str(tmp_path), "test")
    column = 1
    collected = ""
    requests = 0
    while True:
        output = await access.read_file({"path": "long.txt", "offset": 1, "column": column})
        marker = READ_STOP_MARKER.search(output)
        collected += output[: marker.start()] if marker else output
        requests += 1
        if not marker:
            break
        column = int(marker.group(1))
        assert requests < 10
    assert requests > 1
    assert collected == content


async def test_inventory_excludes_generated_directories_and_obeys_the_listing_limit(tmp_path):
    for directory in (".git", "node_modules", "src"):
        (tmp_path / directory).mkdir()
    (tmp_path / ".git" / "secret").write_text("x")
    (tmp_path / "node_modules" / "package.js").write_text("x")
    (tmp_path / "src" / "main.py").write_text("x")
    complete = await WorkspaceAccess(str(tmp_path), "test").refresh_inventory()
    assert complete["files"] == ["src/main.py"]
    empty = await WorkspaceAccess(str(tmp_path), "test", 0).refresh_inventory()
    assert empty["files"] == ["src/main.py"]


async def test_disabled_inventory_still_loads_agents_md_and_permits_a_full_listing(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Project guidance")
    (tmp_path / "README.md").write_text("Project")
    assert parse_directory_entry_limit(None) == 0
    access = WorkspaceAccess(str(tmp_path), "test", 0)
    normal = await access.refresh_inventory()
    assert normal["snapshot"] == ""
    assert normal["files"] == ["AGENTS.md", "README.md"]
    assert "Project guidance" in normal["agents_context"]
    for_init = await access.refresh_inventory(include_snapshot=True, list_limit_override=-1)
    assert "[FILE] README.md" in for_init["snapshot"]
    assert "README.md" in for_init["files"]


async def test_oversized_agents_md_is_omitted_from_model_guidance(tmp_path):
    (tmp_path / "AGENTS.md").write_text("a" * 70_000)
    inventory = await WorkspaceAccess(str(tmp_path), "test").refresh_inventory()
    assert re.search(r"exceeds the .* byte limit", inventory["agents_context"])
    assert inventory["agents_content"] == ""


async def test_workspace_rejects_hard_links(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    os.link(outside, root / "hard.txt")
    access = WorkspaceAccess(str(root), "test")
    with pytest.raises(Exception, match="Hard-linked"):
        await access.read_file({"path": "hard.txt"})


async def test_workspace_rejects_file_symbolic_links(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (root / "soft.txt").symlink_to(outside)
    access = WorkspaceAccess(str(root), "test")
    with pytest.raises(Exception, match="Symbolic links"):
        await access.read_file({"path": "soft.txt"})


async def test_workspace_rejects_directory_symlinks(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("outside")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    access = WorkspaceAccess(str(root), "test")
    with pytest.raises(Exception, match="Symbolic links"):
        await access.read_file({"path": "linked/file.txt"})
    with pytest.raises(Exception, match="Symbolic links"):
        await access.write_file({"path": "linked/new.txt", "content": "x"})
    with pytest.raises(Exception, match="Symbolic links"):
        await access.list_directory({"path": "linked"})


async def test_list_directory_includes_hidden_entries_without_recursing(tmp_path):
    (tmp_path / "child").mkdir()
    (tmp_path / ".hidden").write_text("hidden")
    (tmp_path / "child" / "nested.txt").write_text("nested")
    access = WorkspaceAccess(str(tmp_path), "test")
    listing = await access.list_directory()
    assert "[FILE] .hidden" in listing["tool_text"]
    assert "[DIR] child/" in listing["tool_text"]
    assert "nested.txt" not in listing["tool_text"]
    limited = await access.list_directory({"limit": 1})
    assert "Call list_directory with a larger limit" in limited["tool_text"]
    with pytest.raises(Exception, match="requires a directory"):
        await access.list_directory({"path": ".hidden"})
    with pytest.raises(Exception, match="outside the current workspace"):
        await access.list_directory({"path": ".."})


async def test_file_tools_accept_a_redundant_workspace_directory_prefix(tmp_path):
    root = tmp_path / "Test"
    root.mkdir()
    (root / "note.txt").write_text("one")
    access = WorkspaceAccess(str(root), "Test")
    assert access.resolve_path("Test/note.txt") == os.path.join(str(root), "note.txt")
    assert await access.read_file({"path": "Test/note.txt"}) == "one"
    await access.edit_file({"path": "Test/note.txt", "old_text": "one", "new_text": "two"})
    assert (root / "note.txt").read_text() == "two"
    await access.write_file({"path": "Test/nested/new.txt", "content": "new"})
    assert (root / "nested" / "new.txt").read_text() == "new"
    await access.delete_file({"path": "Test/note.txt"})
    await access.delete_directory({"path": "Test/nested"})
    assert not (root / "note.txt").exists()
    assert not (root / "nested" / "new.txt").exists()
    with pytest.raises(Exception, match="requires a file path.*workspace directory"):
        await access.read_file({"path": "Test"})
    with pytest.raises(Exception, match="root cannot be deleted"):
        await access.delete_directory({"path": "Test"})
    with pytest.raises(Exception, match="outside the current workspace"):
        access.resolve_path("Test/../../outside.txt")


async def test_a_real_subdirectory_named_after_the_workspace_keeps_its_own_paths(tmp_path):
    root = tmp_path / "Test"
    (root / "Test").mkdir(parents=True)
    (root / "Test" / "note.txt").write_text("child")
    (root / "note.txt").write_text("root")
    access = WorkspaceAccess(str(root), "Test")
    assert await access.read_file({"path": "Test/note.txt"}) == "child"
    assert await access.read_file({"path": "note.txt"}) == "root"
    await access.write_file({"path": "Test/new.txt", "content": "nested"})
    assert (root / "Test" / "new.txt").read_text() == "nested"


async def test_an_explicitly_relative_path_creates_a_same_named_subdirectory(tmp_path):
    root = tmp_path / "Test"
    root.mkdir()
    access = WorkspaceAccess(str(root), "Test")
    await access.write_file({"path": "./Test/new.txt", "content": "nested"})
    assert (root / "Test" / "new.txt").read_text() == "nested"


async def test_a_same_named_directory_symlink_is_never_treated_as_a_prefix(tmp_path):
    root = tmp_path / "Test"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("outside")
    (root / "Test").symlink_to(outside, target_is_directory=True)
    access = WorkspaceAccess(str(root), "Test")
    with pytest.raises(Exception, match="Symbolic links"):
        await access.read_file({"path": "Test/file.txt"})


async def test_write_and_edit_verify_their_persisted_contents(tmp_path):
    access = WorkspaceAccess(str(tmp_path), "test")
    await access.write_file({"path": "nested/example.txt", "content": "one"})
    await access.edit_file({"path": "nested/example.txt", "old_text": "one", "new_text": "two"})
    assert await access.read_file({"path": "nested/example.txt"}) == "two"


async def test_deletion_stays_within_a_validated_subdirectory(tmp_path):
    access = WorkspaceAccess(str(tmp_path), "test")
    await access.write_file({"path": "folder/one.txt", "content": "one"})
    await access.write_file({"path": "keep.txt", "content": "keep"})
    with pytest.raises(Exception, match="root cannot be deleted"):
        await access.delete_directory({"path": "."})
    await access.delete_directory({"path": "folder"})
    assert (tmp_path / "keep.txt").read_text() == "keep"


# ------------------------------------------------------------------- /init


async def test_project_initialization_includes_source_files_with_unfamiliar_names(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("Project")
    (tmp_path / "src" / "minagent.py").write_text("x = 1")
    access = WorkspaceAccess(str(tmp_path), "test")
    result = await collect_project_essentials(str(tmp_path), access.read_raw_file)
    assert [file["path"] for file in result["files"]] == ["README.md", "src/minagent.py"]


async def test_project_initialization_prioritizes_the_entry_point_within_a_budget(tmp_path):
    root = tmp_path / "MinAgent"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("r" * 1000)
    (root / "src" / "alpha.py").write_text("a" * 1000)
    (root / "src" / "minagent.py").write_text("m" * 1000)
    access = WorkspaceAccess(str(root), "test")
    result = await collect_project_essentials(str(root), access.read_raw_file, max_total_chars=1024)
    assert any(file["path"] == "src/minagent.py" for file in result["files"])
    assert sum(len(file["content"]) for file in result["files"]) <= 1024


async def test_init_writes_agents_md_from_a_real_session(tmp_path, monkeypatch):
    """The app's own budget must be accepted, however small the context window is."""
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# Demo\n\nA small project.\n")
    (root / "src" / "main.py").write_text("print('hola')\n")
    app, _output = _make_app(80)
    app.context_window = 8192
    app.compaction_reserve_tokens = 1024
    app.root_directory = str(root)
    app.workspace_name = root.name
    app.workspace_access = WorkspaceAccess(str(root), root.name, 0)

    async def fake_call(messages, options=None):
        return {
            "payload": {"usage": {}},
            "message": {"content": "```markdown\n# AGENTS.md\n\nDemo.\n```", "tool_calls": []},
        }

    monkeypatch.setattr(app, "call_chat_completions", fake_call)
    assert await app.initialize_project("", None) == "created"
    assert (root / "AGENTS.md").read_text() == "# AGENTS.md\n\nDemo.\n"


async def test_init_rejects_a_non_integer_file_budget(tmp_path):
    access = WorkspaceAccess(str(tmp_path), "test")
    with pytest.raises(AgentError, match="whole-number"):
        await collect_project_essentials(str(tmp_path), access.read_raw_file, max_total_chars=4096.0)


# ------------------------------------------------------------- attachments


async def test_text_attachments_keep_valid_utf8_at_the_excerpt_boundary(tmp_path):
    (tmp_path / "note.txt").write_text("a" * (48 * 1024 - 1) + "\U0001F600end", encoding="utf-8")
    result = await prepare_user_message(
        "note.txt", ["note.txt"], WorkspaceAccess(str(tmp_path), "test"), ["text"]
    )
    assert result["events"][0]["kind"] == "attached"
    assert "File content truncated" in result["message"]["content"][0]["text"]


async def test_workspace_creates_nested_files_and_folders(tmp_path):
    access = WorkspaceAccess(str(tmp_path), "Test", -1)
    assert await access.write_file({"path": "app/src/main.py", "content": "print(1)"}) == "Wrote app/src/main.py."
    assert (tmp_path / "app" / "src" / "main.py").read_text() == "print(1)"
    assert await access.create_directory({"path": "docs/guias"}) == "Created directory docs/guias."
    assert (tmp_path / "docs" / "guias").is_dir()
    assert "already exists" in await access.create_directory({"path": "docs"})
    with pytest.raises(AgentError, match="not a directory"):
        await access.create_directory({"path": "app/src/main.py"})
    with pytest.raises(AgentError, match="create_directory"):
        await access.write_file({"path": "otra/", "content": ""})


# ------------------------------------------------------------------ skills


async def test_skill_resource_reader_rejects_binary_and_invalid_utf8(tmp_path):
    (tmp_path / "valid.txt").write_text("reference")
    (tmp_path / "binary.txt").write_bytes(b"a\x00b")
    (tmp_path / "invalid.txt").write_bytes(b"\xff")
    skills = [{"name": "demo", "directory": str(tmp_path), "instructions": "instructions"}]
    assert [tool["function"]["name"] for tool in create_skill_tools()] == ["load_skill", "write_skill"]
    assert "instructions" in (await execute_skill_tool("load_skill", {"name": "demo"}, skills))["tool_text"]
    assert "reference" in (
        await execute_skill_tool("load_skill", {"name": "demo", "path": "valid.txt"}, skills)
    )["tool_text"]
    with pytest.raises(Exception, match="binary"):
        await execute_skill_tool("load_skill", {"name": "demo", "path": "binary.txt"}, skills)
    with pytest.raises(Exception):
        await execute_skill_tool("load_skill", {"name": "demo", "path": "invalid.txt"}, skills)


# --------------------------------------------------------------- streaming


async def test_streaming_client_keeps_a_response_stopped_at_the_token_limit():
    response = FakeStreamingResponse(
        [{"choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]}]
    )
    result = await read_streaming_response(response)
    assert result["payload"]["truncated"] is True
    assert result["payload"]["finish_reason"] == "length"
    assert result["message"]["content"] == "partial"


async def test_a_turn_continues_a_response_stopped_at_the_token_limit(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)
    responses = [
        {
            "payload": {"usage": {}, "finish_reason": "length", "truncated": True},
            "message": {"content": "Primera mitad ", "tool_calls": []},
        },
        {
            "payload": {"usage": {}, "finish_reason": "stop"},
            "message": {"content": "y segunda mitad.", "tool_calls": []},
        },
    ]

    async def next_response(messages, options=None):
        return responses.pop(0)

    monkeypatch.setattr(app, "call_chat_completions", next_response)
    assert await app.request_assistant_turn(None) == "Primera mitad y segunda mitad."
    assert app.messages[-1]["content"] == "Primera mitad y segunda mitad."


async def test_a_truncated_response_without_text_stops_cleanly(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)

    async def empty_truncated(messages, options=None):
        return {
            "payload": {"usage": {}, "finish_reason": "length", "truncated": True},
            "message": {"content": None, "tool_calls": []},
        }

    monkeypatch.setattr(app, "call_chat_completions", empty_truncated)
    assert await app.request_assistant_turn(None) == ""


async def test_a_turn_keeps_the_partial_response_after_the_continuation_cap(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)

    async def always_truncated(messages, options=None):
        return {
            "payload": {"usage": {}, "finish_reason": "length", "truncated": True},
            "message": {"content": "trozo ", "tool_calls": []},
        }

    monkeypatch.setattr(app, "call_chat_completions", always_truncated)
    answer = await app.request_assistant_turn(None)
    assert answer == "trozo trozo trozo trozo"
    assert app.messages[-1]["content"] == answer


async def test_streaming_client_rejects_incomplete_tool_calls():
    response = FakeStreamingResponse(
        [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call-1", "function": {"name": "read_file", "arguments": "{"}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ]
    )
    with pytest.raises(Exception, match="invalid arguments"):
        await read_streaming_response(response)


async def test_streaming_client_reassembles_fragmented_tool_calls():
    response = FakeStreamingResponse(
        [
            {"choices": [{"delta": {"reasoning_content": "think"}}]},
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read_", "arguments": '{"pa'}}]}}]},
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": 'th":"a.txt"}'}}]},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]
    )
    deltas: list[str] = []
    reasoning: list[str] = []
    result = await read_streaming_response(
        response, on_text_delta=deltas.append, on_reasoning_delta=reasoning.append
    )
    assert result["message"]["content"] == "Hello"
    assert deltas == ["Hel", "lo"]
    assert reasoning == ["think"]
    call = result["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"path": "a.txt"}


# --------------------------------------------------------------- compaction


def test_compaction_splits_long_transcripts_into_bounded_requests():
    chunks = chunk_summary_transcript(
        [
            {"role": "user", "content": "a" * 1000},
            {"role": "assistant", "content": "b" * 1000},
        ],
        300,
    )
    assert len(chunks) > 2
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert chunks[0].startswith("[user]")


async def test_automatic_compaction_uses_the_model_named_window():
    """The budget must follow the window the model states, not OPENAI_CONTEXT_WINDOW.

    The model name says 8k while the configured window is 32768, so a threshold
    taken from the configured value would let the server truncate the request
    before compaction ever ran.
    """
    app, output = _make_app(80)  # model "bonsai27b-8k:latest"
    assert app.context_window == 32768
    assert app.effective_context_window() == 8192
    app.compaction_reserve_tokens = 4096
    app.compaction_keep_recent_tokens = 4096
    app.messages.extend(
        {"role": "user" if index % 2 else "assistant", "content": "x" * 2000}
        for index in range(40)
    )
    summaries: list[str] = []

    async def fake_summary(*_args, **_kwargs) -> str:
        summaries.append("summarized")
        return "earlier history summary"

    app.generate_compaction_summary = fake_summary  # type: ignore[method-assign]
    await app.compact_automatically_if_needed(None)
    assert summaries == ["summarized"], "automatic compaction did not run for the model-named window"
    assert "Automatic compaction" in output.text
    assert app.compacted_summary == "earlier history summary"


def test_a_huge_tool_result_is_bounded_to_the_window():
    """One command must not be able to fill the whole context window on its own."""
    app, _output = _make_app(80)  # effective window 8192
    assert app.effective_context_window() == 8192
    bounded = app.bound_tool_result("x" * 50_000)
    assert len(bounded) < 50_000
    assert "tool output truncated" in bounded
    assert bounded.startswith("x" * 100)
    # A result that already fits is returned untouched.
    assert app.bound_tool_result("hola") == "hola"


def test_a_truncated_tool_result_keeps_the_tail_for_the_error():
    app, _output = _make_app(80)
    bounded = app.bound_tool_result("START" + "x" * 30_000 + "FAILED: no such file")
    assert bounded.startswith("START")
    assert bounded.rstrip().endswith("FAILED: no such file")
    assert "characters omitted" in bounded


def _archiving_app(tmp_path, columns: int = 80) -> MinAgent:
    """A session whose tool-output archive is isolated in ``tmp_path``."""
    app, _output = _make_app(columns)
    app.tool_archive = ToolArchive(str(tmp_path))
    return app


def test_a_truncated_tool_result_is_archived_instead_of_discarded(tmp_path):
    """The omitted middle must stay reachable, not be thrown away."""
    app = _archiving_app(tmp_path)
    evidence = "ENCONTRADO: el fallo esta en la linea 12345"
    original = "START\n" + ("relleno " * 6000) + f"\n{evidence}\n" + ("cola " * 3000) + "\nFAILED"

    bounded = app.bound_tool_result(original)
    assert len(bounded) < len(original) // 4, "the preview should be a small fraction of the result"
    assert 'id="' in bounded, "the note must name the archive reference the model needs"
    assert evidence not in bounded, "the evidence should be exactly what the preview leaves out"

    reference = bounded.split('id="')[1].split('"')[0]
    # The evidence sits past the preview, so reaching it needs an offset.
    at = compress_for_context(original).index(evidence)
    recalled = app.recall_tool_output({"id": reference, "offset": at, "limit": len(evidence) + 5})
    assert evidence in recalled, "the archived middle was not recoverable"


def test_the_archive_stores_exactly_what_the_model_would_have_seen(tmp_path):
    """Recall is lossless, so the stored text must be the compressed result verbatim."""
    import zlib

    app = _archiving_app(tmp_path)
    original = "\x1b[31mrojo\x1b[0m\n\n\n" + ("dato   con   espacios " * 3000)
    bounded = app.bound_tool_result(original)
    reference = bounded.split('id="')[1].split('"')[0]

    path = os.path.join(str(tmp_path), ".minagent", "tool-outputs", f"{reference}.z")
    with open(path, "rb") as handle:
        stored = zlib.decompress(handle.read()).decode("utf-8")
    assert stored == compress_for_context(original)


def test_recall_reports_the_full_size_and_honours_the_offset(tmp_path):
    app = _archiving_app(tmp_path)
    original = "".join(f"linea {index:06d}\n" for index in range(6000))
    reference = app.bound_tool_result(original).split('id="')[1].split('"')[0]
    total = len(compress_for_context(original))

    # Each line is 13 characters, so 1300 lands exactly on the start of a line.
    recalled = app.recall_tool_output({"id": reference, "offset": 1300, "limit": 26})
    assert f"characters 1300-1326 of {total}" in recalled
    assert recalled.endswith("linea 000100\nlinea 000101\n")


def test_recall_cannot_walk_out_of_the_archive(tmp_path):
    app = _archiving_app(tmp_path)
    for hostile in ("../../../../etc/passwd", "a/b", "", "x" * 200, ".hidden"):
        with pytest.raises(AgentError):
            app.recall_tool_output({"id": hostile})


def test_recall_validates_its_own_arguments(tmp_path):
    app = _archiving_app(tmp_path)
    reference = app.bound_tool_result("x" * 50_000).split('id="')[1].split('"')[0]
    with pytest.raises(AgentError, match="requires the id"):
        app.recall_tool_output({"id": "  "})
    with pytest.raises(AgentError, match="offset"):
        app.recall_tool_output({"id": reference, "offset": -1})
    with pytest.raises(AgentError, match="limit"):
        app.recall_tool_output({"id": reference, "limit": 0})


def test_recall_cannot_refill_the_window_it_freed(tmp_path):
    """One recall is bounded, or the archive would be a slower way to blow up."""
    app = _archiving_app(tmp_path)
    original = "y" * 2_000_000
    reference = app.bound_tool_result(original).split('id="')[1].split('"')[0]
    recalled = app.recall_tool_output({"id": reference, "limit": 10_000_000})
    assert len(recalled) <= MAX_ARCHIVE_RECALL_CHARS + 200


def test_an_unconfigured_session_keeps_the_old_truncation(tmp_path, monkeypatch):
    """No archive root means no archive: truncation still has to work."""
    app, _output = _make_app(80)
    app.tool_archive = ToolArchive("")
    bounded = app.bound_tool_result("x" * 50_000)
    assert "tool output truncated" in bounded
    assert 'id="' not in bounded
    assert not os.path.exists(os.path.join(str(tmp_path), ".minagent"))


def test_an_oversized_result_is_not_worth_archiving(tmp_path):
    app = _archiving_app(tmp_path)
    assert app.tool_archive.store("z" * (MAX_ARCHIVED_CHARS + 1)) is None
    assert app.tool_archive.stored_bytes() == 0


def test_token_usage_is_accounted_per_tool(tmp_path):
    """Each tool result must be charged for what it really cost the window."""
    app = _archiving_app(tmp_path)
    app._record_tool_tokens("read_file", "a" * 4000, "a" * 1000)
    app._record_tool_tokens("read_file", "b" * 4000, "b" * 1000)
    app._record_tool_tokens("run_terminal", "c" * 400, "c" * 400)

    assert app._turn_tool_calls == {"read_file": 2, "run_terminal": 1}
    assert app._turn_tool_tokens["read_file"] > app._turn_tool_tokens["run_terminal"]
    # The part moved out of the window by the archive is tracked separately.
    assert app._turn_archived_tokens > 0


def test_resetting_a_turn_keeps_the_session_totals(tmp_path):
    app = _archiving_app(tmp_path)
    app._record_tool_tokens("read_file", "a" * 4000, "a" * 1000)
    session_total = app._session_tool_tokens["read_file"]
    assert session_total > 0

    app.reset_turn_token_usage()
    assert app._turn_tool_tokens == {}
    assert app._turn_tool_calls == {}
    assert app._turn_archived_tokens == 0
    assert app._session_tool_tokens["read_file"] == session_total, "the session tally was reset by mistake"


def test_the_usage_command_lists_every_tool_it_charged(tmp_path):
    app = _archiving_app(tmp_path)
    app._record_tool_tokens("read_file", "a" * 4000, "a" * 1000)
    app._record_tool_tokens("read_file", "a" * 2000, "a" * 500)
    app._record_tool_tokens("run_terminal", "b" * 9000, "b" * 4000)
    output = FakeOutput(100)
    app._stdout = output

    app.print_token_usage()
    text = output.text
    assert "USO DE TOKENS" in text
    assert "read_file" in text and "run_terminal" in text
    assert "2 llamadas" in text
    assert "1 llamadas" not in text, "una sola llamada no se escribe en plural"
    assert "  1 llamada" in text
    assert "fuera de la ventana" in text, "the archived saving was not reported"


def test_the_usage_command_survives_a_turn_with_no_tools(tmp_path):
    app = _archiving_app(tmp_path)
    output = FakeOutput(100)
    app._stdout = output
    app.print_token_usage()
    assert "sin llamadas a herramientas" in output.text


def test_an_archived_result_reports_what_it_saved(tmp_path):
    """The telemetry has to see the archive doing its job, not just the preview."""
    app = _archiving_app(tmp_path)
    original = "z" * 200_000
    bounded = app.bound_tool_result(original)
    app._record_tool_tokens("run_terminal", original, bounded)

    assert app._turn_tool_tokens["run_terminal"] < app._turn_tool_calls["run_terminal"] * 200_000 / 3
    assert app._turn_archived_tokens > 0


def test_the_system_prompt_prefix_survives_within_the_same_minute():
    """Providers only reuse a byte-identical prefix, so the clock must not tick in it."""
    app, _output = _make_app(100)
    app.terminal_mode = "auto"
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.messages = [{"role": "system", "content": ""}]

    app.refresh_system_prompt()
    primero = app.messages[0]["content"]
    seccion = next(s for s in app._current_system_prompt_sections if s["name"] == "Current time")
    minuto = re.search(r"T(\d{2}:\d{2}):00", seccion["content"])
    assert minuto is not None, "the clock is not truncated to the minute"

    # El segundo solo puede cambiar si cruzamos el minuto, en cuyo caso el
    # prefijo debe cambiar; dentro del mismo minuto tiene que ser idéntico.
    time.sleep(1.1)
    app.refresh_system_prompt()
    siguiente = next(s for s in app._current_system_prompt_sections if s["name"] == "Current time")
    if re.search(r"T(\d{2}:\d{2}):00", siguiente["content"]).group(1) == minuto.group(1):
        assert app.messages[0]["content"] == primero, "the system prompt changed within the same minute"


def test_stable_prompt_sections_come_before_the_volatile_ones():
    """Putting the changing parts last keeps the reusable prefix as long as possible."""
    app, _output = _make_app(100)
    app.terminal_mode = "auto"
    app.agents_context = "Reglas."
    app.compacted_summary = "Resumen."
    app.workspace_snapshot = "src/minagent/app.py"
    app.memory_hint_context = "Memoria."
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.messages = [{"role": "system", "content": ""}]

    app.refresh_system_prompt()
    nombres = [section["name"] for section in app._current_system_prompt_sections]
    base = [section["name"] for section in app._base_system_prompt_sections]
    assert nombres[: len(base)] == base, "the stable sections are not first"
    assert nombres.index("Current time") == len(base), "the clock should open the volatile block"
    for nombre in ("AGENTS.md", "Conversation summary", "Workspace inventory", "Memory hints"):
        assert nombre in nombres


async def test_the_request_cache_replays_an_identical_request_without_calling_the_endpoint(tmp_path, monkeypatch):
    """The retry after a bad turn resends the same request; it must not be paid for twice."""
    app = _skill_app(tmp_path)
    llamadas = []

    async def counting(messages, options=None):
        llamadas.append(messages)
        return {
            "payload": {"usage": {}, "finish_reason": "stop"},
            "message": {"role": "assistant", "content": "respuesta"},
        }

    monkeypatch.setattr(app, "call_chat_completions", counting)
    app.messages = [{"role": "system", "content": "sistema"}]
    primera = await app.request_assistant_turn(None)
    assert primera == "respuesta"
    assert len(llamadas) == 1

    # Mismo prompt byte a byte: se reproduce sin volver a llamar al endpoint.
    app.request_cache.put(app.request_cache.key(app.model, app.tools, app.messages), {
        "payload": {"usage": {}, "finish_reason": "stop"},
        "message": {"role": "assistant", "content": "respuesta"},
    })
    assert await app.request_assistant_turn(None) == "respuesta"
    assert len(llamadas) == 1, "an identical request reached the endpoint again"
    assert app.request_cache.stats()["hits"] == 1


def test_the_request_cache_keeps_the_flags_the_caller_branches_on():
    """A replay that dropped 'truncated' would take a different path than the original."""
    cache = RequestCache()
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    mensajes = [{"role": "user", "content": "hola"}]
    clave = cache.key("m", tools, mensajes)
    cache.put(
        clave,
        {
            "payload": {"usage": {"prompt_tokens": 7}, "finish_reason": "length", "truncated": True},
            "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function"}]},
        },
    )
    golpe = cache.get(clave)
    assert golpe is not None
    assert golpe["payload"]["truncated"] is True
    assert golpe["payload"]["finish_reason"] == "length"
    assert golpe["payload"]["usage"] == {"prompt_tokens": 7}
    assert golpe["message"]["tool_calls"] == [{"id": "a", "type": "function"}]


def test_the_request_cache_separates_different_requests():
    cache = RequestCache()
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    base = [{"role": "user", "content": "hola"}]
    clave = cache.key("m", tools, base)
    cache.put(clave, {"payload": {"finish_reason": "stop"}, "message": {"content": "a"}})

    # Cambiar el modelo, las herramientas o el mensaje debe invalidar la entrada.
    assert cache.get(cache.key("otro", tools, base)) is None
    assert cache.get(cache.key("m", tools + [{"type": "function"}], base)) is None
    assert cache.get(cache.key("m", tools, [{"role": "user", "content": "adios"}])) is None
    # El orden de las claves no debe producir un falso fallo.
    assert cache.get(cache.key("m", [{"function": {"name": "read_file"}, "type": "function"}], base)) is not None


def test_the_request_cache_never_replays_a_stopped_response():
    cache = RequestCache()
    clave = cache.key("m", [], [{"role": "user", "content": "hola"}])
    cache.put(clave, {"payload": {"finish_reason": "aborted"}, "message": {"content": "a medias"}})
    assert cache.get(clave) is None
    assert len(cache) == 0

    cache.put(
        clave,
        {"payload": {"finish_reason": "stop"}, "message": {"content": "x", "interrupted": True}},
    )
    assert cache.get(clave) is None, "an interrupted response must be retried, not replayed"


def test_the_request_cache_is_bounded():
    cache = RequestCache(max_entries=2)
    for indice in range(5):
        clave = cache.key("m", [], [{"role": "user", "content": str(indice)}])
        cache.put(clave, {"payload": {"finish_reason": "stop"}, "message": {"content": str(indice)}})
    assert len(cache) == 2
    # Lo más viejo se descartó, lo más reciente sigue ahí.
    assert cache.get(cache.key("m", [], [{"role": "user", "content": "0"}])) is None
    assert cache.get(cache.key("m", [], [{"role": "user", "content": "4"}])) is not None


def test_clearing_old_tool_results_frees_context_without_losing_them(tmp_path):
    """Tool result clearing: the transcript shrinks, the evidence stays retrievable."""
    app = _archiving_app(tmp_path)
    app.tool_result_keep = 1
    app.messages = [{"role": "system", "content": "sistema"}]
    for indice in range(4):
        app.messages.append({"role": "user", "content": f"peticion {indice}"})
        app.messages.append(
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{indice}", "type": "function"}]}
        )
        app.messages.append({"role": "tool", "tool_call_id": f"c{indice}", "content": "dato " * 900 + f"#{indice}"})

    antes = sum(estimate_text_tokens(m.get("content") or "") for m in app.messages)
    liberado = app.clear_old_tool_results()
    despues = sum(estimate_text_tokens(m.get("content") or "") for m in app.messages)

    assert liberado > 0, "nothing was freed"
    assert despues < antes / 2, "the transcript barely shrank"
    # The last result stays verbatim, the older ones become retrievable stubs.
    assert "#3" in app.messages[-1]["content"]
    resultados = [m["content"] for m in app.messages if m.get("role") == "tool"]
    for contenido in resultados[:-1]:
        assert contenido.startswith("[tool result cleared"), "an old result was not stubbed"
        assert 'id="' in contenido, "the stub does not name a reference to recall from"

    # Y lo importante: el dato sigue estando disponible.
    referencia = resultados[0].split('id="')[1].split('"')[0]
    assert "dato" in app.recall_tool_output({"id": referencia, "limit": 20000})


def test_clearing_keeps_the_most_recent_tool_results_verbatim(tmp_path):
    app = _archiving_app(tmp_path)
    app.tool_result_keep = 2
    app.messages = [{"role": "system", "content": "sistema"}]
    for indice in range(5):
        app.messages.append({"role": "tool", "tool_call_id": str(indice), "content": f"resultado {indice} " + "x" * 500})

    app.clear_old_tool_results()
    contenidos = [m["content"] for m in app.messages if m.get("role") == "tool"]
    assert "resultado 4" in contenidos[-1]
    assert "resultado 3" in contenidos[-2]
    assert contenidos[0].startswith("[tool result cleared")
    assert "resultado 2" not in contenidos[0]


def test_clearing_is_idempotent_and_does_not_archive_twice(tmp_path):
    app = _archiving_app(tmp_path)
    app.tool_result_keep = 0
    app.messages = [{"role": "system", "content": "sistema"}]
    app.messages.append({"role": "tool", "tool_call_id": "c", "content": "contenido " * 500})

    app.clear_old_tool_results()
    primer_stub = app.messages[1]["content"]
    liberado_segunda = app.clear_old_tool_results()

    assert liberado_segunda == 0, "a cleared result was archived a second time"
    assert app.messages[1]["content"] == primer_stub
    referencia = primer_stub.split('id="')[1].split('"')[0]
    assert "contenido" in app.recall_tool_output({"id": referencia, "limit": 20000})


def test_a_truncated_result_keeps_its_existing_reference_when_cleared(tmp_path):
    """A result that was already truncated must not be archived a second time."""
    app = _archiving_app(tmp_path)
    app.tool_result_keep = 0
    app.messages = [{"role": "system", "content": "sistema"}]
    acotado = app.bound_tool_result("z" * 60_000)
    app.messages.append({"role": "tool", "tool_call_id": "c", "content": acotado})
    original = acotado.split('id="')[1].split('"')[0]

    app.clear_old_tool_results()
    stub = app.messages[1]["content"]
    assert f'id="{original}"' in stub, "the existing archive reference was replaced"
    assert "z" * 100 in app.recall_tool_output({"id": original, "limit": 20000})


def test_nothing_is_cleared_without_an_archive(tmp_path):
    """Without somewhere to put the text, clearing it would lose it for good."""
    app = _archiving_app(tmp_path)
    app.tool_archive = ToolArchive("")
    app.tool_result_keep = 0
    app.messages = [{"role": "system", "content": "sistema"}]
    app.messages.append({"role": "tool", "tool_call_id": "c", "content": "importante " * 500})

    assert app.clear_old_tool_results() == 0
    assert "importante" in app.messages[1]["content"]


def test_markup_is_stripped_but_code_is_not(tmp_path):
    """Angle brackets are everywhere in code; they are only markup in a document."""
    html = "<html><head><style>body{color:red}</style></head><body><p>Hola</p><p>que tal</p></body></html>"
    limpio = compress_for_context(html)
    assert "<" not in limpio and "Hola" in limpio and "que tal" in limpio

    codigo = 'if x < len(y) and z > 3:\n    return {"a": 1}\n'
    # La indentacion ya se colapsa desde antes, pero los operadores de
    # comparacion son lo que el detector de marcado no debe comerse.
    compacto = compress_for_context(codigo)
    assert "x < len(y) and z > 3" in compacto, "code was mistaken for markup"
    assert 'return {"a": 1}' in compacto

    diff = "--- a/file\n+++ b/file\n@@ -1 +1 @@\n-if a<b:\n+if a > b:\n"
    assert compress_for_context(diff) == diff, "a diff was mistaken for markup"


def test_repeated_lines_are_marked_instead_of_repeated():
    linea = "PASSED tests/test_core.py::test_algo"
    salida = "\n".join([linea] * 40)
    compacto = compress_for_context(salida)
    assert "repeated 40 times" in compacto
    assert compacto.count(linea) == 1, "the line was still repeated 40 times"
    # Una sola vez no se marca: no es una repetición.
    assert compress_for_context(linea) == linea


def test_base64_is_dropped_but_a_run_of_one_character_is_not():
    import base64 as base64_module

    carga = base64_module.b64encode(b"contenido binario " * 200).decode()
    assert "base64 blob omitted" in compress_for_context(f"clave: {carga}")

    # Un solo caracter repetido es relleno, no base64.
    assert compress_for_context("x" * 50_000) == "x" * 50_000


def test_tool_result_keep_is_configurable(tmp_path):
    assert _configuration(tmp_path).tool_result_keep == 3
    assert _configuration(tmp_path, TOOL_RESULT_KEEP="8").tool_result_keep == 8
    with pytest.raises(AgentError, match="TOOL_RESULT_KEEP must be a positive integer"):
        _configuration(tmp_path, TOOL_RESULT_KEEP="0")


def test_the_lossy_filters_never_touch_source_code():
    """An audit found the markup filter eating this very file's regexes.

    Anything that reads as source, a diff, a stylesheet or structured config has
    to come back with every token intact, because a stripped token in code is a
    bug the model can neither see nor fix.
    """
    for nombre, texto in (
        ("python", 'def _TAGS = re.compile(r"<[^>]+>")\n    return _TAGS.sub(" ", text)\n'),
        ("xml", '<?xml version="1.0"?>\n<config><item id="1">clave</item></config>'),
        ("diff", "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-if a<b:\n+if a > b\n"),
        ("css", ".a { color: red; }\n.b { color: blue; }\n"),
        ("json", '{"claves": [1, 2, 3], "mapa": {"a": {"b": "c"}}}'),
        ("sql", "SELECT * FROM t WHERE a < 5 AND b > 2;"),
        ("compilacion", "#include <stdio.h>\nint main(void) { return 0; }\n"),
    ):
        salida = compress_for_context(texto)
        assert [linea.strip() for linea in salida.split("\n") if linea.strip()] == [
            linea.strip() for linea in texto.split("\n") if linea.strip()
        ], f"the lossy filters altered {nombre}"


def test_a_run_of_spaces_inside_a_string_literal_is_not_collapsed():
    """A padding string is data: collapsing it silently changes the program."""
    codigo = 'self.ui_print_wrapped((("     ", "pale", False), (line, "pale", False)))'
    assert '"     "' in compress_for_context(codigo)

    # La misma compressing en salida de terminal, que si se colapsa.
    assert compress_for_context("columna1          columna2") == "columna1 columna2"


def test_the_markup_filter_keeps_a_tag_mentioned_in_a_sentence():
    """The attribute is the whole point of mentioning the tag."""
    frase = "usa <div class='x'> con cuidado"
    assert "<div class='x'>" in compress_for_context(frase)


def test_the_markup_filter_still_strips_a_real_document():
    documento = (
        "<html><head><style>body{color:red}</style></head><body>"
        + "<p>Hola</p><p>que tal</p><p>bien</p><p>gracias</p>"
        + "</body></html>"
    )
    limpio = compress_for_context(documento)
    assert "<" not in limpio
    assert "Hola" in limpio and "que tal" in limpio


def test_repeated_lines_are_never_collapsed_in_code():
    """Two identical assertions in a test are the test, not noise."""
    test = "def test_x():\n    assert calcula(1) == 1\n    assert calcula(1) == 1\n"
    assert "repeated" not in compress_for_context(test)


def test_repeated_lines_are_collapsed_in_command_output():
    salida = "\n".join(["PASSED tests/test_core.py::test_algo"] * 12)
    compacto = compress_for_context(salida)
    assert "repeated 12 times" in compacto
    assert compacto.count("PASSED") == 1


def test_reading_a_source_file_of_this_project_keeps_every_line():
    """The end-to-end version of the audit: this repo's own modules survive."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "minagent"
    modulos = sorted(root.glob("*.py"))
    assert modulos, "no source files found to check"
    for modulo in modulos:
        fuente = modulo.read_text(encoding="utf-8")
        salida = compress_for_context(fuente)
        original = [linea.strip() for linea in fuente.split("\n") if linea.strip()]
        resultante = {linea.strip() for linea in salida.split("\n") if linea.strip()}
        ausentes = [linea for linea in original if linea not in resultante]
        assert not ausentes, f"{modulo.name} lost {len(ausentes)} lines, first: {ausentes[0]!r}"


def _respuesta_con_llamadas(*nombres):
    """A model response that asks for several tools at once."""
    return {
        "payload": {"usage": {}, "finish_reason": "tool_calls"},
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"c{indice}",
                    "type": "function",
                    "function": {"name": nombre, "arguments": json.dumps({"path": "a.py", "command": "date"})},
                }
                for indice, nombre in enumerate(nombres)
            ],
        },
    }


def _una_vez_tras_las_herramientas(*nombres):
    """Un falso endpoint que pide herramientas una vez y luego responde con texto."""

    async def falso(messages, options=None):
        if not falso.pendientes:
            return {
                "payload": {"usage": {}, "finish_reason": "stop"},
                "message": {"role": "assistant", "content": "listo"},
            }
        falso.pendientes = False
        return _respuesta_con_llamadas(*nombres)

    falso.pendientes = True
    return falso


async def test_consecutive_reads_run_together(tmp_path, monkeypatch):
    """Two reads in one response should overlap, not queue behind each other."""
    app = _skill_app(tmp_path)
    en_curso = 0
    maximo_simultaneo = 0

    async def lectura(args, image_enabled=False):
        nonlocal en_curso, maximo_simultaneo
        en_curso += 1
        maximo_simultaneo = max(maximo_simultaneo, en_curso)
        try:
            await asyncio.sleep(0.05)
            return "contenido"
        finally:
            en_curso -= 1

    monkeypatch.setattr(app, "call_chat_completions", _una_vez_tras_las_herramientas("read_file", "read_file"))
    monkeypatch.setattr(app, "execute_tool", lectura)
    await app.request_assistant_turn(None)

    assert maximo_simultaneo == 2, "the two reads did not overlap"


async def test_a_write_is_never_batched_with_a_read(tmp_path, monkeypatch):
    """Order matters: nothing that writes may start before an earlier read finishes."""
    app = _skill_app(tmp_path)
    orden: list[str] = []

    async def dos_lecturas_y_una_escritura(messages, options=None):
        if not dos_lecturas_y_una_escritura.pendientes:
            return {
                "payload": {"usage": {}, "finish_reason": "stop"},
                "message": {"role": "assistant", "content": "listo"},
            }
        dos_lecturas_y_una_escritura.pendientes = False
        return _respuesta_con_llamadas("read_file", "read_file", "write_file", "read_file")

    dos_lecturas_y_una_escritura.pendientes = True

    async def ejecucion(name, args):
        if name == "write_file":
            orden.append("write")
            return "escrito"
        orden.append(f"read:{len([o for o in orden if o.startswith('read')])}")
        await asyncio.sleep(0.02)
        return "contenido"

    monkeypatch.setattr(app, "call_chat_completions", dos_lecturas_y_una_escritura)
    monkeypatch.setattr(app, "execute_tool", ejecucion)
    await app.request_assistant_turn(None)

    assert orden.index("write") == 2, f"the write ran before the reads finished: {orden}"
    # The read after the write stayed in the serial tail.
    assert orden.count("read:2") == 1


async def test_one_read_alone_is_not_worth_batching(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)
    en_curso = 0
    maximo = 0

    async def lectura(args, image_enabled=False):
        nonlocal en_curso, maximo
        en_curso += 1
        maximo = max(maximo, en_curso)
        try:
            await asyncio.sleep(0.02)
            return "contenido"
        finally:
            en_curso -= 1

    monkeypatch.setattr(app, "call_chat_completions", _una_vez_tras_las_herramientas("read_file"))
    monkeypatch.setattr(app, "execute_tool", lectura)
    await app.request_assistant_turn(None)
    assert maximo == 1


async def test_parallel_reads_can_be_turned_off(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)
    app.parallel_tools = False
    en_curso = 0
    maximo = 0

    async def lectura(args, image_enabled=False):
        nonlocal en_curso, maximo
        en_curso += 1
        maximo = max(maximo, en_curso)
        try:
            await asyncio.sleep(0.03)
            return "contenido"
        finally:
            en_curso -= 1

    monkeypatch.setattr(app, "call_chat_completions", _una_vez_tras_las_herramientas("read_file", "read_file"))
    monkeypatch.setattr(app, "execute_tool", lectura)
    await app.request_assistant_turn(None)
    assert maximo == 1, "reads overlapped even with PARALLEL_TOOLS=off"


async def test_a_failing_read_does_not_lose_the_others(tmp_path, monkeypatch):
    """One broken read must not take down the results the rest already produced."""
    app = _skill_app(tmp_path)

    async def ejecucion(name, args):
        if not hasattr(ejecucion, "contado"):
            ejecucion.contado = 0
        ejecucion.contado += 1
        if ejecucion.contado == 1:
            raise AgentError("no such file")
        return "contenido bueno"

    monkeypatch.setattr(app, "call_chat_completions", _una_vez_tras_las_herramientas("read_file", "read_file"))
    monkeypatch.setattr(app, "execute_tool", ejecucion)
    await app.request_assistant_turn(None)

    contenidos = [m.get("content") or "" for m in app.messages if m.get("role") == "tool"]
    assert any("no such file" in c for c in contenidos), "the failure was not reported"
    assert any("contenido bueno" in c for c in contenidos), "the other read's result was lost"


def test_only_read_only_tools_are_eligible_for_batching():
    """The allowlist is the safety property; keep it explicit."""
    from minagent.app import _CONCURRENT_READ_TOOLS

    for nombre in ("read_file", "list_directory", "recall_tool_output", "web_search", "web_fetch"):
        assert nombre in _CONCURRENT_READ_TOOLS
    for nombre in (
        "edit_file",
        "write_file",
        "create_directory",
        "delete_file",
        "delete_directory",
        "run_terminal",
        "remember",
        "record_outcome",
        "write_skill",
        "write_mcp_server",
    ):
        assert nombre not in _CONCURRENT_READ_TOOLS, f"{nombre} can change state and must not be batched"


def test_parallel_tools_defaults_to_on_and_is_configurable(tmp_path):
    assert _configuration(tmp_path).parallel_tools is True
    assert _configuration(tmp_path, PARALLEL_TOOLS="off").parallel_tools is False
    with pytest.raises(AgentError, match="PARALLEL_TOOLS must be on or off"):
        _configuration(tmp_path, PARALLEL_TOOLS="si")


def test_the_capability_note_points_at_web_search_when_it_is_available():
    """Un modelo que dice 'no tengo noticias' necesita que le digan qué herramienta usar."""
    app, _output = _make_app(100)
    app.web_search_enabled = False
    assert "web_search searches the web" not in app.missing_capability_note()

    app.web_search_enabled = True
    note = app.missing_capability_note()
    assert "web_search searches the web" in note
    assert "call web_search first" in note


def test_a_reasoning_only_response_is_answered_instead_of_failing(tmp_path):
    """Un modelo que solo razona no puede parecer un endpoint que no responde."""
    from minagent.openai import read_streaming_response

    class FakeStream:
        def __init__(self, frames):
            self._frames = frames

        async def aiter_bytes(self):
            for frame in self._frames:
                yield frame

    def frame(payload):
        return f"data: {json.dumps(payload)}\n\n".encode()

    stream = FakeStream(
        [
            frame({"choices": [{"delta": {"reasoning_content": "estoy pensando "}}]}),
            frame({"choices": [{"delta": {"reasoning_content": "en la respuesta"}}]}),
            frame({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            b"data: [DONE]\n\n",
        ]
    )
    result = asyncio.run(read_streaming_response(stream))
    assert result["message"]["content"] is None
    assert result["message"]["reasoning_content"] == "estoy pensando en la respuesta"


def test_the_recall_tool_is_exposed_to_the_model():
    app, _output = _make_app(80)
    names = {tool["function"]["name"] for tool in app.tools}
    assert "recall_tool_output" in names
    schema = next(tool for tool in app.tools if tool["function"]["name"] == "recall_tool_output")
    assert schema["function"]["parameters"]["required"] == ["id"]


def test_the_archive_is_pruned_once_it_outgrows_its_budget(tmp_path, monkeypatch):
    """The archive must not fill the disk; the oldest entries go first."""
    from minagent import tool_archive as archive_module

    budget = 4000
    monkeypatch.setattr(archive_module, "MAX_ARCHIVE_BYTES", budget)
    archive = ToolArchive(str(tmp_path))
    # Random text does not compress, so the budget is genuinely exceeded.
    for _ in range(20):
        assert archive.store(os.urandom(600).hex()) is not None

    entries = archive._entries()
    assert len(entries) < 20, "pruning removed nothing"
    assert archive.stored_bytes() <= budget, "pruning left the archive over budget"


def test_pruning_keeps_the_newest_entries(tmp_path, monkeypatch):
    """The reference still in the transcript is the one worth keeping."""
    from minagent import tool_archive as archive_module

    monkeypatch.setattr(archive_module, "MAX_ARCHIVE_BYTES", 2000)
    archive = ToolArchive(str(tmp_path))
    newest = None
    for _ in range(20):
        newest = archive.store(os.urandom(600).hex())
    assert newest is not None
    assert "characters 0-" in archive.read(newest)


def test_a_corrupt_archive_reports_an_error_instead_of_crashing(tmp_path):
    archive = ToolArchive(str(tmp_path))
    reference = archive.store("contenido")
    assert reference is not None
    path = os.path.join(str(tmp_path), ".minagent", "tool-outputs", f"{reference}.z")
    with open(path, "wb") as handle:
        handle.write(b"esto no es zlib")

    with pytest.raises(AgentError, match="corrupt"):
        archive.read(reference)


def test_a_missing_reference_points_back_at_the_command(tmp_path):
    archive = ToolArchive(str(tmp_path))
    with pytest.raises(AgentError, match="rerun the command"):
        archive.read("0badc0de0bad")


def test_a_failed_write_does_not_break_the_turn(tmp_path):
    """A read-only or full disk must fall back to truncation, not raise."""
    archive = ToolArchive(str(tmp_path))
    blocked = os.path.join(str(tmp_path), "blocked")
    os.makedirs(blocked)
    os.chmod(blocked, 0o500)
    try:
        archive = ToolArchive(os.path.join(blocked, "sub"))
        assert archive.store("no se puede guardar") is None
    finally:
        os.chmod(blocked, 0o700)


def test_context_compression_strips_ansi_and_collapses_whitespace():
    raw = "\x1b[31mred\x1b[0m\n\n\n\nline   with    spaces  \n"
    out = compress_for_context(raw)
    assert "\x1b" not in out
    assert "line with spaces" in out
    assert "\n\n\n" not in out


def test_context_compression_minifies_a_json_payload():
    raw = json.dumps({"a": 1, "items": list(range(60))}, indent=4)
    assert len(raw) >= 200
    out = compress_for_context(raw)
    assert out == '{"a":1,"items":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]}'


def test_context_compression_leaves_plain_text_intact():
    assert compress_for_context("hola mundo") == "hola mundo"


async def test_automatic_compaction_skips_a_context_that_fits():
    app, output = _make_app(80)
    app.compaction_reserve_tokens = 4096
    app.messages.append({"role": "user", "content": "hola"})
    await app.compact_automatically_if_needed(None)
    assert "Automatic compaction" not in output.text


# ----------------------------------------------------------------- secrets


def test_approval_preview_masks_authorization_headers():
    preview = approval_preview({"headers": {"Authorization": "Bearer sensitive", "Cookie": "session=sensitive"}})
    assert "sensitive" not in preview
    assert "REDACTED" in preview


# ---------------------------------------------------------------- terminal


class _NoopTerminal:
    async def question(self, prompt: str) -> str:
        return "n"


async def test_terminal_command_requires_approval_in_ask_mode(tmp_path):
    result = await run_terminal_command(
        {"command": "echo should-not-run"},
        terminal_mode="ask",
        terminal_command_shell="/bin/sh",
        root_directory=str(tmp_path),
        interactive_terminal=_NoopTerminal(),
        print=lambda value: None,
        ui_print=lambda value: None,
        ui_text=lambda value, *args: value,
    )
    assert "Permission denied" in result


# ------------------------------------------------------------------ editor


def test_editor_keeps_pasted_newlines_and_ctrl_j_inside_the_current_input():
    editor = FakeEditor("ab", 1)
    state = PasteState()
    assert handle_pasted_input(Key(name="paste-start"), "", editor, state) is True
    pasted_return = Key(name="return")
    assert handle_pasted_input(pasted_return, "\r", editor, state) is True
    assert editor.line == "a\nb"
    assert pasted_return.name == "j"
    handle_pasted_input(Key(name="paste-end"), "", editor, state)
    ctrl_j = Key(name="j", ctrl=True)
    assert handle_control_j_input(ctrl_j, "\n", editor) is True
    assert editor.line == "a\n\nb"


def test_slash_and_file_autocomplete_panels_render_without_closing_the_app():
    commands = [
        {"name": "compact", "description": "Compact conversation history manually"},
        {"name": "exit", "description": "Exit MinAgent"},
    ]
    command_state = build_autocomplete_state("/", 1, [], commands)
    command_lines = format_autocomplete_panel(command_state, columns=80)
    assert len(command_lines) == AUTOCOMPLETE_PANEL_ROWS
    assert any("/compact" in line for line in command_lines)
    assert any("/exit" in line for line in command_lines)
    file_state = build_autocomplete_state("@", 1, ["src/very-long-\U0001F600-filename.py"], commands)
    file_lines = format_autocomplete_panel(file_state, columns=20)
    assert len(file_lines) == AUTOCOMPLETE_PANEL_ROWS
    assert terminal_text_width(file_lines[1]) <= 20
    assert truncate_terminal_text("\U0001F600abc", 3) == "\U0001F600…"


@pytest.mark.parametrize(
    "input_text, workspace_files, expected, selected_file",
    [
        ("/", [], "/exit", None),
        ("@", ["README.md", "src/file.py"], "src/file.py", "src/file.py"),
    ],
)
def test_arrow_keys_select_and_enter_replaces_the_text_in_place(
    input_text, workspace_files, expected, selected_file
):
    """Drives the real editor so key decoding, editing, and completion agree."""
    output = FakeOutput(80)
    editor = LineEditor(output, sys.stdin)
    editor.line = input_text
    editor.cursor = len(input_text)
    commands = [{"name": "compact", "description": "Compact"}, {"name": "exit", "description": "Exit"}]
    state = build_autocomplete_state(editor.line, editor.cursor, workspace_files, commands)

    def key_for(raw: str) -> Key:
        keys = editor._parse(raw)
        assert len(keys) == 1, f"{raw!r} decoded to {[(key.character, key.name) for key in keys]}"
        return keys[0].as_editor_key()

    assert handle_autocomplete_keypress(state, key_for("\x1b[B"), editor)["kind"] == "move"
    assert state["selected_index"] == 1
    assert handle_autocomplete_keypress(state, key_for("\x1b[A"), editor)["kind"] == "move"
    assert state["selected_index"] == 0
    assert handle_autocomplete_keypress(state, key_for("\x1b[B"), editor)["kind"] == "move"

    action = handle_autocomplete_keypress(state, key_for("\r"), editor)
    assert action["kind"] == "complete"
    assert action["selected_file"] == selected_file
    assert editor.line == expected


@pytest.mark.parametrize(
    "raw, name",
    [
        ("\x1b[A", "up"),
        ("\x1b[B", "down"),
        ("\x1b[C", "right"),
        ("\x1b[D", "left"),
        ("\x1bOA", "up"),
        ("\x1b[1~", "home"),
        ("\x1b[4~", "end"),
        ("\x1b[3~", "delete"),
        ("\x1b[5~", "pageup"),
        ("\x1b[6~", "pagedown"),
        ("\x1b[200~", "paste-start"),
        ("\x1b[201~", "paste-end"),
    ],
)
def test_escape_sequences_decode_to_exactly_one_key(raw, name):
    """A sequence must not leave its final byte behind to be typed as text."""
    keys = LineEditor()._parse(raw)
    assert [(key.character, key.name) for key in keys] == [("", name)]


def test_escape_sequences_mixed_with_text_keep_their_order():
    keys = LineEditor()._parse("/\x1b[B!\x1b[C")
    assert [(key.character, key.name) for key in keys] == [("/", "/"), ("", "down"), ("!", "!"), ("", "right")]


def test_a_partial_escape_sequence_is_never_dispatched_as_text():
    assert LineEditor()._parse("\x1b[") == []


async def test_a_lone_escape_reaches_the_client_after_a_short_grace_period():
    """Esc is one byte and has no trailing byte to complete a sequence."""
    master, slave = pty.openpty()
    tty.setraw(slave)  # The editor takes over the terminal too; a cooked read would wait for a line.
    editor = LineEditor(_DescriptorStream(slave), _DescriptorStream(slave))
    seen: list[str] = []
    editor.prepend_keypress(lambda character, key: seen.append(key.name))
    try:
        os.write(master, b"\x1b")
        editor._read_available()
        assert seen == [], "an unresolved prefix must not fire immediately"
        await asyncio.sleep(0.2)
        assert seen == ["escape"]
    finally:
        editor.close()
        os.close(master)
        os.close(slave)


async def test_an_escape_sequence_split_across_reads_is_not_lost():
    master, slave = pty.openpty()
    tty.setraw(slave)
    editor = LineEditor(_DescriptorStream(slave), _DescriptorStream(slave))
    seen: list[str] = []
    editor.prepend_keypress(lambda character, key: seen.append(key.name))
    try:
        os.write(master, b"\x1b")
        editor._read_available()
        os.write(master, b"[A")
        editor._read_available()
        assert seen == ["up"]
        await asyncio.sleep(0.2)
        assert seen == ["up"], "a completed sequence must not also fire an Escape"
    finally:
        editor.close()
        os.close(master)
        os.close(slave)


async def test_submitting_a_line_clears_the_buffer_for_the_next_prompt():
    """The submitted text must not reappear, nor collect what is typed later."""
    output = FakeOutput(80)
    editor = LineEditor(output, sys.stdin)
    editor.line = "/context"
    editor.cursor = len("/context")
    editor._submit = asyncio.get_running_loop().create_future()
    pending = editor._submit
    editor._dispatch(editor._parse("\r")[0])
    assert await pending == "/context"
    assert editor.line == "" and editor.cursor == 0

    # A key pressed while the model is answering is ignored, not glued to the old line.
    painted = len(output.text)
    editor._dispatch(editor._parse("x")[0])
    assert editor.line == ""
    assert len(output.text) == painted, "a key sent without a pending prompt repainted the line"


async def test_history_records_submissions_and_skips_blank_lines():
    """Only real inputs are recalled; blank turns are not history."""
    output = FakeOutput(80)
    editor = LineEditor(output, sys.stdin)
    for value in ("first", "second", "second", "  ", ""):
        editor.line = value
        editor.cursor = len(value)
        editor._submit = asyncio.get_running_loop().create_future()
        editor._dispatch(editor._parse("\r")[0])
    assert editor.history == ["first", "second"]


def test_up_arrow_recalls_history_and_down_restores_the_draft():
    output = FakeOutput(80)
    editor = LineEditor(output, sys.stdin)
    editor.history = ["/context", "hello"]
    editor.line = "draft"
    editor.cursor = len("draft")

    editor._apply_default(editor._parse("\x1b[A")[0])
    assert editor.line == "hello" and editor.cursor == len("hello")
    editor._apply_default(editor._parse("\x1b[A")[0])
    assert editor.line == "/context"
    # At the oldest entry, another up is a no-op rather than wrapping around.
    editor._apply_default(editor._parse("\x1b[A")[0])
    assert editor.line == "/context"
    editor._apply_default(editor._parse("\x1b[B")[0])
    assert editor.line == "hello"
    editor._apply_default(editor._parse("\x1b[B")[0])
    assert editor.line == "draft"


def test_up_arrow_moves_between_buffer_lines_before_recalling_history():
    """Inside a multiline buffer the arrows edit the text, not the history."""
    output = FakeOutput(80)
    editor = LineEditor(output, sys.stdin)
    editor.history = ["old"]
    editor.line = "a\nb"
    editor.cursor = len("a\nb")
    editor._apply_default(editor._parse("\x1b[A")[0])
    assert editor.line == "a\nb" and editor.cursor == len("a")
    editor._apply_default(editor._parse("\x1b[A")[0])
    assert editor.line == "old"


def test_prompt_mode_keeps_the_carriage_return_the_ui_needs():
    """Raw input must leave output post-processing alone.

    ``tty.setraw`` clears ``OPOST``, which stops the terminal turning a bare
    ``\n`` into a carriage return plus line feed. Every writer in the app ends
    its lines with ``\n``, so without that the cursor stays where the previous
    line ended and the whole UI stairs to the right.
    """
    master, slave = pty.openpty()
    editor = LineEditor(_DescriptorStream(slave), _DescriptorStream(slave))
    try:
        editor.start()
        assert editor._saved_termios is not None, "the editor did not take over the terminal"
        os.write(slave, b"| panel line |\n")
        assert os.read(master, 4096) == b"| panel line |\r\n"
    finally:
        editor.close()
        os.close(master)
        os.close(slave)


def test_autocomplete_replaces_a_whole_token_when_the_cursor_is_in_its_middle():
    editor = FakeEditor("/compXYZ", 5)
    state = build_autocomplete_state(editor.line, editor.cursor, [], [{"name": "compact", "description": "Compact"}])
    action = handle_autocomplete_keypress(state, Key(name="enter"), editor)
    assert action["kind"] == "complete"
    assert editor.line == "/compact"


# ------------------------------------------------------------------ layout


class _DescriptorStream:
    """A stream that reports a terminal descriptor but no width of its own."""

    def __init__(self, descriptor: Any = 1) -> None:
        self._descriptor = descriptor

    def fileno(self) -> int:
        if isinstance(self._descriptor, BaseException):
            raise self._descriptor
        return self._descriptor


def test_terminal_columns_prefers_an_explicit_stream_width():
    assert terminal_columns(FakeOutput(42)) == 42


def test_terminal_columns_reads_the_attached_terminal(monkeypatch):
    monkeypatch.setattr(os, "get_terminal_size", lambda fd=1: os.terminal_size((123, 40)))
    assert terminal_columns(_DescriptorStream()) == 123


def test_terminal_columns_falls_back_when_there_is_no_terminal(monkeypatch):
    def no_terminal(fd: int = 1) -> Any:
        raise OSError("not a terminal")

    monkeypatch.setattr(os, "get_terminal_size", no_terminal)
    assert terminal_columns(_DescriptorStream(OSError("detached"))) == 80
    assert terminal_columns(_DescriptorStream(OSError("detached")), default=100) == 100


def _plain(lines: list[list[tuple[str, str, bool]]]) -> list[str]:
    return ["".join(part for part, _, _ in line) for line in lines]


def test_styled_segments_wrap_at_a_space_and_keep_each_style():
    lines = wrap_styled_segments(
        (
            ("/", "magenta", True),
            (" commands  ", "muted", False),
            ("@", "cyan", True),
            (" files", "muted", False),
        ),
        14,
    )
    assert _plain(lines) == ["/ commands  @", "files"]
    assert lines[0][-1] == ("@", "cyan", True)


def test_styled_segments_hard_break_a_word_wider_than_the_terminal():
    lines = wrap_styled_segments(
        (("◆", "magenta", False), (" verylongmodelname  ", "pale", True), ("Context 42%", "cyan", False)),
        12,
    )
    assert _plain(lines) == ["◆", "verylongmode", "lname", "Context 42%"]


def test_styled_segments_never_exceed_the_terminal_width():
    hints = (
        ("/", "magenta", True),
        (" commands  ", "muted", False),
        ("@", "cyan", True),
        (" files  ", "muted", False),
        ("Ctrl+J", "pale", True),
        (" new line  ", "muted", False),
        ("Esc", "pale", True),
        (" stop", "muted", False),
    )
    for width in range(4, 61):
        lines = wrap_styled_segments(hints, width)
        assert lines
        for text in _plain(lines):
            assert terminal_text_width(text) <= width, (width, text)
            assert text == text.strip(), (width, text)
    assert _plain(wrap_styled_segments(hints, 60)) == ["/ commands  @ files  Ctrl+J new line  Esc stop"]


def test_styled_segments_move_a_whole_block_to_the_next_line():
    status = (
        ("◆", "magenta", False),
        (" bonsai27b-8k:latest  ", "pale", True),
        ("Context ~711 / 32,768 (2.2%)", "cyan", False),
    )
    assert _plain(wrap_styled_segments(status, 60)) == [
        "◆ bonsai27b-8k:latest  Context ~711 / 32,768 (2.2%)"
    ]
    assert _plain(wrap_styled_segments(status, 40)) == [
        "◆ bonsai27b-8k:latest",
        "Context ~711 / 32,768 (2.2%)",
    ]


class _ColorOutput(FakeOutput):
    """A recorder that claims to be a terminal, so MinAgent colors its output."""

    def isatty(self) -> bool:
        return True


_SGR = re.compile(r"\x1b\[[0-9;]*m")

_LONG_NOTE = "the worksheet could not be read because the path is outside the workspace and it is quite long"
_TABLE_MARKDOWN = (
    "| Feature | Notes |\n"
    "| --- | :---: |\n"
    f"| Tables | {_LONG_NOTE} · 日本語のテキストは全角 |\n"
    "| Emoji | ✅ 🎯 👨‍👩‍👧 |\n"
    "| One | Two | Three | Four | Five |\n"
    "| --- | --- | --- | --- | --- |\n"
    "| a | bb | ccc | dddd | eeeee |\n"
)


def _make_app(columns: int, color: bool = False) -> tuple[MinAgent, FakeOutput]:
    """A configured session writing to a recorder of the given terminal width."""
    output = _ColorOutput(columns) if color else FakeOutput(columns)
    app = MinAgent(stdout=output)
    app.model = "bonsai27b-8k:latest"
    app.workspace_name = "MinAgent"
    app.input_modalities = ["text", "image"]
    app.context_window = 32768
    app.terminal_mode = "auto"
    return app, output


def _layout_lines(columns: int, color: bool = False) -> list[str]:
    """Render every width-driven surface the way a ``columns``-wide terminal shows it."""
    app, output = _make_app(columns, color)
    app.print_startup_panel()
    app.print_user_bubble("como vas? " + "con el proyecto " * 6)
    app.print_turn_status()
    app.print_command_menu()
    app.print_prompt_token_breakdown()
    app.print_error(Exception(_LONG_NOTE))
    app.print_tool_result("read_file", {"path": "src/minagent/" + _LONG_NOTE}, "ok")
    app.print_tool_result("run_terminal", {}, "exit code 0\n" + _LONG_NOTE)
    app.print_tool_result("write_file", {}, {"tool_text": _LONG_NOTE, "display_text": None})
    if color:
        assert "\x1b[38;2;" in output.text, "expected coloured output"
    lines = output.text.split("\n")

    # The streaming bubble (with its tables) and the autocomplete panel.
    mark = len(output.chunks)
    rendering = create_terminal_rendering(
        output,
        lambda: color,
        UI_COLORS,
        app.ui_text,
        lambda value: output.write(f"{value}\n"),
        lambda value: output.write(f"{value}\n"),
    )
    bubble = rendering.create_streaming_output("Model · bonsai27b-8k:latest")
    bubble.write(_TABLE_MARKDOWN)
    bubble.close()
    lines += "".join(output.chunks[mark:]).split("\n")

    state = build_autocomplete_state("@" + _LONG_NOTE, 1 + len(_LONG_NOTE), ["src/" + _LONG_NOTE], [])
    lines += format_autocomplete_panel(state, columns, color, app.ui_text)
    return [_SGR.sub("", line) for line in lines if line.strip()]


@pytest.mark.parametrize("columns", [200, 120, 100, 80, 72, 64, 50, 40, 30, 24, 16, 12, 8])
@pytest.mark.parametrize("color", [False, True])
def test_ui_lines_fit_the_terminal_width(columns, color, monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    for line in _layout_lines(columns, color):
        assert terminal_text_width(line) <= columns, (columns, line)


@pytest.mark.parametrize("columns", [200, 120, 100, 80, 64, 50, 40, 30, 24, 16, 12, 8])
def test_markdown_table_fits_the_terminal_width(columns):
    rendering, output, buffer = make_rendering(columns)
    bubble = rendering.create_streaming_output("Model · bonsai27b-8k:latest")
    bubble.write(_TABLE_MARKDOWN)
    bubble.close()
    lines = [line for line in (output.text + "".join(buffer)).split("\n") if line.strip()]
    assert lines
    for line in lines:
        assert terminal_text_width(line) <= columns, (columns, line)


def test_session_panel_shrinks_its_meter_on_a_narrow_terminal():
    wide, wide_output = _make_app(80)
    wide.print_startup_panel()
    assert any("Context" in line and "░░" in line for line in wide_output.text.split("\n"))

    narrow, narrow_output = _make_app(30)
    narrow.print_startup_panel()
    panel = narrow_output.text.split("\n")
    assert any("Context" in line for line in panel)
    assert not any("░" in line or "█" in line for line in panel)


# ------------------------------------------------------- terminal rendering


def test_streaming_bubble_keeps_joined_emoji_on_one_line():
    rendering, output, buffer = make_rendering(12)
    bubble = rendering.create_streaming_output("Test")
    for character in "\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466X":
        bubble.write(character)
    bubble.close()
    text = output.text + "".join(buffer)
    assert any("\U0001F468‍\U0001F469‍\U0001F467‍\U0001F466X" in line for line in text.split("\n"))


def _skill_app(tmp_path, output_columns: int = 80) -> MinAgent:
    """An app whose skills come from a temporary directory."""
    app, _output = _make_app(output_columns)
    app.root_directory = str(tmp_path)
    app.skills_enabled = True
    app.skill_write_directory = str(tmp_path / "skills")
    app.skill_directories = [app.skill_write_directory]
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", -1)
    return app


async def test_skill_tools_are_available_before_any_skill_exists(tmp_path):
    app = _skill_app(tmp_path)
    assert "write_skill" not in [tool["function"]["name"] for tool in app.tools]
    await app.initialize_optional_features()
    names = [tool["function"]["name"] for tool in app.tools]
    assert names.count("write_skill") == 1 and "load_skill" in names
    assert "write_skill" in app.skill_prompt_context


async def test_write_skill_registers_the_skill_without_a_restart(tmp_path):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    saved = await app.execute_tool(
        "write_skill",
        {
            "name": "release-notes",
            "description": "Draft release notes from the recent commits",
            "instructions": "1. Collect the commits since the last tag.",
        },
    )
    assert "release-notes" in saved
    assert (tmp_path / "skills" / "release-notes" / "SKILL.md").is_file()
    assert "release-notes" in app.skill_prompt_context
    loaded = await app.execute_tool("load_skill", {"name": "release-notes"})
    assert "Collect the commits" in loaded["tool_text"]
    assert [tool["function"]["name"] for tool in app.tools].count("write_skill") == 1


async def test_a_skill_written_by_hand_is_registered_on_the_next_scan(tmp_path):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    assert app.available_skills == []
    manual = tmp_path / "skills" / "changelog"
    manual.mkdir(parents=True)
    (manual / "SKILL.md").write_text("---\nname: changelog\ndescription: Keep a changelog\n---\n\nKeep it short.\n")
    assert await app.refresh_skills() == ["changelog"]
    assert await app.refresh_skills() == []  # Unchanged directories are not reloaded.
    assert "changelog" in app.skill_prompt_context


async def test_the_system_prompt_catalogue_follows_a_new_skill(tmp_path):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    app._base_system_prompt_sections = app.build_base_system_prompt()
    assert "changelog" not in app.messages[0]["content"]
    manual = tmp_path / "skills" / "changelog"
    manual.mkdir(parents=True)
    manifest = manual / "SKILL.md"
    manifest.write_text("---\nname: changelog\ndescription: Keep a changelog\n---\n\nKeep it short.\n")
    assert await app.refresh_skills() == ["changelog"]
    assert "changelog" in app.messages[0]["content"]
    manifest.unlink()
    assert await app.refresh_skills() == []
    assert "changelog" not in app.messages[0]["content"]


async def test_write_skill_refuses_to_replace_an_existing_skill(tmp_path):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    args = {"name": "demo", "description": "Demo skill", "instructions": "Do the demo."}
    await app.execute_tool("write_skill", args)
    with pytest.raises(AgentError, match="already exists"):
        await app.execute_tool("write_skill", args)
    assert "Saved skill" in await app.execute_tool("write_skill", {**args, "overwrite": True})


async def test_a_turn_creates_the_requested_folder_and_file(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)
    responses = [
        {
            "payload": {"usage": {}},
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "create_directory",
                            "arguments": json.dumps({"path": "informes/2026"}),
                        },
                    },
                    {
                        "id": "c2",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps({"path": "informes/2026/resumen.md", "content": "# Resumen"}),
                        },
                    },
                ],
            },
        },
        {"payload": {"usage": {}}, "message": {"content": "Listo.", "tool_calls": []}},
    ]

    async def next_response(messages, options=None):
        return responses.pop(0)

    monkeypatch.setattr(app, "call_chat_completions", next_response)
    assert await app.request_assistant_turn(None) == "Listo."
    assert (tmp_path / "informes" / "2026" / "resumen.md").read_text() == "# Resumen"
    tool_messages = [message["content"] for message in app.messages if message["role"] == "tool"]
    assert tool_messages == ["Created directory informes/2026.", "Wrote informes/2026/resumen.md."]


def test_the_prompt_gives_the_clock_and_says_which_tool_checks_the_system(tmp_path):
    app = _skill_app(tmp_path)
    app.tools.append(build_terminal_tool())
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.refresh_system_prompt()
    sections = {section["name"]: section["content"] for section in app._current_system_prompt_sections}
    assert "run_terminal" in sections["Core"] and "run_terminal" in sections["Terminal"]
    stamp = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", sections["Current time"])
    assert stamp, sections["Current time"]
    assert abs((datetime.fromisoformat(stamp.group(0)) - datetime.now()).total_seconds()) < 60


def test_the_startup_warning_names_the_prompt_overhead(tmp_path):
    app, output = _make_app(80)
    app.context_window = 2000
    app.workspace_list_limit = -1
    app.workspace_snapshot = "a.txt\n" * 4000
    app.agents_context = "Keep the changelog current."
    app.available_skills = [{"name": "changelog", "description": "Keep a changelog"}]
    app.refresh_system_prompt()
    app.warn_about_prompt_overhead()
    raw = _SGR.sub("", output.text)
    for line in raw.split("\n"):
        assert terminal_text_width(line) <= 80, line
    text = " ".join(raw.split())
    assert "Prompt overhead" in text
    assert "workspace inventory" in text and "AGENTS.md" in text and "tool schemas" in text
    assert "the skill catalogue" in text
    assert "lower WORKSPACE_LIST_LIMIT" in text and "shorten AGENTS.md" in text
    assert "set SKILLS_ENABLED=off" in text
    assert "refused until this is trimmed" in text


def test_the_prompt_overhead_warning_reports_truncation_risk(tmp_path):
    app, output = _make_app(80)
    app.context_window = 10000
    app.compaction_reserve_tokens = 500
    app.workspace_snapshot = "a.txt\n" * 4000
    app.refresh_system_prompt()
    app.warn_about_prompt_overhead()
    text = " ".join(_SGR.sub("", output.text).split())
    assert "if the model's real window is smaller" in text.lower()
    assert "truncates this prompt" in text
    assert "refused until this is trimmed" not in text


def test_no_startup_warning_when_the_fixed_prompt_is_small(tmp_path):
    app, output = _make_app(80)
    app.refresh_system_prompt()
    assert app.prompt_overhead_warning() == ""
    app.warn_about_prompt_overhead()
    assert "Prompt overhead" not in output.text


def _refusal_app(tmp_path, monkeypatch):
    """An app with the terminal tool registered, ready to answer a clock question."""
    app = _skill_app(tmp_path)
    app.tools.append(build_terminal_tool())
    commands: list[str] = []

    async def fake_terminal(args, **kwargs):
        commands.append(args["command"])
        return "Fri Sep 25 18:03:12 CEST 2026"

    monkeypatch.setattr("minagent.app.execute_terminal_command", fake_terminal)
    return app, commands


def _queued_responses(app, monkeypatch, responses):
    async def next_response(messages, options=None):
        return responses.pop(0)

    monkeypatch.setattr(app, "call_chat_completions", next_response)


async def test_a_refusal_without_tool_calls_is_retried_with_the_available_tools(tmp_path, monkeypatch):
    app, commands = _refusal_app(tmp_path, monkeypatch)
    await app.initialize_optional_features()
    app.messages.append({"role": "user", "content": "¿Qué hora es?"})
    _queued_responses(
        app,
        monkeypatch,
        [
            {
                "payload": {"usage": {}},
                "message": {"content": "No tengo acceso al sistema ni a la hora del sistema.", "tool_calls": []},
            },
            {
                "payload": {"usage": {}},
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "run_terminal", "arguments": json.dumps({"command": "date"})}}
                    ],
                },
            },
            {"payload": {"usage": {}}, "message": {"content": "Son las 18:03.", "tool_calls": []}},
        ],
    )
    assert await app.request_assistant_turn(None) == "Son las 18:03."
    assert commands == ["date"]
    notes = [message["content"] for message in app.messages if message["role"] == "user" and "system-note" in str(message["content"])]
    assert len(notes) == 1
    assert "run_terminal" in notes[0] and "write_skill" in notes[0]
    assert "Fri Sep 25 18:03:12 CEST 2026" in [m["content"] for m in app.messages if m["role"] == "tool"]


@pytest.mark.parametrize(
    "phrasing",
    [
        "No tengo acceso al sistema.",
        "Lo siento, no tengo capacidad para acceder a internet ni ejecutar comandos.",
        "No puedo ejecutar comandos de terminal.",
        "No puedo buscar en internet noticias de hoy.",
        "I cannot access the internet or run commands.",
        "I don't have the ability to browse the web.",
        "I am unable to reach the network.",
    ],
)
def test_capability_refusals_are_detected_in_spanish_and_english(phrasing):
    assert _MISSING_CAPABILITY_REQUEST.search(phrasing), phrasing


def test_a_normal_answer_is_not_mistaken_for_a_refusal():
    assert not _MISSING_CAPABILITY_REQUEST.search("He resumido las tres noticias principales.")
    assert not _MISSING_CAPABILITY_REQUEST.search("The build passes and the tests are green.")


async def test_a_second_refusal_is_returned_instead_of_looping(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    app.messages.append({"role": "user", "content": "¿Qué hora es?"})
    refusal = "No tengo acceso al sistema."
    _queued_responses(
        app,
        monkeypatch,
        [
            {"payload": {"usage": {}}, "message": {"content": refusal, "tool_calls": []}},
            {"payload": {"usage": {}}, "message": {"content": refusal, "tool_calls": []}},
        ],
    )
    assert await app.request_assistant_turn(None) == refusal
    notes = [message for message in app.messages if message["role"] == "user" and "system-note" in str(message["content"])]
    assert len(notes) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Voy a listar los correos de tu bandeja Inbox.",
        "Primero comprobaré la cuenta. A continuación listaré los mensajes.",
        "I'll read your inbox now.",
        "Let me fetch the messages.",
    ],
)
def test_announced_plans_are_detected(text):
    assert _ANNOUNCED_ACTION.search(text), text


async def test_an_announced_plan_without_a_tool_call_is_retried(tmp_path, monkeypatch):
    """A model that describes the work instead of doing it is nudged once."""
    app, commands = _refusal_app(tmp_path, monkeypatch)
    await app.initialize_optional_features()
    app.messages.append({"role": "user", "content": "Lee mi correo con himalaya."})
    _queued_responses(
        app,
        monkeypatch,
        [
            {
                "payload": {"usage": {}},
                "message": {"content": "Voy a listar los correos de tu bandeja Inbox.", "tool_calls": []},
            },
            {
                "payload": {"usage": {}},
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "run_terminal",
                                "arguments": json.dumps({"command": "himalaya envelope list -m Inbox -s 3"}),
                            },
                        }
                    ],
                },
            },
            {"payload": {"usage": {}}, "message": {"content": "Estos son tus correos.", "tool_calls": []}},
        ],
    )
    assert await app.request_assistant_turn(None) == "Estos son tus correos."
    assert commands == ["himalaya envelope list -m Inbox -s 3"]
    notes = [
        message
        for message in app.messages
        if message["role"] == "user" and "system-note" in str(message["content"])
    ]
    assert len(notes) == 1


async def test_a_refusal_after_a_tool_ran_is_returned_as_is(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    app.messages.append({"role": "user", "content": "¿Qué hora es?"})
    _queued_responses(
        app,
        monkeypatch,
        [
            {
                "payload": {"usage": {}},
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": "list_directory", "arguments": "{}"}}],
                },
            },
            {"payload": {"usage": {}}, "message": {"content": "No tengo acceso al reloj.", "tool_calls": []}},
        ],
    )
    assert await app.request_assistant_turn(None) == "No tengo acceso al reloj."
    assert not [m for m in app.messages if m["role"] == "user" and "system-note" in str(m["content"])]


def _write_manual_skill(root, name: str, description: str = "Keep a changelog") -> None:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nKeep it short.\n"
    )


async def test_skills_command_shows_and_deletes_a_workspace_skill(tmp_path):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    _write_manual_skill(tmp_path / "skills", "changelog")
    assert await app.refresh_skills() == ["changelog"]

    await app.handle_skills_command("show changelog")
    assert "Keep it short." in app._stdout.text
    await app.handle_skills_command("")  # Listing prints the catalogue.
    assert "changelog" in app._stdout.text

    await app.handle_skills_command("delete changelog")
    assert not (tmp_path / "skills" / "changelog").exists()
    assert app.available_skills == []


async def test_skills_command_reload_reports_a_new_skill(tmp_path):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    _write_manual_skill(tmp_path / "skills", "changelog")
    await app.handle_skills_command("reload")
    assert "new: changelog" in app._stdout.text


async def test_skills_command_refuses_to_delete_a_skill_outside_the_workspace(tmp_path, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside-skills")
    _write_manual_skill(outside, "global")
    app = _skill_app(tmp_path)
    app.skill_directories = [app.skill_write_directory, str(outside)]
    await app.initialize_optional_features()
    assert "global" in [skill["name"] for skill in app.available_skills]
    with pytest.raises(AgentError, match="outside the workspace"):
        await app.handle_skills_command("delete global")
    assert (outside / "global" / "SKILL.md").is_file()


def test_skill_names_are_slugified_with_accents_folded():
    assert slugify_skill_name("Notas de versión") == "notas-de-version"
    assert slugify_skill_name("  Release Notes!! ") == "release-notes"
    assert slugify_skill_name("a" * 80) == "a" * 64
    assert slugify_skill_name("!!!") == ""


async def test_skill_command_drafts_and_registers_a_skill(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    draft = (
        "```markdown\n---\nname: Release Notes\ndescription: Draft release notes from the commits.\n---\n\n"
        "1. Collect the commits since the last tag.\n```"
    )

    async def fake_call(messages, options=None):
        return {"payload": {"usage": {}}, "message": {"content": draft, "tool_calls": []}}

    monkeypatch.setattr(app, "call_chat_completions", fake_call)
    created = await app.generate_skill("redactar notas de versión", None)
    assert created["name"] == "release-notes"
    assert "release-notes" in app.skill_prompt_context

    async def empty_draft(messages, options=None):
        return {"payload": {"usage": {}}, "message": {"content": "no frontmatter here", "tool_calls": []}}

    monkeypatch.setattr(app, "call_chat_completions", empty_draft)
    with pytest.raises(AgentError, match="frontmatter"):
        await app.generate_skill("algo sin formato", None)


def _mcp_app(tmp_path) -> MinAgent:
    app, _output = _make_app(80)
    app.root_directory = str(tmp_path)
    app.mcp_enabled = True
    app.mcp_config_path = str(tmp_path / ".minagent" / "mcp.json")
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    return app


async def test_mcp_tools_follow_a_configuration_change(tmp_path, monkeypatch):
    catalog = {
        "one": (
            {"type": "function", "function": {"name": "mcp_0_0_echo", "description": "Echo"}},
            "echo",
        ),
        "two": (
            {"type": "function", "function": {"name": "mcp_0_0_sum", "description": "Sum"}},
            "add",
        ),
    }
    state = {"tool": "one"}
    connects: list[str] = []

    async def fake_connect(config_path, default_cwd, timeout_ms=0):
        connects.append(config_path)
        definition, remote_name = catalog[state["tool"]]
        return {
            "clients": [],
            "tool_definitions": [definition],
            "tool_lookup": {definition["function"]["name"]: {
                "server_name": state["tool"], "remote_tool_name": remote_name,
            }},
            "server_guidance": [],
            "warnings": [],
        }

    monkeypatch.setattr("minagent.app.connect_mcp_servers", fake_connect)
    app = _mcp_app(tmp_path)
    (tmp_path / ".minagent").mkdir()
    config_file = tmp_path / ".minagent" / "mcp.json"
    config_file.write_text('{"mcpServers": {}}')

    await app.initialize_optional_features()
    names = lambda: [tool["function"]["name"] for tool in app.tools]
    assert "mcp_0_0_echo" in names()
    assert await app.refresh_mcp_servers() == []  # Unchanged file: no reconnect.
    assert len(connects) == 1

    state["tool"] = "two"
    config_file.write_text('{"mcpServers": {"two": {"url": "http://127.0.0.1:1/mcp"}}}')
    assert await app.refresh_mcp_servers() == ["add"]
    assert "mcp_0_0_sum" in names() and "mcp_0_0_echo" not in names()
    assert len(connects) == 2


async def test_authoring_an_mcp_server_needs_approval_and_registers_its_tools(tmp_path, monkeypatch):
    app = _mcp_app(tmp_path)
    await app.initialize_optional_features()
    assert "write_mcp_server" in [tool["function"]["name"] for tool in app.tools]

    class _AnsweringEditor:
        def __init__(self, answer: str) -> None:
            self.answer = answer

        async def question(self, prompt: str) -> str:
            return self.answer

    written: list[dict[str, Any]] = []

    def fake_author(application_root, config_path, args):
        written.append(args)
        return {
            "name": args["name"],
            "entry": {},
            "script_path": str(tmp_path / ".agents" / "mcp" / args["name"] / "server.mjs"),
            "config_path": config_path,
        }

    async def fake_refresh(force: bool = False) -> list[str]:
        app.mcp_connections = {
            "tool_lookup": {"mcp_0_0_nuevo_saludar": {"server_name": "nuevo", "remote_tool_name": "saludar"}}
        }
        return ["saludar"]

    monkeypatch.setattr("minagent.app.author_mcp_server", fake_author)
    monkeypatch.setattr(app, "refresh_mcp_servers", fake_refresh)

    app.editor = _AnsweringEditor("n")
    denied = await app.execute_tool("write_mcp_server", {"name": "nuevo", "command": "node"})
    assert "denied" in denied and written == []

    app.editor = _AnsweringEditor("y")
    accepted = await app.execute_tool("write_mcp_server", {"name": "nuevo", "command": "node"})
    assert written == [{"name": "nuevo", "command": "node"}]
    assert "mcp_0_0_nuevo_saludar" in accepted


async def test_each_request_sends_the_current_tool_list(tmp_path, monkeypatch):
    app = _skill_app(tmp_path)
    await app.initialize_optional_features()
    captured: dict[str, Any] = {}

    async def stop_turn(messages, options=None):
        captured.update(options or {})
        raise _TurnStopped

    monkeypatch.setattr(app, "call_chat_completions", stop_turn)
    with pytest.raises(_TurnStopped):
        await app.request_assistant_turn(None)
    assert captured["available_tools"] is app.tools
    assert "write_skill" in [tool["function"]["name"] for tool in captured["available_tools"]]


def test_reasoning_stream_wraps_to_the_terminal_width():
    rendering, output, buffer = make_rendering(20)
    reasoning = rendering.create_reasoning_streaming_output()
    for chunk in ("El usuario quiere ", "un juego de la serpiente ", "en ./app/snake.html."):
        reasoning.write(chunk)
    reasoning.close()
    lines = [line for line in output.text.split("\n") if line]
    assert len(lines) > 1
    for line in lines:
        assert terminal_text_width(line) <= 20, line
    assert " ".join(lines).split() == "El usuario quiere un juego de la serpiente en ./app/snake.html.".split()


def test_reasoning_stream_keeps_a_word_together_across_chunks():
    rendering, output, buffer = make_rendering(20)
    reasoning = rendering.create_reasoning_streaming_output()
    for chunk in ("th", "inking ", "about "):
        reasoning.write(chunk)
    assert output.text == ""  # Nothing is cut until a line is actually full.
    reasoning.write("the problem")
    reasoning.close()
    assert output.text == "thinking about the\nproblem\n"


def test_reasoning_stream_keeps_explicit_line_breaks():
    rendering, output, buffer = make_rendering(20)
    reasoning = rendering.create_reasoning_streaming_output()
    reasoning.write("uno\ndos tres cuatro cinco\nseis\n")
    reasoning.close()
    assert output.text == "uno\ndos tres cuatro\ncinco\nseis\n"


def test_interrupted_streaming_bubble_is_labelled_incomplete():
    rendering, output, buffer = make_rendering(40)
    bubble = rendering.create_streaming_output("Test")
    bubble.write("partial")
    bubble.close("incomplete")
    assert "incomplete" in output.text + "".join(buffer)


def _configuration(tmp_path, **env: str) -> Config:
    return load_configuration(
        application_root=str(tmp_path),
        cwd=str(tmp_path),
        env={"OPENAI_MODEL": "test-model", **env},
    )


def test_endpoint_timeout_defaults_to_seven_minutes(tmp_path):
    assert _configuration(tmp_path).endpoint_timeout_ms == 7 * 60 * 1000


def test_endpoint_timeout_is_configurable_in_seconds(tmp_path):
    config = _configuration(tmp_path, OPENAI_TIMEOUT_SECONDS="30")
    assert config.endpoint_timeout_ms == 30_000


def test_endpoint_timeout_rejects_a_non_positive_value(tmp_path):
    with pytest.raises(AgentError, match="OPENAI_TIMEOUT_SECONDS must be a positive integer"):
        _configuration(tmp_path, OPENAI_TIMEOUT_SECONDS="0")


def test_tool_rounds_default_to_sixty_four(tmp_path):
    assert _configuration(tmp_path).max_tool_rounds == 64


def test_tool_rounds_are_configurable(tmp_path):
    assert _configuration(tmp_path, MAX_TOOL_ROUNDS="8").max_tool_rounds == 8


def test_tool_rounds_reject_a_non_positive_value(tmp_path):
    with pytest.raises(AgentError, match="MAX_TOOL_ROUNDS must be a positive integer"):
        _configuration(tmp_path, MAX_TOOL_ROUNDS="0")


def test_the_tool_preview_defaults_to_twelve_thousand_characters(tmp_path):
    assert _configuration(tmp_path).tool_preview_chars == 12000


def test_the_tool_preview_is_configurable(tmp_path):
    assert _configuration(tmp_path, TOOL_PREVIEW_CHARS="2000").tool_preview_chars == 2000


def test_the_tool_preview_rejects_a_non_positive_value(tmp_path):
    with pytest.raises(AgentError, match="TOOL_PREVIEW_CHARS must be a positive integer"):
        _configuration(tmp_path, TOOL_PREVIEW_CHARS="0")


def test_the_preview_is_capped_by_the_configured_budget():
    """A big window must not let one result eat a quarter of it."""
    app, _output = _make_app(80)
    app.model = "modelo-sin-pista-de-tamano"
    app.model_context_length = None
    app.context_window = 262144
    app.tool_preview_chars = 12000
    assert app.effective_context_window() == 262144
    bounded = app.bound_tool_result("x" * 500_000)
    assert len(bounded) <= 12000 + 400, "the inline preview ignored TOOL_PREVIEW_CHARS"


def test_the_preview_never_exceeds_the_window_itself(tmp_path):
    """A tiny window still bounds the result even with a large preview budget."""
    app, _output = _make_app(80)
    assert app.effective_context_window() == 8192
    app.tool_preview_chars = 500_000
    bounded = app.bound_tool_result("x" * 500_000)
    assert len(bounded) <= 8192 + 400


def test_extension_timeouts_default_to_seven_minutes(tmp_path):
    config = _configuration(tmp_path)
    assert config.mcp_timeout_ms == 7 * 60 * 1000
    assert config.terminal_timeout_seconds == 7 * 60


def test_extension_timeouts_are_configurable_in_seconds(tmp_path):
    config = _configuration(tmp_path, MCP_TIMEOUT_SECONDS="5", TERMINAL_TIMEOUT_SECONDS="30")
    assert config.mcp_timeout_ms == 5_000
    assert config.terminal_timeout_seconds == 30


def test_extension_timeouts_reject_a_non_positive_value(tmp_path):
    with pytest.raises(AgentError, match="TERMINAL_TIMEOUT_SECONDS must be a positive integer"):
        _configuration(tmp_path, TERMINAL_TIMEOUT_SECONDS="0")
    with pytest.raises(AgentError, match="MCP_TIMEOUT_SECONDS must be a positive integer"):
        _configuration(tmp_path, MCP_TIMEOUT_SECONDS="forever")


async def test_terminal_command_honours_a_custom_timeout(tmp_path):
    result = await run_terminal_command(
        {"command": "sleep 5"},
        terminal_mode="auto",
        terminal_command_shell="/bin/sh",
        root_directory=str(tmp_path),
        interactive_terminal=None,
        print=lambda value: None,
        ui_print=lambda value: None,
        ui_text=lambda text, *styles: text,
        timeout_seconds=1,
    )
    assert "Command stopped after 1 seconds" in result


async def test_startup_passes_the_configured_timeout_to_the_endpoint_client(tmp_path, monkeypatch):
    config = _configuration(tmp_path, OPENAI_TIMEOUT_SECONDS="30")
    monkeypatch.setattr("minagent.app.load_configuration", lambda *_args, **_kwargs: config)
    app = MinAgent(stdout=FakeOutput(80))
    await app.initialize_configuration()
    assert app.open_ai_client.timeout_ms == 30_000


def test_markdown_table_keeps_escaped_pipes_in_a_single_cell():
    rendering, output, buffer = make_rendering(60)
    bubble = rendering.create_streaming_output("Test")
    bubble.write("| A\\|B | C |\n| --- | --- |\n| x | y |")
    bubble.close()
    assert "A|B" in output.text + "".join(buffer)


async def test_a_new_conversation_starts_empty_with_no_state_from_the_last_one(tmp_path):
    """``/new`` must leave nothing of the finished run behind, and say why the bar is not empty."""
    app = _make_app(80)[0]
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    app._base_system_prompt_sections = app.build_base_system_prompt()
    app.messages.append({"role": "user", "content": "hola"})
    app.messages.append({"role": "assistant", "content": "respuesta larga " * 200})
    app.compacted_summary = "resumen " * 200
    app._tools_used_this_turn = ["read_file"]
    app._steps_this_turn = ["read_file(path=README.md)"]
    app._current_user_request = "hola"
    app._record_tool_tokens("read_file", "salida " * 4000, "salida " * 100)
    app._session_cleared_tool_result_tokens = 4242
    app._tool_error_this_turn = True
    app._web_search_prompted_this_turn = True
    app.refresh_system_prompt()
    app.request_cache.put(
        app.request_cache.key(app.model, app.tools, app.messages),
        {"payload": {"finish_reason": "stop"}, "message": {"content": "respuesta"}},
    )

    await app.start_new_conversation()

    assert app.messages == [app.messages[0]], "only the system prompt should survive"
    assert app.compacted_summary == ""
    assert app._tools_used_this_turn == [] and app._steps_this_turn == []
    assert app._current_user_request == ""
    assert app._turn_tool_tokens == {} and app._session_tool_tokens == {}
    assert app._session_archived_tokens == 0 and app._session_cleared_tool_result_tokens == 0
    assert app._tool_error_this_turn is False and app._web_search_prompted_this_turn is False
    assert app.last_prompt_tokens is None and app.last_usage_message_count == 0
    # A replay entry from the previous conversation would answer the next
    # identical question without ever reaching the endpoint.
    assert len(app.request_cache) == 0

    usage = app._context_usage()
    assert usage["conversation"] == 0
    assert usage["fixed"] > 0, "the system sections and tool schemas are resent every request"
    assert usage["used"] == pytest.approx(usage["fixed"])
    assert "The conversation is empty" in app._stdout.text
    assert "Fixed" in app._stdout.text
