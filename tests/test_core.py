"""Core behaviour tests: workspace safety, attachments, streaming, and the editor."""

from __future__ import annotations

import asyncio
import json
import os
import pty
import re
import sys
import tty
from datetime import datetime
from typing import Any

import pytest

from minagent.app import UI_COLORS, MinAgent, build_terminal_tool
from minagent.attachments import prepare_user_message
from minagent.config import load_configuration, parse_directory_entry_limit
from minagent.errors import AgentError
from minagent.context import chunk_summary_transcript
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
from minagent.init_project import collect_project_essentials
from minagent.line_editor import LineEditor
from minagent.markdown_terminal import create_terminal_rendering
from minagent.openai import read_streaming_response
from minagent.secrets import approval_preview
from minagent.skills import create_skill_tools, execute_skill_tool, slugify_skill_name
from minagent.terminal_command import run_terminal_command
from minagent.terminal_text import (
    terminal_columns,
    terminal_text_width,
    truncate_terminal_text,
    wrap_styled_segments,
)
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


def _configuration(tmp_path, **env: str) -> dict[str, Any]:
    return load_configuration(
        application_root=str(tmp_path),
        cwd=str(tmp_path),
        env={"OPENAI_MODEL": "test-model", **env},
    )


def test_endpoint_timeout_defaults_to_seven_minutes(tmp_path):
    assert _configuration(tmp_path)["endpoint_timeout_ms"] == 7 * 60 * 1000


def test_endpoint_timeout_is_configurable_in_seconds(tmp_path):
    config = _configuration(tmp_path, OPENAI_TIMEOUT_SECONDS="30")
    assert config["endpoint_timeout_ms"] == 30_000


def test_endpoint_timeout_rejects_a_non_positive_value(tmp_path):
    with pytest.raises(AgentError, match="OPENAI_TIMEOUT_SECONDS must be a positive integer"):
        _configuration(tmp_path, OPENAI_TIMEOUT_SECONDS="0")


def test_extension_timeouts_default_to_seven_minutes(tmp_path):
    config = _configuration(tmp_path)
    assert config["mcp_timeout_ms"] == 7 * 60 * 1000
    assert config["terminal_timeout_seconds"] == 7 * 60


def test_extension_timeouts_are_configurable_in_seconds(tmp_path):
    config = _configuration(tmp_path, MCP_TIMEOUT_SECONDS="5", TERMINAL_TIMEOUT_SECONDS="30")
    assert config["mcp_timeout_ms"] == 5_000
    assert config["terminal_timeout_seconds"] == 30


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
