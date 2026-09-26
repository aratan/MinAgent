"""Prompt editing: autocomplete, pasted input, and Ctrl+J newlines.

The prompt is a single logical line that may contain newlines, so a pasted
multi-line block or a Ctrl+J keystroke inserts a break instead of submitting the
turn. ``@`` starts a file search and ``/`` a command search.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .terminal_text import safe_terminal_text, terminal_text_width, truncate_terminal_text

MAX_AUTOCOMPLETE_CANDIDATES = 1000
AUTOCOMPLETE_PANEL_ROWS = 7
AUTOCOMPLETE_MAX_ITEMS = AUTOCOMPLETE_PANEL_ROWS - 2

_SELECTED_ROW_STYLE = "\x1b[48;2;32;93;112;38;2;226;239;241m"
_RESET = "\x1b[0m"
_WHITESPACE = re.compile(r"\s")
_NON_SPACE_TOKEN = re.compile(r"^[^\s]*")
_COMMAND_PREFIX = re.compile(r"^(\s*)\/([^\s]*)$")
_MODEL_PREFIX = re.compile(r"^\s*/model\s+")
_WHITESPACE_RUN = re.compile(r"\s+")

NAVIGATION_KEYS = {
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown", "delete", "backspace", "escape",
}


@dataclass
class Key:
    """One keypress, matching the fields the prompt logic inspects."""

    name: str = ""
    ctrl: bool = False
    meta: bool = False
    sequence: str = ""


@dataclass
class PasteState:
    """Bracketed-paste bookkeeping for the current prompt."""

    active: bool = False
    bulk_input_chunk: bool = False
    skip_next_line_feed: bool = False
    line_feed_timer: Any = None


def insert_newline(terminal: Any) -> None:
    """Insert a newline at the cursor instead of submitting the turn."""
    line = getattr(terminal, "line", None)
    if not isinstance(line, str):
        return
    cursor = terminal.cursor if isinstance(getattr(terminal, "cursor", None), int) else len(line)
    cursor = max(0, min(cursor, len(line)))
    terminal.line = f"{line[:cursor]}\n{line[cursor:]}"
    terminal.cursor = cursor + 1
    mark_multiline = getattr(terminal, "mark_multiline", None)
    if callable(mark_multiline):
        mark_multiline()
    prompt = getattr(terminal, "prompt", None)
    if callable(prompt):
        prompt(True)


def is_bulk_input_chunk(terminal: Any) -> bool:
    """True when the chunk arrived as one paste burst rather than a keystroke."""
    if getattr(terminal, "is_completion_enabled", None) is False:
        return True
    return getattr(terminal, "saw_key_press", True) is False


def reset_prompt_rows(terminal: Any) -> None:
    """Force the prompt to repaint from a clean row count."""
    terminal.prev_rows = 0
    prompt = getattr(terminal, "prompt", None)
    if callable(prompt):
        prompt(True)


def measure_submitted_input_rows(
    terminal: Any, value: str, prompt_width: int, columns: int
) -> int:
    """Rows the submitted input occupied, so the prompt can erase exactly that."""
    from .terminal_text import terminal_rows_for_input

    get_cursor_pos = getattr(terminal, "get_cursor_pos", None)
    if callable(get_cursor_pos) and isinstance(getattr(terminal, "line", None), str):
        original_cursor = terminal.cursor
        try:
            terminal.cursor = len(terminal.line)
            position = get_cursor_pos()
            rows = getattr(position, "rows", None)
            if isinstance(rows, int) and not isinstance(rows, bool):
                return rows + 1
        finally:
            terminal.cursor = original_cursor
    return terminal_rows_for_input(value, prompt_width, columns)


def suppress_readline_key(key: Key | None) -> None:
    """Rewrite a keypress so the underlying editor ignores it."""
    if key is None:
        return
    key.name = "j"
    key.ctrl = True
    key.meta = False


def _clear_pending_line_feed(state: PasteState) -> None:
    state.skip_next_line_feed = False
    if state.line_feed_timer is not None:
        state.line_feed_timer.cancel()
        state.line_feed_timer = None


def _schedule_line_feed_clear(state: PasteState) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    state.line_feed_timer = loop.call_later(0.1, lambda: _clear_pending_line_feed(state))


def handle_pasted_input(key: Key | None, character: str, terminal: Any, state: PasteState) -> bool:
    """Keep pasted line breaks inside one prompt instead of submitting per line.

    Arrow keys are still single physical keypresses and must still reach
    autocomplete, so they are excluded from bulk-chunk detection.
    """
    key_name = getattr(key, "name", None)
    state.bulk_input_chunk = key_name not in NAVIGATION_KEYS and is_bulk_input_chunk(terminal)

    if key_name == "paste-start":
        state.active = True
        _clear_pending_line_feed(state)
        if key is not None:
            key.name = "unbound"
        return True
    if key_name == "paste-end":
        state.active = False
        _clear_pending_line_feed(state)
        if key is not None:
            key.name = "unbound"
        return True
    if not state.active and not state.bulk_input_chunk:
        return False

    if character == "\r":
        insert_newline(terminal)
        state.skip_next_line_feed = True
        if state.line_feed_timer is not None:
            state.line_feed_timer.cancel()
        _schedule_line_feed_clear(state)
        suppress_readline_key(key)
        return True
    if character == "\n":
        if not state.skip_next_line_feed:
            insert_newline(terminal)
        _clear_pending_line_feed(state)
        suppress_readline_key(key)
        return True
    if state.skip_next_line_feed:
        _clear_pending_line_feed(state)
    return True


def handle_control_j_input(key: Key | None, character: str, terminal: Any) -> bool:
    """Treat Ctrl+J (and a bare LF) as a newline rather than a submit."""
    is_control_j = (getattr(key, "ctrl", False) and getattr(key, "name", None) == "j") or (
        character == "\n"
        and getattr(key, "sequence", None) == "\n"
        and getattr(key, "name", None) == "enter"
        and not getattr(key, "ctrl", False)
        and not getattr(key, "meta", False)
    )
    if not is_control_j:
        return False
    insert_newline(terminal)
    # Otherwise the LF is treated as Enter and submits the line.
    suppress_readline_key(key)
    return True


def build_autocomplete_state(
    line: str,
    cursor: int,
    workspace_files: Sequence[str],
    slash_commands: Sequence[dict[str, str]],
    models: Sequence[str] = (),
    current_model: str = "",
) -> dict[str, Any] | None:
    """Build the autocomplete panel state for the token under the cursor."""
    prefix = line[:cursor]
    token_match = _NON_SPACE_TOKEN.match(line[cursor:])
    token_end = cursor + (len(token_match.group(0)) if token_match else 0)

    model_match = _MODEL_PREFIX.match(prefix)
    if model_match and models:
        query = prefix[model_match.end():]
        lowered = query.lower()
        candidates = [
            {
                "value": name,
                "label": f"{name}  current" if name == current_model else name,
            }
            for name in models
            if name.lower().startswith(lowered)
        ]
        if not candidates:
            return None
        return {
            "kind": "model",
            "line": line,
            "cursor": cursor,
            "start": model_match.end(),
            "end": token_end,
            "query": query,
            "candidates": candidates,
            "total_matches": len(candidates),
            "selected_index": 0,
        }

    command_match = _COMMAND_PREFIX.match(prefix)
    if command_match:
        query = command_match.group(2).lower()
        candidates = [
            {"value": f"/{command['name']}", "label": f"/{command['name']}  {command['description']}"}
            for command in slash_commands
            if command["name"].startswith(query) and command["name"] != query
        ]
        if not candidates:
            return None
        return {
            "kind": "command",
            "line": line,
            "cursor": cursor,
            "start": len(command_match.group(1)),
            "end": token_end,
            "query": query,
            "candidates": candidates,
            "total_matches": len(candidates),
            "selected_index": 0,
        }

    at_index = prefix.rfind("@")
    if at_index < 0 or (at_index > 0 and not _WHITESPACE.match(prefix[at_index - 1])):
        return None
    raw_query = prefix[at_index + 1:]
    query = raw_query.rstrip()
    ranked = rank_workspace_files(query, workspace_files)
    if not ranked["paths"]:
        return None
    return {
        "kind": "file",
        "line": line,
        "cursor": cursor,
        "start": at_index,
        "end": token_end,
        "query": query,
        "trailing_space": len(raw_query) > len(query),
        "candidates": [{"value": path, "label": path} for path in ranked["paths"]],
        "total_matches": ranked["total_matches"],
        "selected_index": 0,
    }


def handle_autocomplete_keypress(state: dict[str, Any] | None, key: Key | None, terminal: Any) -> dict[str, Any] | None:
    """Move the selection or complete the selected candidate in place."""
    if not state or not state.get("candidates") or key is None:
        return None
    if terminal.line != state["line"] or terminal.cursor != state["cursor"]:
        return None
    if key.name in ("up", "down"):
        direction = -1 if key.name == "up" else 1
        suppress_readline_key(key)
        total = len(state["candidates"])
        state["selected_index"] = (state["selected_index"] + direction + total) % total
        return {"kind": "move"}
    if key.name not in ("return", "enter") or key.ctrl or key.meta:
        return None
    suppress_readline_key(key)
    selected = state["candidates"][state["selected_index"]] or state["candidates"][0]
    replacement = f"{selected['value']} " if state["kind"] == "file" and state.get("trailing_space") else selected["value"]
    terminal.line = f"{state['line'][:state['start']]}{replacement}{state['line'][state['end']:]}"
    terminal.cursor = state["start"] + len(replacement)
    return {"kind": "complete", "selected_file": selected["value"] if state["kind"] == "file" else None}


def format_autocomplete_panel(
    state: dict[str, Any],
    columns: int = 80,
    use_color: bool = False,
    ui_text: Callable[..., str] = lambda value, *args: value,
) -> list[str]:
    """Render the autocomplete panel as fixed-height lines."""
    title = {"file": "Files", "model": "Models"}.get(state["kind"], "Commands")
    matches = state["total_matches"]
    # The panel keeps a fixed height, so its own chrome is truncated rather than wrapped.
    lines = [
        ui_text(
            truncate_terminal_text(
                f"  ┌─ {title.upper()} · {matches} match{'es' if matches != 1 else ''}", columns
            ),
            "magenta" if state["kind"] in ("command", "model") else "cyan",
            True,
        )
    ]
    first_visible_index = max(
        0,
        min(
            state["selected_index"] - AUTOCOMPLETE_MAX_ITEMS // 2,
            len(state["candidates"]) - AUTOCOMPLETE_MAX_ITEMS,
        ),
    )
    for visible_index in range(AUTOCOMPLETE_MAX_ITEMS):
        index = first_visible_index + visible_index
        candidate = state["candidates"][index] if index < len(state["candidates"]) else None
        if not candidate:
            lines.append("")
            continue
        marker = "›" if index == state["selected_index"] else " "
        label = _WHITESPACE_RUN.sub(" ", safe_terminal_text(candidate["label"]))
        shown = truncate_terminal_text(label, max(1, columns - 6))
        option = f"  {marker} {shown}"
        lines.append(
            f"{_SELECTED_ROW_STYLE}{safe_terminal_text(option)}{_RESET}"
            if use_color and index == state["selected_index"]
            else ui_text(option, "muted")
        )
    lines.append(
        ui_text(truncate_terminal_text("  └─ ↑/↓ select · Enter complete · Esc close", columns), "muted")
    )
    return lines


def rank_workspace_files(query: str, workspace_files: Sequence[str]) -> dict[str, Any]:
    """Rank file paths against a query; exact prefixes and basenames win."""
    normalized = query.lower()
    ranked: list[tuple[str, float]] = []
    total_matches = 0
    for path in workspace_files:
        lower_path = path.lower()
        file_name = lower_path[lower_path.rfind("/") + 1:]
        if not normalized:
            score = 0.0
        elif file_name.startswith(normalized):
            score = 0.0
        elif lower_path.startswith(normalized):
            score = 1.0
        elif normalized in file_name:
            score = 2 + file_name.find(normalized) / 1000
        elif normalized in lower_path:
            score = 3 + lower_path.find(normalized) / 1000
        else:
            query_index = 0
            gaps = 0
            for character in lower_path:
                if query_index < len(normalized) and character == normalized[query_index]:
                    query_index += 1
                elif query_index > 0:
                    gaps += 1
                if query_index == len(normalized):
                    break
            if query_index != len(normalized):
                continue
            score = 10 + gaps
        total_matches += 1
        ranked.append((path, score))
    ranked.sort(key=lambda item: (item[1], item[0]))
    return {
        "paths": [path for path, _score in ranked[:MAX_AUTOCOMPLETE_CANDIDATES]],
        "total_matches": total_matches,
    }


__all__ = [
    "AUTOCOMPLETE_PANEL_ROWS",
    "Key",
    "PasteState",
    "build_autocomplete_state",
    "format_autocomplete_panel",
    "handle_autocomplete_keypress",
    "handle_control_j_input",
    "handle_pasted_input",
    "insert_newline",
    "is_bulk_input_chunk",
    "measure_submitted_input_rows",
    "rank_workspace_files",
    "reset_prompt_rows",
    "terminal_text_width",
]
