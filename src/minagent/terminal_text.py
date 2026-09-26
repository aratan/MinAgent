"""Terminal text measurement: sanitising, grapheme clustering, and wrapping.

Everything here works in terminal cells, not code points, so CJK, emoji
sequences, and combining marks lay out correctly.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import regex

_CARRIAGE_RETURN = regex.compile(r"\r\n?")
_UNSAFE_CONTROLS = regex.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_GRAPHEME = regex.compile(r"\X")
_COMBINING_MARK = regex.compile(r"\p{M}")
_EMOJI_PRESENTATION = regex.compile(r"[️️⃣]")
_EXTENDED_PICTOGRAPHIC = regex.compile(r"\p{Extended_Pictographic}|\p{Regional_Indicator}")
_WHITESPACE = regex.compile(r"\s")


def safe_terminal_text(value: object) -> str:
    """Strip carriage returns and control characters that would corrupt layout."""
    text = _CARRIAGE_RETURN.sub("\n", str(value))
    return _UNSAFE_CONTROLS.sub("", text)


def terminal_columns(stream: Any = None, default: int = 80) -> int:
    """Terminal width to lay out against.

    Node's ``process.stdout.columns`` has no Python equivalent on text streams,
    so an explicit ``columns`` attribute wins when present (test stand-ins) and
    the size of the attached terminal is queried otherwise. The size is read on
    every call so a resized window is picked up.
    """
    explicit = getattr(stream, "columns", None)
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    descriptor = getattr(stream, "fileno", None)
    if callable(descriptor):
        try:
            columns = os.get_terminal_size(descriptor()).columns
        except (OSError, ValueError):
            columns = 0
        if columns > 0:
            return columns
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = 0
    return columns if columns > 0 else default


def graphemes(value: str) -> list[str]:
    """Split text into user-perceived characters (extended grapheme clusters)."""
    return _GRAPHEME.findall(str(value))


def _code_point_width(character: str) -> int:
    """Terminal cell width of a single code point."""
    code_point = ord(character)
    if _COMBINING_MARK.fullmatch(character):
        return 0
    if code_point < 32 or 0x7F <= code_point < 0xA0:
        return 0
    if (
        0x1100 <= code_point <= 0x11FF
        or 0x2E80 <= code_point <= 0xA4CF
        or 0xAC00 <= code_point <= 0xD7AF
        or 0xF900 <= code_point <= 0xFAFF
        or 0xFE10 <= code_point <= 0xFE6F
        or 0xFF00 <= code_point <= 0xFF60
        or 0x1F300 <= code_point <= 0x1FAFF
        or 0x20000 <= code_point <= 0x3FFFD
    ):
        return 2
    return 1


def terminal_character_width(cluster: str) -> int:
    """Terminal cell width of a grapheme cluster."""
    if _EMOJI_PRESENTATION.search(cluster) or _EXTENDED_PICTOGRAPHIC.search(cluster):
        return 2
    return sum(_code_point_width(character) for character in cluster)


def terminal_text_width(value: str) -> int:
    """Total terminal cell width of a string."""
    return sum(terminal_character_width(cluster) for cluster in graphemes(value))


def truncate_terminal_text(value: str, max_width: int) -> str:
    """Truncate to ``max_width`` cells, reserving one cell for the ellipsis."""
    safe = safe_terminal_text(value)
    if terminal_text_width(safe) <= max_width:
        return safe
    if max_width < 1:
        return ""
    output = ""
    width = 0
    for cluster in graphemes(safe):
        cluster_width = terminal_character_width(cluster)
        if width + cluster_width > max_width - 1:
            break
        output += cluster
        width += cluster_width
    return f"{output.rstrip()}…"


def wrap_text_line(value: str, width: int) -> list[str]:
    """Wrap one logical line to ``width`` cells, preferring word boundaries."""
    remaining = graphemes(value)
    lines: list[str] = []
    while terminal_text_width("".join(remaining)) > width:
        used_width = 0
        cut = 0
        last_space = -1
        for index, character in enumerate(remaining):
            character_width = terminal_character_width(character)
            if used_width + character_width > width:
                break
            used_width += character_width
            cut = index + 1
            if _WHITESPACE.fullmatch(character):
                last_space = index
        break_at = last_space if last_space > 0 else max(1, cut)
        lines.append("".join(remaining[:break_at]).rstrip())
        remaining = remaining[break_at:]
        while remaining and _WHITESPACE.fullmatch(remaining[0]):
            remaining = remaining[1:]
    lines.append("".join(remaining))
    return lines


def wrap_message(text: str, width: int) -> list[str]:
    """Wrap a multi-line message to ``width`` cells."""
    lines: list[str] = []
    for line in safe_terminal_text(text).split("\n"):
        lines.extend(wrap_text_line(line, width))
    return lines


# A piece of one UI line: its text, its colour name, and whether it is bold.
StyledSegment = tuple[str, str, bool]


def split_segment_lines(segments: Sequence[StyledSegment]) -> list[list[StyledSegment]]:
    """Split styled segments on embedded newlines, one segment list per line."""
    lines: list[list[StyledSegment]] = [[]]
    for value, color, bold in segments:
        for index, part in enumerate(safe_terminal_text(value).split("\n")):
            if index:
                lines.append([])
            if part:
                lines[-1].append((part, color, bold))
    return lines


def wrap_styled_segments(segments: Sequence[StyledSegment], width: int) -> list[list[StyledSegment]]:
    """Wrap styled segments into lines of at most ``width`` cells.

    A segment that does not fit moves to the next line whole, so a block such as
    ``Context ~1 / 2`` is never split merely to fill the current line. Only a
    segment wider than the terminal itself is wrapped internally, and the space
    a break lands on is dropped.
    """
    limit = max(1, width)
    lines: list[list[StyledSegment]] = []
    current: list[StyledSegment] = []
    current_width = 0

    def append(text: str, color: str, bold: bool) -> None:
        nonlocal current_width
        if current and current[-1][1] == color and current[-1][2] == bold:
            current[-1] = (current[-1][0] + text, color, bold)
        else:
            current.append((text, color, bold))
        current_width += terminal_text_width(text)

    def trim_trailing_space() -> None:
        """Drop the space a break lands on, which may sit inside a merged token."""
        nonlocal current_width
        while current:
            text, color, bold = current[-1]
            stripped = text.rstrip()
            if stripped == text:
                return
            if stripped:
                current[-1] = (stripped, color, bold)
            else:
                current.pop()
            current_width = sum(terminal_text_width(part) for part, _, _ in current)

    def flush() -> None:
        nonlocal current, current_width
        trim_trailing_space()
        if current:
            lines.append(current)
        current = []
        current_width = 0

    for value, color, bold in segments:
        text = safe_terminal_text(value)
        if not text:
            continue
        if current_width + terminal_text_width(text) <= limit:
            append(text, color, bold)
            continue
        if current:
            # The break lands on this segment's leading space, which is a separator.
            flush()
            text = text.lstrip()
            if not text.strip():
                continue
        if terminal_text_width(text) <= limit:
            append(text, color, bold)
            continue
        for index, fragment in enumerate(wrap_text_line(text, limit)):
            if index:
                flush()
            append(fragment, color, bold)
    flush()
    return lines or [[("", "pale", False)]]


def render_styled_line(segments: Sequence[StyledSegment], ui_text: Callable[..., str]) -> str:
    """Render one line of styled segments into a colored string."""
    return "".join(ui_text(text, color, bold) for text, color, bold in segments)


def terminal_rows_for_input(value: str, prompt_width: int, columns: int) -> int:
    """Number of terminal rows a prompt plus ``value`` occupies once wrapped."""
    rows = 1
    column = prompt_width
    for character in graphemes(value):
        if character == "\n":
            rows += 1
            column = 0
            continue
        width = terminal_character_width(character)
        if column > 0 and column + width > columns:
            rows += 1
            column = 0
        column += width
    return rows
