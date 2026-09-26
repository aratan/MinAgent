"""Streaming Markdown rendering for the terminal.

Model output arrives token by token, so the renderer is a small state machine:
it probes each line start to detect fences, tables, headings, lists, and quotes,
tracks inline emphasis and links as characters arrive, and pads each rendered
line to the assistant bubble's width.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from .terminal_text import (
    graphemes,
    safe_terminal_text,
    terminal_character_width,
    terminal_columns,
    terminal_text_width,
    truncate_terminal_text,
    wrap_text_line,
)

_TOKEN = re.compile(r"\x1b\[[0-9;]*m|[\s\S]")
_HEADING_PROBE = re.compile(r"^#{1,6}$")
_HEADING_SPACE = re.compile(r"^#{1,6} $")
_HEADING_TAB = re.compile(r"^#{1,6}\s")
_RULE_PROBE = re.compile(r"^-{3,}$")
_RULE_SPACE = re.compile(r"^-{3,} $")
_NUMBERED = re.compile(r"^(\d{1,5})\. $")
_SEQUENCE = re.compile(r"^\d{1,5}(\.)?$")
_TABLE_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
_WHITESPACE = re.compile(r"\s")
_IMAGE_LINK = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_CODE_SPAN = re.compile(r"(`+)(.*?)\1", re.DOTALL)
_STRONG = re.compile(r"(\*\*|__)(.*?)\1", re.DOTALL)
_EMPHASIS = re.compile(r"(?<!\w)([*_])([^*_]+)\1(?!\w)")
_COLLAPSE_WHITESPACE = re.compile(r"\s+")

_RESET = "\x1b[0m"


class MarkdownTerminalRenderer:
    """Incremental Markdown renderer writing into a bounded-width bubble."""

    def __init__(
        self,
        write_display: Callable[[str], None],
        get_use_color: Callable[[], bool],
        ui_colors: dict[str, Any],
        stdout: Any,
    ):
        self._write_display = write_display
        self._get_use_color = get_use_color
        self._ui_colors = ui_colors
        self._stdout = stdout
        self.at_line_start = True
        self.start_probe = ""
        self.table_line: str | None = None
        self.pending_table_header: dict[str, Any] | None = None
        self.table_mode = False
        self.table_widths: list[int] = []
        self.table_alignments: list[str] = []
        self.in_fence = False
        self.opening_fence = False
        self.closing_fence = False
        self.ignore_line = False
        self.fence_info = ""
        self.link_buffer: str | None = None
        self.pending_bang = False
        self.pending_marker = ""
        self.bold = False
        self.italic = False
        self.italic_marker = ""
        self.inline_code = False
        self.heading = False
        self.quote = False
        self.link_style = False
        self.last_visible_char = ""

    # ------------------------------------------------------------- styling

    def emit_text(self, value: str) -> None:
        self._write_display(safe_terminal_text(value))

    def sync_style(self) -> None:
        """Re-emit the current background and text attributes."""
        if not self._get_use_color():
            return
        background = self._ui_colors["assistantBackground"]
        codes = [f"48;2;{';'.join(str(part) for part in background)}"]
        if self.bold or self.heading:
            codes.append("1")
        if self.italic:
            codes.append("3")
        if self.link_style:
            codes.append("4")
        if self.heading or self.inline_code or self.link_style:
            color = self._ui_colors["cyan"]
        elif self.quote:
            color = self._ui_colors["muted"]
        elif self.in_fence:
            color = self._ui_colors["pale"]
        else:
            color = None
        if color:
            codes.append(f"38;2;{';'.join(str(part) for part in color)}")
        self._write_display(f"\x1b[0m\x1b[{';'.join(codes)}m" if codes else "\x1b[0m")

    # -------------------------------------------------------------- input

    def write(self, value: str) -> None:
        """Feed a chunk of model output through the state machine."""
        for character in str(value):
            if character == "\r":
                continue
            if character == "\n":
                self.newline()
                continue
            self.accept(character)

    def accept(self, character: str) -> None:
        if self.table_line is not None:
            self.table_line += character
            return
        if self.opening_fence:
            self.fence_info += character
            return
        if self.ignore_line:
            return
        if self.at_line_start:
            self.accept_line_start(character)
            return
        if self.in_fence:
            self.emit_text(character)
            return
        self.accept_inline(character)

    def accept_line_start(self, character: str) -> None:
        """Probe a line's opening characters to detect its block type."""
        if character != "|":
            self.finish_table_before_text()
        self.start_probe += character
        probe = self.start_probe

        if self.in_fence:
            if probe in ("`", "``"):
                return
            if probe == "```":
                self.in_fence = False
                self.closing_fence = True
                self.start_probe = ""
                self.at_line_start = False
                self.sync_style()
                return
            self.start_probe = ""
            self.at_line_start = False
            self.emit_text(f"  {probe}")
            return

        if probe == "|":
            self.table_line = probe
            self.start_probe = ""
            self.at_line_start = False
            return
        if re.fullmatch(r"`{1,2}", probe):
            return
        if probe == "```":
            self.in_fence = True
            self.opening_fence = True
            self.fence_info = ""
            self.start_probe = ""
            self.at_line_start = False
            self.sync_style()
            return
        if _HEADING_PROBE.match(probe):
            return
        if _HEADING_SPACE.match(probe):
            self.start_probe = ""
            self.at_line_start = False
            self.heading = True
            self.sync_style()
            return
        if _HEADING_TAB.match(probe):
            self.start_probe = ""
            self.at_line_start = False
            self.heading = True
            self.sync_style()
            self.accept_inline(_HEADING_TAB.sub("", probe, count=1))
            return
        if probe in ("-", "--", "---"):
            return
        if _RULE_SPACE.match(probe):
            self.start_probe = ""
            self.at_line_start = False
            self.ignore_line = True
            self.emit_text("─" * 36)
            return
        if probe in ("- ", "+ ", "* "):
            self.start_probe = ""
            self.at_line_start = False
            self.emit_text("• ")
            return
        if probe == "> ":
            self.start_probe = ""
            self.at_line_start = False
            self.quote = True
            self.sync_style()
            self.emit_text("│ ")
            return
        if _SEQUENCE.match(probe):
            return
        numbered = _NUMBERED.match(probe)
        if numbered:
            self.start_probe = ""
            self.at_line_start = False
            self.emit_text(f"{numbered.group(1)}. ")
            return
        if _RULE_PROBE.match(probe) or probe in ("+", "*", "-"):
            return

        self.start_probe = ""
        self.at_line_start = False
        for pending in probe:
            self.accept_inline(pending)

    def accept_inline(self, character: str) -> None:
        """Handle inline emphasis, code spans, and links as they stream in."""
        if self.pending_bang:
            self.pending_bang = False
            if character == "[":
                self.link_buffer = "!["
                return
            self.emit_text("!")
        if self.link_buffer is not None:
            self.link_buffer += character
            if len(self.link_buffer) > 4096:
                self.emit_text(self.link_buffer)
                self.link_buffer = None
                return
            target_start = 2 if self.link_buffer.startswith("![") else 1
            link_start = self.link_buffer.find("](", target_start)
            if link_start < 0:
                last_close = self.link_buffer.rfind("]")
                if last_close >= 0 and last_close < len(self.link_buffer) - 1:
                    self.emit_text(self.link_buffer)
                    self.link_buffer = None
                return
            if not self.link_buffer.endswith(")"):
                return
            is_image = self.link_buffer.startswith("![")
            label = self.link_buffer[target_start:link_start]
            url = self.link_buffer[link_start + 2:-1]
            self.link_buffer = None
            if is_image:
                self.emit_text("🖼 ")
            self.link_style = True
            self.sync_style()
            self.emit_text(label or url)
            self.link_style = False
            self.sync_style()
            if label and url:
                self.emit_text(f" ({url})")
            return
        if character == "!":
            self.pending_bang = True
            return
        if character == "[":
            self.link_buffer = "["
            return
        if self.inline_code:
            if character == "`":
                self.inline_code = False
                self.sync_style()
            else:
                self.emit_text(character)
            return
        if self.pending_marker:
            marker = self.pending_marker
            if character == marker[0] and len(marker) == 1:
                self.pending_marker += character
                return
            self.pending_marker = ""
            if len(marker) == 2:
                self.bold = not self.bold
                self.sync_style()
                self.accept_inline(character)
                return
            if self.italic and self.italic_marker == marker:
                self.italic = False
                self.italic_marker = ""
                self.sync_style()
            elif not _WHITESPACE.search(character) and (marker == "*" or not re.search(r"\w", self.last_visible_char or " ")):
                self.italic = True
                self.italic_marker = marker
                self.sync_style()
            else:
                self.emit_text(marker)
            self.accept_inline(character)
            return
        if character == "`":
            self.inline_code = True
            self.sync_style()
            return
        if character in ("*", "_"):
            self.pending_marker = character
            return
        self.emit_text(character)
        if not _WHITESPACE.search(character):
            self.last_visible_char = character

    def flush_inline_pending(self) -> None:
        """Emit any partially-typed inline marker at a line boundary."""
        if self.pending_bang:
            self.emit_text("!")
            self.pending_bang = False
        if self.link_buffer is not None:
            self.emit_text(self.link_buffer)
            self.link_buffer = None
        if self.pending_marker:
            if len(self.pending_marker) == 1 and self.italic and self.italic_marker == self.pending_marker:
                self.italic = False
                self.italic_marker = ""
                self.sync_style()
            elif len(self.pending_marker) == 2:
                self.bold = not self.bold
                self.sync_style()
            else:
                self.emit_text(self.pending_marker)
            self.pending_marker = ""

    def flush_start_probe(self) -> None:
        """Emit a line-start probe that turned out to be ordinary text."""
        if not self.start_probe:
            return
        if _RULE_PROBE.match(self.start_probe):
            self.emit_text("─" * 36)
            self.start_probe = ""
            self.at_line_start = False
            return
        pending = self.start_probe
        self.start_probe = ""
        self.at_line_start = False
        for character in pending:
            self.accept_inline(character)

    # -------------------------------------------------------------- tables

    def parse_table_cells(self, line: str) -> list[str]:
        """Split a table row into cells, honouring escaped pipes and code spans."""
        source = line.strip()
        if source.startswith("|"):
            source = source[1:]
        cells: list[str] = []
        cell = ""
        code_ticks = 0
        index = 0
        while index < len(source):
            character = source[index]
            if character == "\\" and index + 1 < len(source) and source[index + 1] == "|":
                cell += "|"
                index += 2
                continue
            if character == "`":
                run = 1
                while index + run < len(source) and source[index + run] == "`":
                    run += 1
                if code_ticks == 0:
                    code_ticks = run
                elif code_ticks == run:
                    code_ticks = 0
                cell += "`" * run
                index += run
                continue
            if character == "|" and code_ticks == 0:
                cells.append(cell.strip())
                cell = ""
                index += 1
                continue
            cell += character
            index += 1
        cells.append(cell.strip())
        if source.endswith("|") and cells and cells[-1] == "":
            cells.pop()
        return cells

    @staticmethod
    def _is_table_separator(cells: Sequence[str]) -> bool:
        return len(cells) > 1 and all(_TABLE_SEPARATOR_CELL.match(cell) for cell in cells)

    def format_table_cell(self, value: str) -> str:
        """Strip inline Markdown down to the text a table cell should show."""
        text = safe_terminal_text(value)
        text = _IMAGE_LINK.sub(r"\1", text)
        text = _LINK.sub(r"\1", text)
        text = _CODE_SPAN.sub(r"\2", text)
        text = _STRONG.sub(r"\2", text)
        text = _EMPHASIS.sub(r"\2", text)
        return _COLLAPSE_WHITESPACE.sub(" ", text).strip()

    def wrap_table_cell(self, value: str, width: int) -> list[str]:
        """Wrap one cell's text to ``width`` columns, breaking long words."""
        text = self.format_table_cell(value)
        if not text:
            return [""]
        lines: list[str] = []
        line = ""
        line_width = 0
        for word in re.split(r"\s+", text):
            word_width = terminal_text_width(word)
            if line and line_width + 1 + word_width <= width:
                line += f" {word}"
                line_width += 1 + word_width
                continue
            if line:
                lines.append(line)
                line = ""
                line_width = 0
            for character in graphemes(word):
                character_width = terminal_character_width(character)
                if line and line_width + character_width > width:
                    lines.append(line)
                    line = ""
                    line_width = 0
                line += character
                line_width += character_width
        if line:
            lines.append(line)
        return lines or [""]

    def initialize_table_columns(self, header: Sequence[str], separator: Sequence[str], columns: int) -> None:
        """Choose column widths that fit the terminal and follow the source."""
        column_count = max(len(header), len(separator))
        content_width = max(1, min(96, columns - 4))
        available_width = max(column_count, content_width - 3 * (column_count - 1))
        minimum_width = max(1, min(4, available_width // column_count))
        self.table_widths = [
            max(minimum_width, terminal_text_width(self.format_table_cell(header[index] if index < len(header) else "")))
            for index in range(column_count)
        ]
        total_width = sum(self.table_widths)
        while total_width > available_width:
            widest = -1
            for index, width in enumerate(self.table_widths):
                if width <= minimum_width:
                    continue
                if widest < 0 or width > self.table_widths[widest]:
                    widest = index
            if widest < 0:
                break
            self.table_widths[widest] -= 1
            total_width -= 1
        if total_width < available_width and self.table_widths:
            self.table_widths[-1] += available_width - total_width
        self.table_alignments = [
            ("center" if cell.startswith(":") and cell.endswith(":") else "right" if cell.endswith(":") else "left")
            for cell in (separator[index] if index < len(separator) else "" for index in range(column_count))
        ]

    def render_table_row(self, cells: Sequence[str], header: bool = False) -> None:
        """Render one table row, wrapping every cell to its column width."""
        wrapped = [self.wrap_table_cell(cells[index] if index < len(cells) else "", width)
                   for index, width in enumerate(self.table_widths)]
        line_count = max(1, *(len(lines) for lines in wrapped))
        for line_index in range(line_count):
            if line_index > 0:
                self.emit_text("\n")
            if header:
                self.bold = True
                self.heading = True
                self.sync_style()
            rendered = []
            for column_index, width in enumerate(self.table_widths):
                cell = wrapped[column_index][line_index] if line_index < len(wrapped[column_index]) else ""
                padding = max(0, width - terminal_text_width(cell))
                alignment = self.table_alignments[column_index]
                left_padding = padding if alignment == "right" else padding // 2 if alignment == "center" else 0
                rendered.append(f"{' ' * left_padding}{cell}{' ' * (padding - left_padding)}")
            self.emit_text(" │ ".join(rendered))
            if header:
                self.bold = False
                self.heading = False
                self.sync_style()

    def render_table_separator(self) -> None:
        self.emit_text("─┼─".join("─" * width for width in self.table_widths))

    def flush_pending_table_header(self, include_newline: bool = True) -> None:
        if not self.pending_table_header:
            return
        self.emit_text(self.pending_table_header["raw"])
        if include_newline:
            self.emit_text("\n")
        self.pending_table_header = None

    def finish_table_before_text(self) -> None:
        self.table_mode = False
        self.table_widths = []
        self.table_alignments = []
        self.flush_pending_table_header(True)

    def consume_table_line(self, line: str, include_newline: bool, columns: int) -> None:
        cells = self.parse_table_cells(line)
        if self.table_mode:
            if not self._is_table_separator(cells):
                self.render_table_row(cells)
                if include_newline:
                    self.emit_text("\n")
            return
        if self.pending_table_header and self._is_table_separator(cells):
            header = self.pending_table_header["cells"]
            self.pending_table_header = None
            self.initialize_table_columns(header, cells, columns)
            self.render_table_row(header, True)
            if include_newline:
                self.emit_text("\n")
            self.render_table_separator()
            if include_newline:
                self.emit_text("\n")
            self.table_mode = True
            return
        if self.pending_table_header:
            self.flush_pending_table_header(True)
        if self._is_table_separator(cells):
            self.emit_text(line)
            if include_newline:
                self.emit_text("\n")
            return
        self.pending_table_header = {"raw": line, "cells": cells}

    def newline(self) -> None:
        """End the current line and reset per-line state."""
        columns = terminal_columns(self._stdout)
        if self.table_line is not None:
            self.consume_table_line(self.table_line, True, columns)
            self.table_line = None
            self.at_line_start = True
            return
        self.table_mode = False
        self.table_widths = []
        self.table_alignments = []
        self.flush_pending_table_header(True)
        if self.opening_fence:
            info = self.fence_info.strip()
            self.emit_text(f"  Code{f' {info}' if info else ''}\n")
            self.opening_fence = False
            self.fence_info = ""
            self.at_line_start = True
            return
        if self.closing_fence:
            self.closing_fence = False
            self.emit_text("\n")
            self.at_line_start = True
            return
        if self.ignore_line:
            self.ignore_line = False
            self.emit_text("\n")
            self.at_line_start = True
            return
        self.flush_start_probe()
        self.flush_inline_pending()
        self.heading = False
        self.quote = False
        self.sync_style()
        self.emit_text("\n")
        self.at_line_start = True

    def end(self) -> None:
        """Flush every buffered fragment at the end of a response."""
        columns = terminal_columns(self._stdout)
        if self.table_line is not None:
            self.consume_table_line(self.table_line, False, columns)
            self.table_line = None
        self.table_mode = False
        self.flush_pending_table_header(False)
        if self.opening_fence:
            info = self.fence_info.strip()
            self.emit_text(f"  Code{f' {info}' if info else ''}")
            self.opening_fence = False
        self.flush_start_probe()
        self.flush_inline_pending()
        self.bold = False
        self.italic = False
        self.inline_code = False
        self.heading = False
        self.quote = False
        self.link_style = False
        self.sync_style()


class _AssistantBubbleWriter:
    """Writes into a titled, background-filled box, wrapping at its width."""

    def __init__(
        self,
        label: str,
        stdout: Any,
        get_use_color: Callable[[], bool],
        ui_colors: dict[str, Any],
        ui_text: Callable[..., str],
        ui_print: Callable[[str], None],
    ) -> None:
        columns = terminal_columns(stdout)
        self._stdout = stdout
        self._get_use_color = get_use_color
        self._ui_text = ui_text
        self._ui_print = ui_print
        # Never wider than the terminal: the box adds four cells of its own.
        self.content_width = max(1, min(96, columns - 4))
        background = ui_colors["assistantBackground"]
        self._background_style = (
            f"\x1b[48;2;{';'.join(str(part) for part in background)}m" if get_use_color() else ""
        )
        self._line_width = 0
        self._line_started = False
        self._active_style = ""
        self._pending_text = ""

        title_prefix = "╭─"
        # The label gives way to the border on a narrow terminal, never the reverse.
        title_text = f" {truncate_terminal_text(label, max(1, self.content_width - 2))} "
        title_fill = max(
            1,
            self.content_width + 4 - terminal_text_width(title_prefix) - terminal_text_width(title_text) - 1,
        )
        ui_print(
            f"{ui_text(title_prefix, 'cyan')}"
            f"{ui_text(title_text, 'magenta', True)}"
            f"{ui_text('─' * title_fill + '╮', 'cyan')}"
        )

    def _start_line(self) -> None:
        self._stdout.write(f"{self._ui_text('│', 'cyan')}{self._background_style} ")
        if self._active_style:
            self._stdout.write(self._active_style)
        self._line_started = True

    def _finish_line(self) -> None:
        if not self._line_started:
            self._start_line()
        padding = max(0, self.content_width - self._line_width) + 1
        if self._get_use_color():
            self._stdout.write(f"\x1b[0m{self._background_style}{' ' * padding}")
        else:
            self._stdout.write(" " * padding)
        self._stdout.write(f"{self._ui_text('│', 'cyan')}\n")
        self._line_width = 0
        self._line_started = False

    def _emit_cluster(self, cluster: str) -> None:
        character_width = terminal_character_width(cluster)
        if self._line_width > 0 and self._line_width + character_width > self.content_width:
            self._finish_line()
        if not self._line_started:
            self._start_line()
        self._stdout.write(cluster)
        self._line_width += character_width

    def _flush_pending_text(self) -> None:
        if self._pending_text:
            self._emit_cluster(self._pending_text)
        self._pending_text = ""

    def write(self, value: str) -> None:
        """Write styled output, holding back the last cluster for joining."""
        for token in _TOKEN.findall(str(value)):
            if token.startswith("\x1b["):
                self._flush_pending_text()
                if token == _RESET:
                    self._active_style = ""
                    self._stdout.write(f"{token}{self._background_style}")
                else:
                    self._active_style = token
                    self._stdout.write(token)
                continue
            if token == "\n":
                self._flush_pending_text()
                self._finish_line()
                continue
            clusters = graphemes(self._pending_text + token)
            for cluster in clusters[:-1]:
                self._emit_cluster(cluster)
            self._pending_text = clusters[-1] if clusters else ""

    def close(self, status: str = "complete") -> None:
        self._flush_pending_text()
        if self._line_started:
            self._finish_line()
        footer_prefix = "╰─"
        footer_text = f" {truncate_terminal_text(status, max(1, self.content_width - 2))} "
        footer_fill = max(
            1,
            self.content_width + 4 - terminal_text_width(footer_prefix) - terminal_text_width(footer_text) - 1,
        )
        self._ui_print(
            f"{self._ui_text(footer_prefix, 'cyan')}"
            f"{self._ui_text(footer_text, 'muted')}"
            f"{self._ui_text('─' * footer_fill + '╯', 'cyan')}"
        )


class TerminalRendering:
    """Factory for the streaming outputs the conversation loop writes to."""

    def __init__(
        self,
        stdout: Any,
        get_use_color: Callable[[], bool],
        ui_colors: dict[str, Any],
        ui_text: Callable[..., str],
        ui_print: Callable[[str], None],
        print: Callable[[str], None],
    ) -> None:
        self._stdout = stdout
        self._get_use_color = get_use_color
        self._ui_colors = ui_colors
        self._ui_text = ui_text
        self._ui_print = ui_print
        self._print = print

    def _renderer(self, write_display: Callable[[str], None]) -> MarkdownTerminalRenderer:
        return MarkdownTerminalRenderer(write_display, self._get_use_color, self._ui_colors, self._stdout)

    def create_streaming_output(self, label: str) -> _StreamingOutput:
        return _StreamingOutput(label, self)

    def create_reasoning_streaming_output(self) -> _ReasoningStreamingOutput:
        return _ReasoningStreamingOutput(
            self._stdout, self._ui_text, self._print, terminal_columns(self._stdout)
        )

    def create_assistant_bubble_writer(self, label: str) -> _AssistantBubbleWriter:
        return _AssistantBubbleWriter(
            label, self._stdout, self._get_use_color, self._ui_colors, self._ui_text, self._ui_print
        )


class _StreamingOutput:
    """An assistant response bubble that opens on its first write."""

    def __init__(self, label: str, rendering: TerminalRendering) -> None:
        self._label = label
        self._rendering = rendering
        self._renderer: MarkdownTerminalRenderer | None = None
        self._bubble: _AssistantBubbleWriter | None = None
        self._opened = False
        self._wrote_output = False

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def has_output(self) -> bool:
        return self._wrote_output

    def write(self, chunk: str) -> None:
        if not self._opened:
            self._rendering._print("")
            self._bubble = self._rendering.create_assistant_bubble_writer(self._label)
            self._renderer = self._rendering._renderer(self._bubble.write)
            self._opened = True
        assert self._renderer is not None
        self._renderer.write(chunk)
        self._wrote_output = True

    def close(self, status: str = "complete") -> None:
        if not self._opened:
            return
        assert self._renderer is not None and self._bubble is not None
        self._renderer.end()
        self._bubble.close(status)
        self._opened = False


class _ReasoningStreamingOutput:
    """Muted gray progress text for the optional reasoning channel.

    Reasoning arrives as free text, so the app wraps it to the terminal width
    itself: a line the terminal wraps on its own cannot be measured, and it
    pushes whatever follows it out of place.
    """

    def __init__(
        self, stdout: Any, ui_text: Callable[..., str], print: Callable[[str], None], columns: int
    ) -> None:
        self._stdout = stdout
        self._ui_text = ui_text
        self._print = print
        self.width = max(1, columns)
        self._buffer = ""
        self._wrote_output = False

    def _emit_line(self, line: str) -> None:
        self._stdout.write(self._ui_text(f"{line}\n", "muted"))

    def _drain(self, final: bool) -> None:
        """Write every line the buffer already commits to, keeping the last one back.

        While streaming, a trailing word may still grow, so a line is only cut
        once the buffer no longer fits; at the end the rest is flushed as it is.
        """
        while self._buffer:
            newline = self._buffer.find("\n")
            if newline >= 0:
                head, self._buffer = self._buffer[:newline], self._buffer[newline + 1:]
                for line in wrap_text_line(head, self.width):
                    self._emit_line(line)
                continue
            content = self._buffer.rstrip()
            if not content:
                return
            lines = wrap_text_line(content, self.width)
            if final:
                for line in lines:
                    self._emit_line(line)
                self._buffer = ""
                return
            if len(lines) <= 1:
                return
            for line in lines[:-1]:
                self._emit_line(line)
            self._buffer = f"{lines[-1]}{self._buffer[len(content):]}"

    def write(self, chunk: str) -> None:
        text = safe_terminal_text(chunk)
        if not text:
            return
        if not self._wrote_output:
            self._print("")
            self._wrote_output = True
        self._buffer += text
        self._drain(False)

    def close(self) -> None:
        if self._wrote_output:
            self._drain(True)
        self._buffer = ""
        self._wrote_output = False


def create_terminal_rendering(
    stdout: Any,
    get_use_color: Callable[[], bool],
    ui_colors: dict[str, Any],
    ui_text: Callable[..., str],
    ui_print: Callable[[str], None],
    print: Callable[[str], None],
) -> TerminalRendering:
    """Build the terminal rendering factory used by the conversation loop."""
    return TerminalRendering(stdout, get_use_color, ui_colors, ui_text, ui_print, print)


__all__ = [
    "MarkdownTerminalRenderer",
    "TerminalRendering",
    "create_terminal_rendering",
]
