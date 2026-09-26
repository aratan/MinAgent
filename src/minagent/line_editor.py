"""A raw-mode line editor for the prompt.

Node's readline has no standard-library equivalent, so this provides the same
surface the agent needs: a ``line``/``cursor`` buffer, keypress listeners that
may suppress a key by renaming it, multiline redraw, bracketed paste, and an
async ``question`` that resolves when the user submits.

Raw mode is entered once for the whole session so ``Esc`` can still reach the
agent while a model response is streaming, not just while the prompt is idle.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import sys
import termios
import tty
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .terminal_text import terminal_columns, terminal_rows_for_input, terminal_text_width

# How long an unresolved escape prefix waits for the rest of its sequence before
# it is delivered as a plain Escape key.
ESCAPE_GRACE_SECONDS = 0.05

# ``termios.tcgetattr`` returns [iflag, oflag, cflag, lflag, ispeed, ospeed, cc]
# and the module exposes no index constants, exactly as the stdlib ``tty`` module
# indexes them itself.
_OFLAG = 1

_CSI_FINAL_NAMES = {
    "A": ("up", False), "B": ("down", False), "C": ("right", False), "D": ("left", False),
    "H": ("home", False), "F": ("end", False),
    "1~": ("home", False), "2~": ("insert", False), "3~": ("delete", False),
    "4~": ("end", False), "5~": ("pageup", False), "6~": ("pagedown", False),
    "7~": ("home", False), "8~": ("end", False),
    "200~": ("paste-start", False), "201~": ("paste-end", False),
}

_CONTROL_NAMES = {
    0x03: ("c", "interrupt"),
    0x04: ("d", "eof"),
    0x09: ("tab", "tab"),
    0x0A: ("enter", "enter"),
    0x0D: ("return", "return"),
    0x1A: ("z", "suspend"),
    0x7F: ("backspace", "backspace"),
}


class EditorClosed(Exception):
    """Raised by ``question`` when the editor is closed, e.g. on Ctrl+C."""


@dataclass
class CursorPosition:
    """Where the cursor sits relative to the start of the prompt."""

    rows: int
    cols: int


@dataclass
class _RawKey:
    """A decoded keypress."""

    character: str
    name: str
    ctrl: bool = False
    meta: bool = False
    sequence: str = ""

    def as_editor_key(self) -> Any:
        from .editor import Key

        return Key(name=self.name, ctrl=self.ctrl, meta=self.meta, sequence=self.sequence)


class LineEditor:
    """A single-line (but newline-capable) editor driven by raw terminal input."""

    def __init__(self, output: Any = None, input_file: Any = None) -> None:
        self._output = output or sys.stdout
        self._input = input_file or sys.stdin
        self.line = ""
        self.cursor = 0
        self.prev_rows = 0
        self.is_completion_enabled = True
        self.saw_key_press = True
        # Submitted inputs, recalled with the up/down arrows.
        self.history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""

        self._prompt = ""
        self._multiline = False
        self._capture_listeners: list[Callable[[str, Any], None]] = []
        self._listeners: list[Callable[[str, Any], None]] = []
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._pending = bytearray()
        self._pending_text = ""
        self._escape_grace: asyncio.TimerHandle | None = None
        self._saved_termios: Any = None
        self._reader_installed = False
        self._closed = False
        self._submit: asyncio.Future[str] | None = None
        self._on_interrupt: Callable[[], None] | None = None

    # ------------------------------------------------------------ listeners

    def prepend_keypress(self, listener: Callable[[str, Any], None]) -> None:
        """Register a listener that runs before every other listener."""
        self._capture_listeners.append(listener)

    def on_keypress(self, listener: Callable[[str, Any], None]) -> None:
        """Register a listener for keypresses the prompt has not handled."""
        self._listeners.append(listener)

    def remove_keypress(self, listener: Callable[[str, Any], None]) -> None:
        """Remove a listener registered with either ``on`` or ``prepend``."""
        for collection in (self._capture_listeners, self._listeners):
            if listener in collection:
                collection.remove(listener)

    def on_interrupt(self, callback: Callable[[], None]) -> None:
        """Called when the user presses Ctrl+C."""
        self._on_interrupt = callback

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Enter raw mode and start reading keys from stdin."""
        if self._closed:
            return
        try:
            descriptor = self._input.fileno()
        except (AttributeError, ValueError):
            return
        if self._saved_termios is None:
            try:
                self._saved_termios = termios.tcgetattr(descriptor)
                tty.setraw(descriptor)
                self._keep_output_post_processing(descriptor)
            except termios.error:
                self._saved_termios = None
        if not self._reader_installed:
            try:
                asyncio.get_running_loop().add_reader(descriptor, self._read_available)
                self._reader_installed = True
            except (RuntimeError, NotImplementedError, ValueError):
                self._reader_installed = False

    @staticmethod
    def _keep_output_post_processing(descriptor: int) -> None:
        """Keep turning ``\n`` into a carriage return plus line feed.

        ``tty.setraw`` clears ``OPOST``, which is right for keyboard input but
        wrong for this session: every writer here ends lines with a bare ``\n``,
        so without it the cursor stays where the previous line ended and the
        whole UI stairs to the right.
        """
        attributes = termios.tcgetattr(descriptor)
        attributes[_OFLAG] |= termios.OPOST | termios.ONLCR
        termios.tcsetattr(descriptor, termios.TCSAFLUSH, attributes)

    def close(self) -> None:
        """Restore the terminal to its original mode."""
        if self._closed:
            return
        self._closed = True
        descriptor = None
        try:
            descriptor = self._input.fileno()
        except (AttributeError, ValueError):
            pass
        self._cancel_escape_grace()
        self._pending_text = ""
        if self._reader_installed and descriptor is not None:
            try:
                asyncio.get_running_loop().remove_reader(descriptor)
            except (RuntimeError, ValueError):
                pass
            self._reader_installed = False
        if self._saved_termios is not None and descriptor is not None:
            try:
                termios.tcsetattr(descriptor, termios.TCSADRAIN, self._saved_termios)
            except termios.error:
                pass
            self._saved_termios = None
        if self._submit is not None and not self._submit.done():
            self._submit.cancel()

    # -------------------------------------------------------------- prompt

    def mark_multiline(self) -> None:
        """Record that the buffer gained a newline, so readline-like redraws apply."""
        self._multiline = True

    @property
    def columns(self) -> int:
        return terminal_columns(self._output)

    def _rows(self, text: str) -> int:
        return terminal_rows_for_input(text, terminal_text_width(self._prompt), self.columns)

    def get_cursor_pos(self) -> CursorPosition:
        """Cursor position counted from the first prompt row."""
        before = self.line[: self.cursor]
        rows = self._rows(before)
        last_line = before.split("\n")[-1]
        return CursorPosition(rows=rows - 1, cols=terminal_text_width(last_line))

    def prompt(self, preserve: bool = False) -> None:
        """Repaint the prompt and the current buffer, then restore the cursor."""
        columns = self.columns
        output = self._output
        if self.prev_rows > 0:
            output.write("\r")
            if self.prev_rows > 1:
                output.write(f"\x1b[{self.prev_rows - 1}A")
        output.write("\x1b[0J")
        output.write(self._prompt + self.line.replace("\r\n", "\n"))
        total_rows = terminal_rows_for_input(self.line, terminal_text_width(self._prompt), columns)
        self.prev_rows = total_rows
        cursor_rows = terminal_rows_for_input(self.line[: self.cursor], terminal_text_width(self._prompt), columns)
        delta = total_rows - cursor_rows
        column = terminal_text_width((self._prompt + self.line[: self.cursor]).split("\n")[-1])
        output.write("\r")
        if delta > 0:
            output.write(f"\x1b[{delta}A")
        output.write(f"\x1b[{column}C")

    async def question(self, prompt: str) -> str:
        """Display a prompt and resolve with the submitted text."""
        self._prompt = prompt
        self.line = ""
        self.cursor = 0
        self.prev_rows = 0
        self._multiline = False
        self._history_index = None
        self._history_draft = ""
        self.start()
        loop = asyncio.get_running_loop()
        self._submit = loop.create_future()
        self.prompt(True)
        try:
            return await self._submit
        except asyncio.CancelledError:
            # close() cancels the pending submit; surface that as a clean exit.
            if self._closed:
                raise EditorClosed from None
            raise
        finally:
            self._submit = None

    # ---------------------------------------------------------------- input

    def _read_available(self) -> None:
        try:
            data = os.read(self._input.fileno(), 4096)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            return
        if not data:
            return
        self._pending.extend(data)
        self._dispatch_available()

    def _dispatch_available(self) -> None:
        """Dispatch what the last read produced, holding an incomplete escape back.

        A terminal sends ``Esc`` on its own but arrow keys and friends as a longer
        sequence, so a prefix that cannot be decoded yet is kept until the rest of
        the sequence arrives. Nothing else would reach the client otherwise: a lone
        ``Esc`` has no trailing byte to complete it, which is how a running request
        is stopped.
        """
        text = self._pending_text + self._decoder.decode(bytes(self._pending))
        self._pending.clear()
        self._pending_text = ""
        if not text:
            return
        keys, remainder = self._parse_with_remainder(text)
        self._pending_text = remainder
        if remainder:
            self._arm_escape_grace()
        else:
            self._cancel_escape_grace()
        for key in keys:
            self._dispatch(key)

    def _arm_escape_grace(self) -> None:
        """Deliver a still-incomplete escape prefix once nothing more arrives."""
        self._cancel_escape_grace()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._escape_grace = loop.call_later(ESCAPE_GRACE_SECONDS, self._deliver_pending_escape)

    def _cancel_escape_grace(self) -> None:
        if self._escape_grace is not None:
            self._escape_grace.cancel()
            self._escape_grace = None

    def _deliver_pending_escape(self) -> None:
        """Deliver a prefix that never completed: the Escape itself, then its text."""
        self._escape_grace = None
        text = self._pending_text
        self._pending_text = ""
        if not text or self._closed:
            return
        if text.startswith("\x1b"):
            self._dispatch(_RawKey("", "escape", sequence="\x1b"))
            text = text[1:]
        for character in text:
            self._dispatch(_RawKey(character, character, sequence=character))

    def _parse(self, text: str) -> list[_RawKey]:
        """Turn decoded text into keypresses, holding partial escape sequences back."""
        keys, _remainder = self._parse_with_remainder(text)
        return keys

    def _parse_with_remainder(self, text: str) -> tuple[list[_RawKey], str]:
        """Parse keypresses and return whatever is still an incomplete escape."""
        keys: list[_RawKey] = []
        index = 0
        length = len(text)
        while index < length:
            character = text[index]
            if character == "\x1b":
                consumed, key = self._parse_escape(text, index)
                if consumed == 0:
                    return keys, text[index:]  # Wait for the rest of the sequence.
                keys.append(key)
                index += consumed
                continue
            index += 1
            control = _CONTROL_NAMES.get(ord(character))
            if control is not None:
                name, _label = control
                keys.append(
                    _RawKey(character, name, ctrl=character in ("\x03", "\x04", "\x1a"), sequence=character)
                )
                continue
            if character < " " or character == "\x7f":
                keys.append(_RawKey("", character, ctrl=True, sequence=character))
                continue
            keys.append(_RawKey(character, character, sequence=character))
        return keys, ""

    def _parse_escape(self, text: str, start: int) -> tuple[int, _RawKey]:
        """Decode one escape sequence; returns (0, key) when more input is needed.

        The returned length covers the whole sequence. Consuming one byte too few
        would leave the final byte to be dispatched as a typed character, so a CSI
        sequence runs from just after ``[`` through its final byte in ``@``-``~``,
        which is also what ``_CSI_FINAL_NAMES`` is keyed by (``A``, ``3~``, ``200~``).
        """
        rest = text[start + 1:]
        if not rest:
            return 0, _RawKey("", "escape", sequence="\x1b")
        head = rest[0]
        if head == "[":
            for offset, character in enumerate(rest[1:], start=1):
                if "@" <= character <= "~":
                    final = rest[1:offset + 1]
                    name = _CSI_FINAL_NAMES.get(final, ("", False))[0]
                    return len(final) + 2, _RawKey("", name or "escape", sequence="\x1b[" + final)
            return 0, _RawKey("", "escape", sequence="\x1b[")
        if head == "O" and len(rest) > 1:
            name = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}.get(
                rest[1], ""
            )
            return 3, _RawKey("", name or "escape", sequence="\x1bO" + rest[1])
        return 2, _RawKey(head, "escape", meta=True, sequence="\x1b" + head)

    def _dispatch(self, key: _RawKey) -> None:
        """Run listeners, then apply the default editing action if allowed."""
        editor_key = key.as_editor_key()
        for listener in self._capture_listeners:
            listener(key.character, editor_key)
        if self._submit is None:
            # No prompt is waiting, so a model turn is running. Editing here would
            # glue the keystrokes to the line that was just submitted and then drop
            # them when the next prompt clears the buffer; Escape still reaches the
            # capture listeners above, which is how a running request is stopped.
            return
        if editor_key.name != "unbound":
            for listener in self._listeners:
                listener(key.character, editor_key)
        if editor_key.name != "unbound":
            self._apply_default(key)
        if not self._closed:
            self.prompt(True)

    def _apply_default(self, key: _RawKey) -> None:
        name = key.name
        if name in ("paste-start", "paste-end"):
            return
        if name == "interrupt":
            if self._on_interrupt is not None:
                self._on_interrupt()
            return
        if name in ("return", "enter"):
            self._submit_line()
            return
        if name == "backspace":
            if self.cursor > 0:
                self.line = self.line[: self.cursor - 1] + self.line[self.cursor:]
                self.cursor -= 1
            return
        if name == "delete":
            if self.cursor < len(self.line):
                self.line = self.line[: self.cursor] + self.line[self.cursor + 1:]
            return
        if name == "left":
            self.cursor = max(0, self.cursor - 1)
            return
        if name == "right":
            self.cursor = min(len(self.line), self.cursor + 1)
            return
        if name == "home":
            self.cursor = 0
            return
        if name == "end":
            self.cursor = len(self.line)
            return
        if name in ("up", "down"):
            # Move between the buffer's own lines first; at the edge, recall history.
            upward = name == "up"
            if not self._move_line(upward):
                self._navigate_history(upward)
            return
        if key.ctrl:
            if name == "k":
                self.line = self.line[: self.cursor]
            elif name == "u":
                self.line = self.line[self.cursor:]
                self.cursor = 0
            elif name == "a":
                self.cursor = 0
            elif name == "e":
                self.cursor = len(self.line)
            return
        if key.character and not key.meta:
            self.line = self.line[: self.cursor] + key.character + self.line[self.cursor:]
            self.cursor += 1

    def _move_line(self, upward: bool) -> bool:
        """Move the cursor between the buffer's logical lines.

        Returns ``False`` when the buffer has no line that way, which lets the
        caller fall back to history navigation.
        """
        before = self.line[: self.cursor]
        segments = before.split("\n")
        column = len(segments[-1])
        line_index = len(segments) - 1
        target_index = line_index - 1 if upward else line_index + 1
        all_segments = self.line.split("\n")
        if target_index < 0 or target_index >= len(all_segments):
            return False
        offset = sum(len(segment) + 1 for segment in all_segments[:target_index])
        self.cursor = min(offset + column, offset + len(all_segments[target_index]))
        return True

    def _navigate_history(self, upward: bool) -> None:
        """Recall a previously submitted input, restoring the draft on the way down."""
        if not self.history:
            return
        if upward:
            if self._history_index is None:
                self._history_draft = self.line
                self._history_index = len(self.history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            else:
                return
        else:
            if self._history_index is None:
                return
            if self._history_index >= len(self.history) - 1:
                self._history_index = None
                self.line = self._history_draft
                self.cursor = len(self.line)
                return
            self._history_index += 1
        self.line = self.history[self._history_index]
        self.cursor = len(self.line)

    def _submit_line(self) -> None:
        """Resolve the pending question and clear the buffer for the next prompt.

        Leaving the submitted text in the buffer used to make it reappear in the
        following prompt, with anything typed during the model's response glued on
        to it, only to be discarded when that prompt started.
        """
        self._output.write("\n")
        future = self._submit
        self._submit = None
        submitted = self.line
        self.line = ""
        self.cursor = 0
        self._history_index = None
        self._history_draft = ""
        if submitted.strip() and (not self.history or self.history[-1] != submitted):
            self.history.append(submitted)
        if future is not None and not future.done():
            future.set_result(submitted)
