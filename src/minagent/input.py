"""Keyboard and mouse control on this machine, through ydotool.

pyautogui is the usual answer to "let an agent drive the desktop", and it is the
wrong one here. This session is Wayland, not X11: pyautogui speaks Xlib, so it
sees an empty display. ``xdotool`` fails the same way for a different reason -
it can only reach XWayland clients, which on a modern KDE desktop means it
cannot touch the native windows at all.

``ydotool`` writes to ``/dev/uinput`` instead, so the compositor cannot tell
which client produced an event. It therefore reaches every application,
Wayland or X11, which is the only thing that makes desktop automation
reliable on this desktop.

Two things about it shape this module. Its ``key`` command takes Linux keycodes,
not key names, so the names are translated here from a table of
``input-event-codes.h`` rather than handed over and hoped for: ``ydotool`` does
not fail loudly on a name it cannot read, it emits a delay and moves on, which
would leave the model believing a shortcut had been pressed when nothing
happened. And its ``click`` command has no wheel, so scrolling is expressed
with the keys a window scrolls with, because the alternative on Wayland is
reaching into the compositor for something the available tools do not offer.

Everything here is gated by the session's ``INPUT_ENABLED`` switch. Nothing in
this module decides whether it is allowed to act: that decision belongs to the
app, which refuses to build a controller when the feature is off.
"""

from __future__ import annotations

import asyncio
import shutil
from typing import Any

from .errors import AgentError

YDOTOOL_BINARY = "ydotool"

DEFAULT_TIMEOUT_SECONDS = 15
"""A uinput write is a local device write; anything slower than this is a hang."""

MAX_KEYS_PER_CALL = 32
MAX_TEXT_CHARS = 4_000
MAX_MOVE_PIXELS = 8_000
MAX_SCROLL_CLICKS = 500
"""Bounds one call, so a runaway coordinate cannot fling the pointer off screen."""

DOUBLE_CLICK_DELAY_MS = 80
"""Below the platform double-click threshold and above a perceptible pause."""

DEFAULT_CLICK_DELAY_MS = 25

BUTTON_CLICK_CODES = {"left": 0xC0, "right": 0xC1, "middle": 0xC2}
"""ydotool indexes buttons from zero and ORs in 0x40 for down, 0x80 for up."""

BUTTON_DOWN_CODES = {"left": 0x40, "right": 0x41, "middle": 0x42}
BUTTON_UP_CODES = {"left": 0x80, "right": 0x81, "middle": 0x82}

#: Wheel directions as the signed vertical count ``mousemove --wheel`` expects.
#: The sign is the whole scroll, so it is stored as the number that means it.
SCROLL_KEYS = {"down": -1, "up": 1, "top": -1, "bottom": 1}


def _letters() -> dict[str, int]:
    """a-z, which are NOT contiguous in the kernel's numbering.

    They follow the three physical rows: Q starts at 16, A at 30, and Z at 44.
    Treating them as one run from KEY_A would make 't' send KEY_N, which is
    the kind of bug that looks like a working click landing on the wrong thing.
    """
    rows = {16: "qwertyuiop", 30: "asdfghjkl", 44: "zxcvbnm"}
    return {
        letter: base + offset
        for base, letters in rows.items()
        for offset, letter in enumerate(letters)
    }


def _digits() -> dict[str, int]:
    """0-9, which start at KEY_1 and wrap: '1' is 2 and '0' is 11."""
    return {str(digit): 1 + digit for digit in range(1, 10)} | {"0": 11}


#: Key names to Linux keycodes, from ``linux/input-event-codes.h``.
#: Only what a desktop task actually needs; an unknown name is an error rather
#: than a silent no-op, because ydotool would accept it and do nothing.
KEYCODES: dict[str, int] = {
    **_letters(),
    **_digits(),
    "esc": 1, "escape": 1,
    "minus": 12, "dash": 12, "equal": 13, "equals": 13,
    "backspace": 14, "tab": 15,
    "leftbrace": 26, "bracketleft": 26, "rightbrace": 27, "bracketright": 27,
    "enter": 28, "return": 28,
    "ctrl": 29, "control": 29, "leftctrl": 29,
    "semicolon": 39, "apostrophe": 40, "quote": 40, "grave": 41, "backtick": 41,
    "shift": 42, "leftshift": 42,
    "backslash": 43,
    "rightshift": 54,
    "alt": 56, "leftalt": 56,
    "space": 57,
    "capslock": 58,
    "f1": 59, "f2": 60, "f3": 61, "f4": 62, "f5": 63,
    "f6": 64, "f7": 65, "f8": 66, "f9": 67, "f10": 68, "f11": 87, "f12": 88,
    "numlock": 69, "scrolllock": 70,
    "home": 102, "up": 103, "pageup": 104, "pgup": 104,
    "left": 105, "right": 106, "end": 107,
    "down": 108, "pagedown": 109, "pgdown": 109,
    "insert": 110, "delete": 111, "del": 111,
    "mute": 113, "volumedown": 114, "volumeup": 115,
    "super": 125, "meta": 125, "win": 125, "leftmeta": 125, "command": 125,
    "rightmeta": 126, "menu": 139, "compose": 127,
}


class InputClient:
    """Drives the keyboard and mouse through the local ydotool binary.

    The binary is resolved at construction so a missing install is reported as
    a missing install, rather than as a failure of whichever call ran first.
    """

    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        runner: Any = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._runner = runner
        self._binary = shutil.which(YDOTOOL_BINARY)

    def available(self) -> bool:
        """Whether the binary is present."""
        return self._binary is not None

    def _require(self) -> str:
        """The ydotool path, or an error naming how to install it."""
        if self._binary is None:
            raise AgentError(
                "ydotool is not installed. Install it with: sudo pacman -S ydotool"
            )
        return self._binary

    async def _run(self, arguments: list[str]) -> str:
        """Run one ydotool invocation and return its stderr, which is where it reports."""
        binary = self._require()
        if self._runner is not None:
            return str(self._runner(arguments))

        process = await asyncio.create_subprocess_exec(
            binary,
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, errors = await asyncio.wait_for(process.communicate(), timeout=self.timeout_seconds)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise AgentError(
                f"ydotool {arguments[0] if arguments else ''} did not answer in "
                f"{self.timeout_seconds}s. If the daemon is wedged, restart it: "
                "systemctl --user restart ydotoold"
            ) from None
        detail = errors.decode("utf-8", "replace").strip()
        if process.returncode != 0:
            raise AgentError(
                f"ydotool failed (exit {process.returncode}): {detail or 'no output'}. "
                "If this mentions the socket or ydotoold, the daemon is not running: "
                "systemctl --user status ydotoold"
            )
        return detail

    async def key(self, keys: str) -> str:
        """Press and release key combinations, e.g. ``ctrl+shift+t``.

        Modifiers are held for the whole combination and released in reverse
        order, which is what makes ``ctrl+c`` behave like a chord rather than
        three separate events.
        """
        names = _split_keys(keys)
        if not names:
            raise AgentError("press_keys requires at least one key name.")
        if len(names) > MAX_KEYS_PER_CALL:
            raise AgentError(
                f"press_keys takes at most {MAX_KEYS_PER_CALL} keys in one call, got {len(names)}."
            )
        codes = [keycode(name) for name in names]
        sequence: list[str] = []
        for code in codes:
            sequence.append(f"{code}:1")
        for code in reversed(codes):
            sequence.append(f"{code}:0")
        await self._run(["key", "--", *sequence])
        return f"Pressed {'+'.join(names)}."

    async def type_text(self, text: str) -> str:
        """Type a string as keystrokes."""
        if not text:
            raise AgentError("type_text requires some text.")
        if len(text) > MAX_TEXT_CHARS:
            raise AgentError(
                f"type_text takes at most {MAX_TEXT_CHARS} characters in one call, got {len(text)}."
            )
        await self._run(["type", "--key-delay", "12", "--", text])
        return f"Typed {len(text)} characters."

    async def move_mouse(self, x: int, y: int, relative: bool = False) -> str:
        """Move the pointer, to an absolute screen position by default.

        ydotool takes absolute coordinates directly, so no read-back and no
        accumulation of relative steps is involved: the pointer lands where it
        was told to, or the call fails. A relative move is available for nudging
        it, and is asked for explicitly rather than inferred from a missing
        coordinate, which is what made "one axis only" ambiguous.
        """
        if x is None or y is None:
            raise AgentError("move_mouse requires both x and y.")
        if abs(int(x)) > MAX_MOVE_PIXELS or abs(int(y)) > MAX_MOVE_PIXELS:
            raise AgentError(
                f"Coordinates must be within +/-{MAX_MOVE_PIXELS}, got ({x}, {y}). "
                "Screen coordinates match the screenshot's own size; a far larger number "
                "means the two are not in the same space."
            )
        if relative:
            await self._run(["mousemove", "--", str(int(x)), str(int(y))])
            return f"Moved the pointer by ({int(x)}, {int(y)})."
        await self._run(["mousemove", "--absolute", "--", str(int(x)), str(int(y))])
        return f"Moved the pointer to ({int(x)}, {int(y)})."

    async def click(self, button: str = "left", count: int = 1) -> str:
        """Click a mouse button once, or twice for a double click."""
        _check_button(button)
        if int(count) not in (1, 2):
            raise AgentError(f"click count must be 1 or 2, got {count}.")
        code = BUTTON_CLICK_CODES[button]
        if int(count) == 2:
            await self._run(["click", "--next-delay", str(DOUBLE_CLICK_DELAY_MS), "--", hex(code), hex(code)])
            return f"Double-clicked the {button} button."
        await self._run(["click", "--next-delay", str(DEFAULT_CLICK_DELAY_MS), "--", hex(code)])
        return f"Clicked the {button} button."

    async def scroll(self, direction: str = "down", amount: int = 1) -> str:
        """Turn the wheel, which scrolls whatever is under the pointer.

        ``mousemove --wheel`` takes a signed vertical count, and the negative
        sign has to be shielded from the option parser with ``--``: without it
        a scroll down is read as an unknown flag and the call fails, having
        scrolled nothing.
        """
        key = direction.strip().lower()
        if key not in SCROLL_KEYS:
            raise AgentError(
                f"Unknown scroll direction {direction!r}. Use one of: "
                f"{', '.join(sorted(SCROLL_KEYS))}."
            )
        if int(amount) < 1:
            raise AgentError(f"scroll amount must be at least 1, got {amount}.")
        if int(amount) > MAX_SCROLL_CLICKS:
            raise AgentError(f"scroll amount must be at most {MAX_SCROLL_CLICKS}, got {amount}.")
        clicks = int(amount) * (1 if SCROLL_KEYS[key] > 0 else -1)
        await self._run(["mousemove", "--wheel", "--", "0", str(clicks)])
        return f"Scrolled {key} by {int(amount)} wheel click(s)."

    async def mouse_button_down(self, button: str = "left") -> str:
        """Press a mouse button and hold it; pair with ``mouse_button_up``."""
        _check_button(button)
        await self._run(["click", "--", hex(BUTTON_DOWN_CODES[button])])
        return f"Held the {button} button down."

    async def mouse_button_up(self, button: str = "left") -> str:
        """Release a mouse button held by ``mouse_button_down``."""
        _check_button(button)
        await self._run(["click", "--", hex(BUTTON_UP_CODES[button])])
        return f"Released the {button} button."


def keycode(name: str) -> int:
    """Translate one key name to its Linux keycode, or say what is available."""
    cleaned = (name or "").strip().lower()
    if len(cleaned) == 1 and cleaned in KEYCODES:
        return KEYCODES[cleaned]
    if cleaned in KEYCODES:
        return KEYCODES[cleaned]
    raise AgentError(
        f"Unknown key {name!r}. Use a single character, or one of: "
        f"{', '.join(sorted(k for k in KEYCODES if len(k) > 2))}."
    )


def _check_button(button: str) -> None:
    """Reject a button name before it reaches the key table."""
    if button not in BUTTON_CLICK_CODES:
        raise AgentError(
            f"Unknown button {button!r}. Use one of: {', '.join(sorted(BUTTON_CLICK_CODES))}."
        )


def _split_keys(keys: str) -> list[str]:
    """Split ``ctrl+shift+t`` into its parts, dropping empties."""
    return [part.strip().lower() for part in (keys or "").split("+") if part.strip()]


def create_input_tools() -> list[dict[str, Any]]:
    """The tool schemas for keyboard and mouse control."""
    return [
        {
            "type": "function",
            "function": {
                "name": "press_keys",
                "description": (
                    "Press and release a key combination, e.g. 'ctrl+shift+t'. Accepts single "
                    "characters and names like ctrl, alt, shift, super (also accepted as meta, win "
                    "or command), enter, tab, escape, backspace, space, delete, home, end, "
                    "pageup, pagedown, arrow keys, f1-f12, and the volume keys. An unrecognised "
                    "name is rejected rather than ignored."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keys": {"type": "string", "description": "Combination like ctrl+alt+t"}
                    },
                    "required": ["keys"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "type_text",
                "description": (
                    "Type a string as keystrokes, for text going into the field that currently has "
                    "focus. It does not click first: focus it with a click, then type."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move_mouse",
                "description": (
                    "Move the pointer to an absolute screen position, the same coordinate space "
                    "as the screenshot. Set relative=true to nudge it by an offset instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Absolute x, or an offset when relative"},
                        "y": {"type": "integer", "description": "Absolute y, or an offset when relative"},
                        "relative": {"type": "boolean", "default": False},
                    },
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "click_mouse",
                "description": (
                    "Click the left, right, or middle button at the current pointer position. "
                    "count=2 is a double click, which is what opening a file or folder needs."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "button": {"type": "string", "enum": sorted(BUTTON_CLICK_CODES), "default": "left"},
                        "count": {"type": "integer", "enum": [1, 2], "default": 1},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scroll_screen",
                "description": (
                    "Turn the mouse wheel, which scrolls whatever is under the pointer. "
                    "direction is down, up, top, or bottom; amount is how many clicks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": sorted(SCROLL_KEYS),
                            "default": "down",
                        },
                        "amount": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SCROLL_CLICKS,
                            "default": 1,
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mouse_button_down",
                "description": (
                    "Hold a mouse button down, for a drag. Always pair it with mouse_button_up, or "
                    "the button stays held and every later click behaves as a drag."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "button": {"type": "string", "enum": sorted(BUTTON_CLICK_CODES), "default": "left"}
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mouse_button_up",
                "description": "Release a mouse button held by mouse_button_down.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "button": {"type": "string", "enum": sorted(BUTTON_CLICK_CODES), "default": "left"}
                    },
                },
            },
        },
    ]


INPUT_GUIDANCE = (
    "The desktop is driven through ydotool writing to /dev/uinput, so it reaches native Wayland "
    "windows, which xdotool and pyautogui cannot do here. Nothing on this desktop can ask a window "
    "what is under a coordinate: the pointer position is the only address there is, so take a "
    "screenshot to see the screen, move the pointer to the target, then click. A click that lands "
    "wrong is invisible otherwise. Click before typing: type_text goes to whatever has focus. Always "
    "pair mouse_button_down with mouse_button_up."
)
"""Sits with the input tools, because none of it is obvious from the schemas."""
