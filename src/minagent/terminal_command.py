"""Running shell commands on the model's behalf.

Terminal access is off by default, requires per-command approval in ``ask``
mode, runs with the user's own permissions, and always reports how the command
ended - including when it was stopped for running too long or printing too much.
"""

from __future__ import annotations

import asyncio
import signal as signal_module
from collections.abc import Callable, Sequence
from typing import Any

from .errors import AgentError
from .jsutil import json_stringify
from .processes import terminate_process_tree
from .terminal_text import (
    StyledSegment,
    render_styled_line,
    safe_terminal_text,
    terminal_columns,
    wrap_styled_segments,
)

_OUTPUT_LIMIT = 64 * 1024
_TIMEOUT_SECONDS = 7 * 60


def _describe_exit(returncode: int | None) -> str:
    """Render a process exit status the way the tool result reads best."""
    if returncode is None:
        return "terminated (unknown signal)"
    if returncode < 0:
        try:
            name = signal_module.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        return f"terminated ({name})"
    return str(returncode)


async def run_terminal_command(
    args: dict[str, Any],
    *,
    terminal_mode: str,
    terminal_command_shell: str,
    root_directory: str,
    interactive_terminal: Any,
    print: Callable[[str], None],
    ui_print: Callable[[str], None],
    ui_text: Callable[..., str],
    ui_print_wrapped: Callable[[Sequence[StyledSegment]], None] | None = None,
    timeout_seconds: int = _TIMEOUT_SECONDS,
) -> str:
    """Run one shell command in the workspace and return its combined output."""

    def emit(segments: Sequence[StyledSegment]) -> None:
        """Print one block, wrapped to the terminal width when it is known."""
        if ui_print_wrapped is not None:
            ui_print_wrapped(segments)
            return
        for line in wrap_styled_segments(segments, terminal_columns()):
            ui_print(render_styled_line(line, ui_text))

    if terminal_mode == "off":
        raise AgentError("Terminal access is disabled by TERMINAL_MODE.")
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        raise AgentError("command must be a non-empty string.")
    if len(command) > 20_000:
        raise AgentError("command is longer than the 20,000 character limit.")

    if terminal_mode == "ask":
        if interactive_terminal is None:
            raise AgentError("Cannot request permission outside the interactive terminal.")
        print("")
        emit((("Terminal permission requested", "warning", True),))
        emit(((json_stringify(command), "pale", False),))
        answer: str = await interactive_terminal.question("Allow this command? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return "Permission denied by the user. The command was not executed."

    process = await asyncio.create_subprocess_exec(
        terminal_command_shell,
        "-c",
        command,
        cwd=root_directory,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )

    chunks: list[bytes] = []
    state = {"bytes": 0, "truncated": False, "stopped": False}

    def stop() -> None:
        if state["stopped"]:
            return
        state["stopped"] = True
        terminate_process_tree(process)

    def append(data: bytes) -> None:
        remaining = _OUTPUT_LIMIT - state["bytes"]
        if remaining <= 0:
            state["truncated"] = True
            stop()
            return
        keep = data[:remaining]
        chunks.append(keep)
        state["bytes"] += len(keep)
        if len(keep) < len(data):
            state["truncated"] = True
            stop()

    async def drain(stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while True:
            data = await stream.read(65536)
            if not data:
                return
            append(data)

    timed_out = False
    try:
        try:
            await asyncio.wait_for(
                asyncio.gather(drain(process.stdout), drain(process.stderr)),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            timed_out = True
            stop()
        returncode = await process.wait()
    finally:
        if process.returncode is None:
            stop()
            await process.wait()

    output = b"".join(chunks).decode("utf-8", errors="replace")
    notes: list[str] = []
    if timed_out:
        notes.append(f"Command stopped after {timeout_seconds} seconds.")
    if state["truncated"]:
        notes.append("Output truncated at 64 KiB; command was stopped.")
    return safe_terminal_text(
        "\n".join(
            part for part in [f"Exit code: {_describe_exit(returncode)}", output, *notes] if part
        )
    )
