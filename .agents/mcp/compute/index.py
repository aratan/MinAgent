#!/usr/bin/env python3
"""Servidor MCP que expone el orquestador de GPU de MinAgent.

Existe para que el cálculo de GPU no dependa de que el cliente sea MinAgent:
cualquier cliente MCP -Claude Desktop, un IDE, otro agente- puede generar un
vídeo, música o voz con los mismos trabajos y la misma política de VRAM.

Está en Python y sin dependencias a propósito. El protocolo MCP sobre stdio es
JSON-RPC 2.0 con delimitación por líneas, así que implementarlo aquí son unas
pocascientas líneas y ni una dependencia más que instalar en un servidor que
simplemente lanza un proceso. La alternativa, el paquete `mcp`, obligaría a
tenerlo instalado en el entorno del servidor para poco más.

Delega la ejecución en `minagent.compute`, el mismo módulo que usan las
herramientas nativas del agente. Esa es la razón de no reimplementar aquí la
lógica de VRAM: hay un solo sitio donde vive la política, y un cliente MCP no
puede acabar con un presupuesto distinto al del agente.

La serialización de trabajos pesados también es la misma: el servidor toma el
mismo lock, de modo que un vídeo lanzado por MCP y una canción lanzada por el
agente no pueden competir por la tarjeta.

Uso en `.minagent/mcp.json`:
    {
      "mcpServers": {
        "compute": {
          "command": "python3",
          "args": [".agents/mcp/compute/index.py"],
          "cwd": "/ruta/al/proyecto"
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "2025-11-25"
SERVER_NAME = "minagent-compute"

# El paquete vive en src/, así que un checkout sin instalar necesita el path.
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from minagent.compute import (  # noqa: E402
    DEFAULT_JOB_TIMEOUT_SECONDS,
    DEFAULT_QUEUE_LIMIT,
    DEFAULT_VRAM_TOTAL_MIB,
    QUEUE_TOOL_NAME,
    RESULT_TOOL_NAME,
    VOICE_TIMEOUT_SECONDS,
    ComputeOrchestrator,
    create_compute_tools,
    format_heavy_result,
    format_speak_result,
    format_transcribe_result,
)
from minagent.config import (  # noqa: E402
    load_env_file,
    parse_ollama_unload_mode,
    parse_positive_integer,
)

MAX_MESSAGE_CHARS = 4 * 1024 * 1024
"""A rendered frame or a long transcript is not a 1 MB payload, but a runaway
generation should not be able to buffer without limit either."""


def _log(message: str) -> None:
    """Diagnostics go to stderr: stdout is the JSON-RPC channel."""
    print(message, file=sys.stderr, flush=True)


def _configured_orchestrator(root_directory: str, application_root: str) -> ComputeOrchestrator:
    """An orchestrator that obeys the project's ``.env``, not just its defaults.

    This was built as ``ComputeOrchestrator(root_directory=root_directory)`` and
    nothing else, so every compute setting silently fell back to its default. The
    one that mattered was ``COMPUTE_UNLOAD_OLLAMA``: with the default ``off`` this
    server never moves a resident Ollama model, so a heavy job that the native
    tool renders in 25 s is refused here for lack of VRAM. Same policy on both
    surfaces is the entire point of delegating to ``minagent.compute``, and a
    default was not that.

    Only the compute settings are read, through the same parsers ``Config`` uses,
    rather than through ``load_configuration``: that one demands ``OPENAI_MODEL``
    and every other setting of a full agent session, and a GPU server for other
    MCP clients has no business failing because a model id is absent.
    """
    environment: dict[str, str] = dict(os.environ)
    load_env_file(os.path.join(application_root, ".env"), environment)
    return ComputeOrchestrator(
        root_directory=root_directory,
        # The backends live in the project that owns the agent; the client that
        # connected here may be working somewhere else entirely.
        script_directories=(application_root, root_directory),
        vram_total_mib=parse_positive_integer(
            environment.get("COMPUTE_VRAM_TOTAL_MIB"), "COMPUTE_VRAM_TOTAL_MIB",
            DEFAULT_VRAM_TOTAL_MIB,
        ),
        job_timeout_seconds=parse_positive_integer(
            environment.get("COMPUTE_JOB_TIMEOUT_SECONDS"), "COMPUTE_JOB_TIMEOUT_SECONDS",
            DEFAULT_JOB_TIMEOUT_SECONDS,
        ),
        voice_timeout_seconds=parse_positive_integer(
            environment.get("COMPUTE_VOICE_TIMEOUT_SECONDS"), "COMPUTE_VOICE_TIMEOUT_SECONDS",
            VOICE_TIMEOUT_SECONDS,
        ),
        ollama_mode=parse_ollama_unload_mode(environment.get("COMPUTE_UNLOAD_OLLAMA")),
        queue_limit=parse_positive_integer(
            environment.get("COMPUTE_QUEUE_LIMIT"), "COMPUTE_QUEUE_LIMIT", DEFAULT_QUEUE_LIMIT
        ),
    )


class ComputeServer:
    """The MCP surface over the orchestrator."""

    def __init__(self, root_directory: str, application_root: str | None = None) -> None:
        self.orchestrator = _configured_orchestrator(
            root_directory, application_root or root_directory
        )

    def tools(self) -> list[dict[str, Any]]:
        """The same schemas the agent publishes, so both surfaces stay in step.

        Translated into MCP's shape on the way out. ``create_compute_tools``
        speaks the OpenAI function-calling dialect - ``{"type": "function",
        "function": {...}}`` - because that is what the agent's own request
        expects, while ``tools/list`` answers with a flat
        ``{name, description, inputSchema}``. Handing the first shape to a
        client that reads the second is not a cosmetic mismatch: every one of
        these tools is silently dropped, and the server looks like it works.
        """
        published: list[dict[str, Any]] = []
        for schema in create_compute_tools():
            function = schema.get("function") or {}
            published.append(
                {
                    "name": str(function.get("name") or ""),
                    "description": str(function.get("description") or ""),
                    "inputSchema": function.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return published

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one tool and return the text a model should read.

        Each branch delegates to the same orchestrator method the native agent
        tool uses. That is the point of this server: a video requested over MCP
        and one requested by the agent go through one VRAM check and one lock,
        so they cannot be scheduled as if the card had room for both.
        """
        orchestrator = self.orchestrator
        if name == "compute_status":
            return orchestrator.status_text()
        if name == "speak_text":
            return format_speak_result(
                await orchestrator.speak(
                    str(arguments.get("text", "")),
                    str(arguments.get("voice", "")),
                    float(arguments.get("speed", 1.0) or 1.0),
                )
            )
        if name == "transcribe_audio":
            path = str(arguments.get("path", ""))
            return format_transcribe_result(path, await orchestrator.transcribe(path))
        if name == "generate_video":
            record = await orchestrator.generate_video(
                str(arguments.get("prompt", "")),
                frames=int(arguments.get("frames", 49) or 49),
                steps=int(arguments.get("steps", 40) or 40),
                offload=str(arguments.get("offload", "group") or "group"),
                name=str(arguments.get("name", "")),
            )
            return format_heavy_result(record, record.output)
        if name == "generate_music":
            record = await orchestrator.generate_music(
                str(arguments.get("prompt", "")),
                seconds=int(arguments.get("seconds", 10) or 10),
                device=str(arguments.get("device", "") or "cuda"),
                name=str(arguments.get("name", "")),
            )
            return format_heavy_result(record, record.output)
        if name == QUEUE_TOOL_NAME:
            kind = str(arguments.get("kind", "")).strip().lower()
            prompt = str(arguments.get("prompt", ""))
            if kind == "video":
                job_id = orchestrator.submit_video(
                    prompt,
                    frames=int(arguments.get("frames", 49) or 49),
                    steps=int(arguments.get("steps", 40) or 40),
                    offload=str(arguments.get("offload", "sequential") or "sequential"),
                    name=str(arguments.get("name", "")),
                )
            elif kind == "music":
                job_id = orchestrator.submit_music(
                    prompt,
                    seconds=int(arguments.get("seconds", 10) or 10),
                    device=str(arguments.get("device", "") or "cuda"),
                    name=str(arguments.get("name", "")),
                )
            else:
                raise ValueError("kind must be 'video' or 'music'.")
            return (
                f"Queued {job_id} for {kind}. It renders in the background; the queue is "
                f"{orchestrator.queued} deep behind it. Poll compute_result with this id."
            )
        if name == RESULT_TOOL_NAME:
            return orchestrator.result_text(str(arguments.get("job_id", "")))
        raise ValueError(f"Unknown tool: {name}")


async def read_messages(server: ComputeServer) -> None:
    """Serve JSON-RPC requests until stdin closes."""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        if len(line) > MAX_MESSAGE_CHARS:
            _log("compute: message over the size limit, ignored")
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            _log(f"compute: bad JSON: {error}")
            await write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
            continue
        await dispatch(server, message)


async def dispatch(server: ComputeServer, message: Any) -> None:
    """Route one JSON-RPC message to its handler and answer it."""
    if not isinstance(message, dict):
        return
    identifier = message.get("id")
    method = str(message.get("method", ""))
    params = message.get("params") or {}

    if method == "initialize":
        await write(
            {
                "jsonrpc": "2.0",
                "id": identifier,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
                },
            }
        )
        return
    if method in {"notifications/initialized", "notifications/cancelled"}:
        return
    if method == "tools/list":
        await write({"jsonrpc": "2.0", "id": identifier, "result": {"tools": server.tools()}})
        return
    if method == "tools/call":
        name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        try:
            text = await server.call(name, arguments)
            await write(
                {
                    "jsonrpc": "2.0",
                    "id": identifier,
                    "result": {"content": [{"type": "text", "text": text}]},
                }
            )
        except Exception as error:  # noqa: BLE001 - reported to the client as text
            # A tool error is a result, not a protocol error: the model reads it
            # and can act on it, where a JSON-RPC error is opaque.
            await write(
                {
                    "jsonrpc": "2.0",
                    "id": identifier,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {error}"}],
                        "isError": True,
                    },
                }
            )
        return
    if identifier is not None:
        await write(
            {"jsonrpc": "2.0", "id": identifier, "error": {"code": -32601, "message": f"Unknown: {method}"}}
        )


async def write(payload: dict[str, Any]) -> None:
    """Write one JSON-RPC message, flushing so the client sees it immediately."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    application_root = os.environ.get("MINAGENT_ROOT") or str(_ROOT)
    if not (Path(application_root) / "pyproject.toml").is_file():
        application_root = os.getcwd()
    root = application_root
    try:
        server = ComputeServer(root, application_root)
    except Exception as error:  # noqa: BLE001
        _log(f"compute: cannot start: {error}")
        return 1
    asyncio.run(read_messages(server))
    return 0


if __name__ == "__main__":
    sys.exit(main())
