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
    ComputeOrchestrator,
    create_compute_tools,
    format_heavy_result,
    format_speak_result,
    format_transcribe_result,
)

MAX_MESSAGE_CHARS = 4 * 1024 * 1024
"""A rendered frame or a long transcript is not a 1 MB payload, but a runaway
generation should not be able to buffer without limit either."""


def _log(message: str) -> None:
    """Diagnostics go to stderr: stdout is the JSON-RPC channel."""
    print(message, file=sys.stderr, flush=True)


class ComputeServer:
    """The MCP surface over the orchestrator."""

    def __init__(self, root_directory: str) -> None:
        self.orchestrator = ComputeOrchestrator(root_directory=root_directory)

    def tools(self) -> list[dict[str, Any]]:
        """The same schemas the agent publishes, so both surfaces stay in step."""
        return create_compute_tools()

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
                seconds=int(arguments.get("seconds", 15) or 15),
                name=str(arguments.get("name", "")),
            )
            return format_heavy_result(record, record.output)
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
    root = os.environ.get("MINAGENT_ROOT") or str(_ROOT)
    if not (Path(root) / "pyproject.toml").is_file():
        root = os.getcwd()
    try:
        server = ComputeServer(root)
    except Exception as error:  # noqa: BLE001
        _log(f"compute: cannot start: {error}")
        return 1
    asyncio.run(read_messages(server))
    return 0


if __name__ == "__main__":
    sys.exit(main())
