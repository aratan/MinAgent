"""MCP transport tests against a local Streamable HTTP server."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from minagent.errors import AgentError
from minagent.mcp import connect_mcp_servers, execute_mcp_tool, write_mcp_server


class _McpTestServer(ThreadingHTTPServer):
    """A throwaway HTTP server that answers MCP requests."""

    daemon_threads = True

    def __init__(self, mode: str) -> None:
        self.mode = mode
        super().__init__(("127.0.0.1", 0), _McpTestHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/mcp"


class _McpTestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence the test server
        return

    def _result_for(self, call: dict[str, Any]) -> Any:
        if call["method"] == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "serverInfo": {"name": "test", "version": "1"},
            }
            # One mode publishes instructions and the rest do not, because that
            # is the split that matters: most servers say nothing, which is why
            # the operator's own note has to reach the model as well.
            if self.server.mode == "instructed":  # type: ignore[attr-defined]
                result["instructions"] = "Call the snapshot tool before anything that needs a handle."
            return result
        if call["method"] == "tools/list":
            if self.server.mode == "many":  # type: ignore[attr-defined]
                return {
                    "tools": [
                        {"name": f"tool_{index}", "inputSchema": {"type": "object", "properties": {}}}
                        for index in range(33)
                    ]
                }
            if self.server.mode == "openai":  # type: ignore[attr-defined]
                # The shape a server speaks when it hands over the agent's own
                # schemas unchanged: the name sits under "function", so nothing
                # here is a valid MCP tool and all of them used to be dropped
                # with a warning that named nothing anybody could act on.
                return {
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": f"tool_{index}",
                                "description": "d",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                        for index in range(3)
                    ]
                }
            return {"tools": [{"name": "echo", "inputSchema": {"type": "object", "properties": {}}}]}
        return {"content": [{"type": "text", "text": "ok"}]}

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", "0"))
        call = json.loads(self.rfile.read(length))
        if call.get("id") is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        result = self._result_for(call)
        if self.server.mode == "sse":  # type: ignore[attr-defined]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Mcp-Session-Id", "test-session")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            payload = f"data: {json.dumps({'jsonrpc': '2.0', 'id': call['id'], 'result': result})}\n\n".encode()
            self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()
            # Keep the response open; the client must stop reading after the match.
            time.sleep(2)
            return
        body = json.dumps({"jsonrpc": "2.0", "id": call["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", "limit-test")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()


def _write_config(root, url: str) -> str:
    config_path = root / "mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"local": {"url": url}}}))
    return str(config_path)


def _write_config_with_instructions(root, url: str, instructions: str) -> str:
    config_path = root / "mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"local": {"url": url, "instructions": instructions}}}))
    return str(config_path)


def test_write_mcp_server_stores_a_script_and_registers_it(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    config_path = str(root / ".minagent" / "mcp.json")
    created = write_mcp_server(
        str(root),
        config_path,
        {
            "name": "Tareas Locales",
            "script": {"filename": "server.py", "content": "print('hola')"},
            "env": {"MODO": "test"},
        },
    )
    assert created["name"] == "tareas-locales"
    script = root / ".agents" / "mcp" / "tareas-locales" / "server.py"
    assert script.read_text() == "print('hola')\n"
    stored = json.loads((root / ".minagent" / "mcp.json").read_text())
    assert stored["mcpServers"]["tareas-locales"] == {
        "command": "python3",
        "args": [str(script)],
        "env": {"MODO": "test"},
        "cwd": str(root / ".agents" / "mcp" / "tareas-locales"),
    }


def test_write_mcp_server_keeps_the_servers_already_configured(tmp_path):
    root = tmp_path / "project"
    (root / ".minagent").mkdir(parents=True)
    config_path = root / ".minagent" / "mcp.json"
    config_path.write_text(json.dumps({"mcpServers": {"keep": {"url": "http://127.0.0.1:1/mcp"}}}))
    write_mcp_server(str(root), str(config_path), {"name": "nuevo", "command": "node", "args": ["{script}"]})
    stored = json.loads(config_path.read_text())
    assert stored["mcpServers"]["keep"] == {"url": "http://127.0.0.1:1/mcp"}
    assert stored["mcpServers"]["nuevo"] == {"command": "node", "args": ["{script}"]}
    with pytest.raises(AgentError, match="already configured"):
        write_mcp_server(str(root), str(config_path), {"name": "nuevo", "command": "node"})
    write_mcp_server(str(root), str(config_path), {"name": "nuevo", "command": "node", "overwrite": True})
    assert json.loads(config_path.read_text())["mcpServers"]["nuevo"] == {"command": "node"}


@pytest.mark.parametrize(
    "args, message",
    [
        ({"name": "!!!"}, "must contain letters"),
        ({"name": "ok", "script": {"filename": "../escape.py", "content": "x"}}, "one plain name"),
        ({"name": "ok", "script": {"filename": "server.exe", "content": "x"}}, "must end in one of"),
        ({"name": "ok", "script": {"filename": "server.py", "content": "   "}}, "non-empty string"),
        ({"name": "ok", "command": "node", "env": {"mal nombre": "x"}}, "environment names"),
        ({"name": "ok"}, "stdio server command"),
        ({"name": "ok", "command": "node", "overwrite": "yes"}, "overwrite must be"),
    ],
)
def test_write_mcp_server_rejects_invalid_input(tmp_path, args, message):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(AgentError, match=message):
        write_mcp_server(str(root), str(root / ".minagent" / "mcp.json"), args)
    assert not (root / ".minagent" / "mcp.json").exists()


async def test_mcp_http_accepts_a_response_event_while_the_sse_connection_stays_open(tmp_path):
    server = _McpTestServer("sse")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config_path = _write_config(tmp_path, server.url)
        connections = await asyncio.wait_for(connect_mcp_servers(config_path, str(tmp_path)), timeout=5)
        try:
            assert connections["warnings"] == []
            assert len(connections["tool_definitions"]) == 1
            name = connections["tool_definitions"][0]["function"]["name"]
            result = await execute_mcp_tool(name, {}, connections["tool_lookup"], False)
            assert "ok" in result["tool_text"]
        finally:
            await connections["close"]()
    finally:
        server.shutdown()
        server.server_close()


async def test_mcp_tool_discovery_keeps_the_first_32_tools_from_an_oversized_list(tmp_path):
    server = _McpTestServer("many")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config_path = _write_config(tmp_path, server.url)
        connections = await asyncio.wait_for(connect_mcp_servers(config_path, str(tmp_path)), timeout=5)
        try:
            assert len(connections["tool_definitions"]) == 32
            assert any("first 32" in warning for warning in connections["warnings"])
        finally:
            await connections["close"]()
    finally:
        server.shutdown()
        server.server_close()


async def test_a_server_that_speaks_another_dialect_is_named_in_one_line(tmp_path):
    """A whole group of tools disappearing is worth saying exactly why.

    The compute server in this project answered tools/list with the agent's own
    function-calling schemas, so all seven of its tools were dropped and the
    only trace was "an MCP tool with no valid name" - which names a symptom
    the operator cannot see, at a place they will not look. The warning has to
    name the dialect and the fix, and it has to be said once rather than once
    per lost tool.
    """
    server = _McpTestServer("openai")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config_path = _write_config(tmp_path, server.url)
        connections = await asyncio.wait_for(connect_mcp_servers(config_path, str(tmp_path)), timeout=5)
        try:
            assert connections["tool_definitions"] == []
            dialect = [w for w in connections["warnings"] if "OpenAI function shape" in w]
            assert len(dialect) == 1, f"expected one clear warning, got {connections['warnings']}"
            assert "3" in dialect[0] and "tools/list" in dialect[0]
        finally:
            await connections["close"]()
    finally:
        server.shutdown()
        server.server_close()


async def test_a_local_note_reaches_the_model_when_the_server_says_nothing(tmp_path):
    """The Playwright case: 25 schemas, no instructions, and nothing to drive them.

    A server's guidance comes from its own handshake, and most good ones
    publish none. Without the operator's note the model gets a list of tool
    names and has to work out the order of the calls from the names alone.
    """
    server = _McpTestServer("plain")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config_path = _write_config_with_instructions(
            tmp_path, server.url, "Navigate, then snapshot, then act on a ref from that snapshot."
        )
        connections = await asyncio.wait_for(connect_mcp_servers(config_path, str(tmp_path)), timeout=5)
        try:
            guidance = [entry for entry in connections["server_guidance"] if entry["server_name"] == "local"]
            assert len(guidance) == 1
            assert "Navigate, then snapshot, then act" in guidance[0]["instructions"]
        finally:
            await connections["close"]()
    finally:
        server.shutdown()
        server.server_close()


async def test_the_operator_note_comes_before_what_the_server_said(tmp_path):
    server = _McpTestServer("instructed")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config_path = _write_config_with_instructions(tmp_path, server.url, "Local note first.")
        connections = await asyncio.wait_for(connect_mcp_servers(config_path, str(tmp_path)), timeout=5)
        try:
            instructions = connections["server_guidance"][0]["instructions"]
            assert instructions.startswith("Local note first.")
            assert "Call the snapshot tool" in instructions
        finally:
            await connections["close"]()
    finally:
        server.shutdown()
        server.server_close()


async def test_a_server_with_no_instructions_of_either_kind_adds_no_guidance(tmp_path):
    """Nothing to say is not worth a prompt section."""
    server = _McpTestServer("plain")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config_path = _write_config(tmp_path, server.url)
        connections = await asyncio.wait_for(connect_mcp_servers(config_path, str(tmp_path)), timeout=5)
        try:
            assert connections["server_guidance"] == []
        finally:
            await connections["close"]()
    finally:
        server.shutdown()
        server.server_close()


def test_rewriting_a_server_keeps_the_note_written_for_it(tmp_path):
    """write_mcp_server replaces the whole entry, so a dropped key is lost text.

    The note is the only thing telling the model how to drive a server that
    publishes no instructions, so losing it on the next unrelated registration
    would be a silent regression nobody notices until the agent gets stuck.
    """
    root = tmp_path / "project"
    (root / ".minagent").mkdir(parents=True)
    config_path = root / ".minagent" / "mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"notas": {"url": "http://127.0.0.1:1/mcp", "instructions": "Usa el snapshot."}}})
    )
    write_mcp_server(str(root), str(config_path), {"name": "otro", "command": "node"})
    stored = json.loads(config_path.read_text())
    assert stored["mcpServers"]["notas"]["instructions"] == "Usa el snapshot."


def test_write_mcp_server_rejects_a_note_it_cannot_carry(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    with pytest.raises(AgentError, match="instructions must be a non-empty string"):
        write_mcp_server(str(root), str(root / ".minagent" / "mcp.json"), {"name": "ok", "command": "node", "instructions": "   "})
    with pytest.raises(AgentError, match="instructions exceed"):
        write_mcp_server(
            str(root),
            str(root / ".minagent" / "mcp.json"),
            {"name": "ok", "command": "node", "instructions": "x" * 9000},
        )
    assert not (root / ".minagent" / "mcp.json").exists()
