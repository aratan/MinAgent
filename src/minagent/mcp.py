"""MCP server configuration, transports, and tool discovery.

Servers are declared in ``.minagent/mcp.json`` and reached over either local
stdio or Streamable HTTP. Every exposed tool is bounded - count, schema size,
description length, and result size - so a misbehaving server cannot flood the
model context.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat as stat_module
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from .errors import AgentError, as_agent_error, is_missing
from .jsutil import byte_length, json_stringify
from .processes import terminate_process_tree

MCP_PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18", MCP_PROTOCOL_VERSION}
REQUEST_TIMEOUT_MS = 7 * 60 * 1000

MAX_MCP_TEXT_RESULT_CHARS = 48_000
MAX_MCP_IMAGE_BYTES = 10 * 1024 * 1024
MAX_MCP_IMAGE_COUNT = 4
MAX_MCP_CONFIG_BYTES = 1024 * 1024
MAX_MCP_SERVERS = 32
MAX_MCP_TOOLS = 32
MAX_MCP_SCHEMA_BYTES = 8 * 1024
MAX_MCP_TOTAL_TOOL_BYTES = 64 * 1024
MAX_MCP_TOOL_DESCRIPTION_CHARS = 1_500
MAX_MCP_GUIDANCE_CHARS = 8 * 1024
MAX_MCP_SCRIPT_BYTES = 64 * 1024

# Where authored servers live inside MinAgent's project directory, next to the
# ``.minagent/mcp.json`` entry that points at them.
MCP_SERVER_DIRECTORY = os.path.join(".agents", "mcp")
MCP_SCRIPT_COMMANDS = {
    ".js": "node",
    ".mjs": "node",
    ".cjs": "node",
    ".ts": "node",
    ".py": "python3",
    ".sh": "sh",
}

_FRAME_BOUNDARY = re.compile(r"\r?\n\r?\n")
_SAFE_SCRIPT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BASE64 = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
_IMAGE_MIME = re.compile(r"^image/(?:png|jpeg|gif|webp)$", re.IGNORECASE)
_UNSAFE_FUNCTION_CHARS = re.compile(r"[^A-Za-z0-9_-]")


def empty_connections(warnings: Sequence[str] = ()) -> dict[str, Any]:
    """A no-op connection set used when MCP is disabled or unconfigured."""
    async def _close() -> None:
        return None

    return {
        "clients": [],
        "tool_definitions": [],
        "tool_lookup": {},
        "server_guidance": [],
        "warnings": list(warnings),
        "close": _close,
    }


def mcp_config_fingerprint(config_path: str) -> Any:
    """Cheap identity of the MCP configuration file, so it is only re-read after a change."""
    try:
        entry = os.lstat(config_path)
    except OSError:
        return None
    return (entry.st_ino, entry.st_size, entry.st_mtime_ns)


def _set_protocol_version(client: Any, version: Any) -> None:
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise AgentError(f"Server selected unsupported MCP protocol version: {version or 'missing'}")
    client.protocol_version = version


def _make_function_name(server_index: int, tool_index: int, server_name: str, tool_name: str) -> str:
    safe_names = _UNSAFE_FUNCTION_CHARS.sub("_", f"{server_name}_{tool_name}")
    return f"mcp_{server_index}_{tool_index}_{safe_names}"[:64]


class McpStdioClient:
    """Speaks JSON-RPC over a child process's stdin and stdout."""

    def __init__(
        self,
        server_name: str,
        command: str,
        args: Sequence[str],
        server_env: dict[str, str],
        cwd: str,
        timeout_ms: int = REQUEST_TIMEOUT_MS,
    ) -> None:
        self.server_name = server_name
        self.command = command
        self.args = list(args)
        self.server_env = dict(server_env)
        self.cwd = cwd
        self.timeout_ms = timeout_ms
        self.process: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.next_id = 1
        self.buffer = ""
        self.stderr_tail = ""
        self.closed = False
        self.failure: AgentError | None = None
        self.instructions = ""
        self.protocol_version: str | None = None

    async def connect(self) -> None:
        environment = {**os.environ, **self.server_env}
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            cwd=self.cwd,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert self.process.stdout is not None and self.process.stdin is not None
        asyncio.create_task(self._read_stdout())
        asyncio.create_task(self._read_stderr())
        asyncio.create_task(self._watch_exit())

        result = await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "MinAgent", "version": "1.0.0"},
            },
        )
        _set_protocol_version(self, (result or {}).get("protocolVersion"))
        instructions = (result or {}).get("instructions")
        self.instructions = instructions[:12_000] if isinstance(instructions, str) else ""
        self._notify("notifications/initialized")

    async def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while True:
                chunk = await self.process.stdout.read(65536)
                if not chunk:
                    return
                self._consume_stdout(chunk.decode("utf-8", errors="replace"))
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        try:
            while True:
                chunk = await self.process.stderr.read(65536)
                if not chunk:
                    return
                self.stderr_tail = f"{self.stderr_tail}{chunk.decode('utf-8', errors='replace')}"[-8000:]
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    async def _watch_exit(self) -> None:
        assert self.process is not None
        returncode = await self.process.wait()
        detail = self.stderr_tail.strip()
        self._fail(
            AgentError(
                f"Server exited ({returncode if returncode is not None else 'unknown status'})"
                + (f": {detail}" if detail else "")
            )
        )

    def _consume_stdout(self, chunk: str) -> None:
        self.buffer += chunk
        if len(self.buffer) > 16 * 1024 * 1024:
            self._fail(AgentError("MCP server sent an oversized stdio message."))
            self.buffer = ""
            return
        while "\n" in self.buffer:
            newline = self.buffer.index("\n")
            line = self.buffer[:newline].strip()
            self.buffer = self.buffer[newline + 1:]
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._fail(AgentError("MCP server wrote a non-JSON line to stdout."))
                return
            self._receive(message)

    def _receive(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        identifier = message.get("id")
        if identifier is None:
            return
        future = self.pending.pop(str(identifier), None)
        if future is None or future.done():
            return
        error = message.get("error")
        if error:
            future.set_exception(AgentError((error or {}).get("message") or "MCP request failed."))
        else:
            future.set_result(message.get("result"))

    async def _request(self, method: str, params: Any = None, timeout_ms: int | None = None) -> Any:
        timeout_ms = self.timeout_ms if timeout_ms is None else timeout_ms
        if self.failure is not None:
            raise self.failure
        if self.process is None or self.process.stdin is None or self.process.returncode is not None:
            raise AgentError("MCP server is not available.")
        identifier = self.next_id
        self.next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.pending[str(identifier)] = future
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": identifier, "method": method}
        if params is not None:
            payload["params"] = params
        try:
            self.process.stdin.write((json_stringify(payload) + "\n").encode("utf-8"))
            await self.process.stdin.drain()
        except Exception as error:
            self.pending.pop(str(identifier), None)
            raise AgentError(f"Could not write to the MCP server: {error}") from error
        try:
            return await asyncio.wait_for(future, timeout=timeout_ms / 1000)
        except TimeoutError:
            self.pending.pop(str(identifier), None)
            raise AgentError(f"MCP request timed out: {method}") from None

    def _notify(self, method: str, params: Any = None) -> None:
        if self.process is None or self.process.stdin is None:
            return
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        try:
            self.process.stdin.write((json_stringify(payload) + "\n").encode("utf-8"))
        except Exception as error:
            self._fail(as_agent_error(error))

    def _fail(self, error: AgentError) -> None:
        if self.failure is not None:
            return
        self.failure = error
        for future in list(self.pending.values()):
            if not future.done():
                future.set_exception(error)
        self.pending.clear()

    async def list_tools(self) -> dict[str, Any]:
        return await _list_tools(self)

    async def call_tool(self, name: str, arguments: Any) -> Any:
        return await self._request("tools/call", {"name": name, "arguments": arguments})

    async def close(self) -> None:
        if self.closed or self.process is None:
            return
        self.closed = True
        if self.process.stdin is not None and self.process.returncode is None:
            try:
                self.process.stdin.close()
            except Exception:
                pass
        if self.process.returncode is None:
            try:
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
            except TimeoutError:
                terminate_process_tree(self.process)
                await self.process.wait()


class McpHttpClient:
    """Speaks JSON-RPC over Streamable HTTP, including server-sent events."""

    def __init__(self, server_name: str, config: dict[str, Any], timeout_ms: int = REQUEST_TIMEOUT_MS) -> None:
        self.server_name = server_name
        self.timeout_ms = timeout_ms
        self.url = str(config.get("url"))
        parsed = urlsplit(self.url)
        if parsed.scheme not in ("http", "https"):
            raise AgentError("MCP server URLs must use HTTP or HTTPS.")
        self.server_headers = config.get("headers", {})
        if (
            not isinstance(self.server_headers, dict)
            or any(not isinstance(value, str) for value in self.server_headers.values())
        ):
            raise AgentError("HTTP server headers must be an object containing string values.")
        self.protocol_version: str | None = None
        self.session_id: str | None = None
        self.next_id = 1
        self.instructions = ""
        self._client = httpx.AsyncClient(timeout=self.timeout_ms / 1000, follow_redirects=False)

    def _headers(self, base: dict[str, str]) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            **self.server_headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        return headers

    async def connect(self) -> None:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "MinAgent", "version": "1.0.0"},
            },
        )
        _set_protocol_version(self, (result or {}).get("protocolVersion"))
        instructions = (result or {}).get("instructions")
        self.instructions = instructions[:12_000] if isinstance(instructions, str) else ""
        await self._notify("notifications/initialized")

    async def _post(self, message: dict[str, Any], identifier: Any = None) -> Any:
        async with self._client.stream(
            "POST", self.url, headers=self._headers({}), content=json_stringify(message).encode("utf-8")
        ) as response:
            session_id = response.headers.get("mcp-session-id")
            if session_id:
                self.session_id = session_id
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                detail = body[:1000]
                try:
                    parsed = json.loads(body)
                    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
                        detail = parsed["error"].get("message") or detail
                except json.JSONDecodeError:
                    pass
                raise AgentError(f"MCP HTTP {response.status_code}: {detail or response.reason_phrase}")
            if identifier is None:
                await response.aread()
                return None
            content_type = response.headers.get("content-type", "").lower()
            if "text/event-stream" in content_type:
                result = await self._read_sse_response(response, identifier, 16 * 1024 * 1024)
            else:
                body = (await response.aread()).decode("utf-8", errors="replace")
                if not body.strip():
                    raise AgentError(f"MCP server returned an empty response for {message['method']}.")
                result = json.loads(body)
            if not result:
                raise AgentError(f"MCP server did not return a response for {message['method']}.")
            if str(result.get("id")) != str(identifier):
                raise AgentError(f"MCP server returned the wrong response ID for {message['method']}.")
            if result.get("error"):
                raise AgentError((result["error"] or {}).get("message") or "MCP request failed.")
            return result.get("result")

    async def _request(self, method: str, params: Any = None) -> Any:
        identifier = self.next_id
        self.next_id += 1
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": identifier, "method": method}
        if params is not None:
            payload["params"] = params
        return await self._post(payload, identifier)

    async def _notify(self, method: str, params: Any = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params
        await self._post(payload)

    @staticmethod
    async def _read_sse_response(response: httpx.Response, identifier: Any, max_bytes: int) -> Any:
        """Read events until the matching response id arrives, then stop."""
        buffer = ""
        total_bytes = 0
        try:
            async for chunk in response.aiter_bytes():
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise AgentError(f"MCP HTTP response exceeds {max_bytes} bytes.")
                buffer += chunk.decode("utf-8", errors="replace")
                while True:
                    match = _FRAME_BOUNDARY.search(buffer)
                    if match is None:
                        break
                    frame = buffer[: match.start()]
                    buffer = buffer[match.end():]
                    data = "\n".join(
                        line[5:].lstrip() for line in re.split(r"\r?\n", frame) if line.startswith("data:")
                    )
                    if not data.strip():
                        continue
                    try:
                        message = json.loads(data)
                    except json.JSONDecodeError:
                        raise AgentError("MCP server sent an invalid JSON-RPC event.") from None
                    if str((message or {}).get("id")) == str(identifier):
                        return message
            raise AgentError("MCP server closed the event stream before returning the requested response.")
        finally:
            await response.aclose()

    async def list_tools(self) -> dict[str, Any]:
        return await _list_tools(self)

    async def call_tool(self, name: str, arguments: Any) -> Any:
        return await self._request("tools/call", {"name": name, "arguments": arguments})

    async def close(self) -> None:
        if not self.session_id:
            await self._client.aclose()
            return
        try:
            headers = dict(self.server_headers)
            headers["Mcp-Session-Id"] = self.session_id
            if self.protocol_version:
                headers["MCP-Protocol-Version"] = self.protocol_version
            await self._client.delete(self.url, headers=headers, timeout=3.0)
        except Exception:
            pass
        finally:
            await self._client.aclose()


async def _list_tools(client: Any) -> dict[str, Any]:
    """Page through ``tools/list`` up to the tool budget."""
    tools: list[Any] = []
    cursor: Any = None
    for _page in range(100):
        result = await client._request("tools/list", {"cursor": cursor} if cursor else {})
        page_tools = result.get("tools") if isinstance(result, dict) else None
        page_tools = page_tools if isinstance(page_tools, list) else []
        remaining = MAX_MCP_TOOLS - len(tools)
        tools.extend(page_tools[:remaining])
        if len(page_tools) > remaining or (len(tools) >= MAX_MCP_TOOLS and result.get("nextCursor")):
            return {"tools": tools, "truncated": True}
        if not result.get("nextCursor"):
            return {"tools": tools, "truncated": False}
        cursor = result["nextCursor"]
    raise AgentError("MCP tools/list exceeded the 100-page limit.")


def _create_client(server_name: str, config: Any, default_cwd: str, timeout_ms: int = REQUEST_TIMEOUT_MS) -> Any:
    """Build a stdio or HTTP client from one ``mcpServers`` entry."""
    if not isinstance(config, dict):
        raise AgentError("Server settings must be a JSON object.")
    url = config.get("url")
    if isinstance(url, str) and url.strip():
        return McpHttpClient(server_name, config, timeout_ms)
    command = config.get("command")
    if not isinstance(command, str) or not command.strip():
        raise AgentError("Set either a stdio server command or an HTTP server URL.")
    args = config.get("args", [])
    if not isinstance(args, list) or any(not isinstance(value, str) for value in args):
        raise AgentError("Server args must be an array of strings.")
    env = config.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(value, str) for value in env.values()):
        raise AgentError("Server env must be an object containing string values.")
    if config.get("cwd") is not None and not isinstance(config["cwd"], str):
        raise AgentError("Server cwd must be a string.")
    cwd = os.path.normpath(os.path.join(default_cwd, config["cwd"])) if config.get("cwd") else default_cwd
    return McpStdioClient(server_name, command, args, env, cwd, timeout_ms)


async def connect_mcp_servers(
    config_path: str, default_cwd: str, timeout_ms: int = REQUEST_TIMEOUT_MS
) -> dict[str, Any]:
    """Connect every configured server and collect a bounded set of tools."""
    config: Any = None
    try:
        directory_mode = os.lstat(os.path.dirname(config_path)).st_mode
        if stat_module.S_ISLNK(directory_mode) or not stat_module.S_ISDIR(directory_mode):
            raise AgentError("MCP configuration directory must be a regular directory.")
        file_entry = os.lstat(config_path)
        if stat_module.S_ISLNK(file_entry.st_mode) or not stat_module.S_ISREG(file_entry.st_mode) or file_entry.st_nlink > 1:
            raise AgentError("MCP configuration must be a regular, unlinked file.")
        if file_entry.st_size > MAX_MCP_CONFIG_BYTES:
            raise AgentError(f"MCP configuration exceeds {MAX_MCP_CONFIG_BYTES} bytes.")
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except OSError as error:
        if is_missing(error):
            return empty_connections()
        return empty_connections([f"Could not load MCP configuration {config_path}: {error}"])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return empty_connections([f"Could not load MCP configuration {config_path}: {error}"])
    except AgentError as error:
        return empty_connections([f"Could not load MCP configuration {config_path}: {error}"])

    if not isinstance(config, dict) or not isinstance(config.get("mcpServers"), dict):
        return empty_connections([f'MCP configuration must contain an object named "mcpServers": {config_path}'])

    clients: list[Any] = []
    warnings: list[str] = []
    entries = list(config["mcpServers"].items())
    if len(entries) > MAX_MCP_SERVERS:
        return empty_connections([f"MCP configuration contains more than {MAX_MCP_SERVERS} servers."])

    tool_definitions: list[dict[str, Any]] = []
    tool_lookup: dict[str, dict[str, Any]] = {}
    server_guidance: list[dict[str, str]] = []
    total_definition_bytes = 0
    total_tool_budget_reached = False

    for server_index, (server_name, server_config) in enumerate(entries):
        if total_tool_budget_reached:
            break
        if len(tool_definitions) >= MAX_MCP_TOOLS:
            warnings.append(f"Ignoring additional MCP tools after the {MAX_MCP_TOOLS}-tool limit.")
            break
        client = None
        try:
            client = _create_client(server_name, server_config, default_cwd, timeout_ms)
            await client.connect()
            tool_list = await client.list_tools()
            remote_tools = tool_list["tools"]
            if tool_list["truncated"]:
                warnings.append(
                    f'MCP server "{server_name}" returned more than {MAX_MCP_TOOLS} tools; '
                    f"only the first {MAX_MCP_TOOLS} were considered."
                )
            clients.append(client)
            if client.instructions:
                server_guidance.append({"server_name": server_name, "instructions": client.instructions})

            # A server that answers tools/list in another dialect is worth one
            # clear line rather than one per tool: the agent silently loses the
            # whole group otherwise, and "no valid name" names nothing that can
            # be acted on. Checked once, before the loop, so the warning says
            # how many tools went and why.
            if remote_tools and all(
                isinstance(item, dict) and isinstance(item.get("function"), dict) and not item.get("name")
                for item in remote_tools
            ):
                warnings.append(
                    f'Ignoring all {len(remote_tools)} tools from server "{server_name}": it answers '
                    f"tools/list in the OpenAI function shape (type/function) instead of MCP's "
                    f"(name, description, inputSchema). The server's logic is fine; its tools() has to "
                    f"flatten each definition."
                )
                continue

            for tool_index, remote_tool in enumerate(remote_tools):
                if len(tool_definitions) >= MAX_MCP_TOOLS:
                    warnings.append(f"Ignoring additional MCP tools after the {MAX_MCP_TOOLS}-tool limit.")
                    total_tool_budget_reached = True
                    break
                if (
                    not isinstance(remote_tool, dict)
                    or not isinstance(remote_tool.get("name"), str)
                    or not remote_tool["name"].strip()
                ):
                    warnings.append(f'Ignoring an MCP tool with no valid name from server "{server_name}".')
                    continue
                function_name = _make_function_name(server_index, tool_index, server_name, remote_tool["name"])
                raw_schema = remote_tool.get("inputSchema")
                parameters = (
                    raw_schema
                    if isinstance(raw_schema, dict)
                    else {"type": "object", "properties": {}}
                )
                try:
                    properties = parameters.get("properties")
                    if parameters.get("type") != "object" or (
                        properties is not None
                        and (not isinstance(properties, dict))
                    ):
                        raise AgentError("inputSchema must describe an object with object properties.")
                    required = parameters.get("required")
                    if required is not None and (
                        not isinstance(required, list) or any(not isinstance(item, str) for item in required)
                    ):
                        raise AgentError("inputSchema.required must be an array of strings.")
                    if byte_length(json_stringify(parameters)) > MAX_MCP_SCHEMA_BYTES:
                        raise AgentError(f"inputSchema exceeds {MAX_MCP_SCHEMA_BYTES} bytes.")
                except AgentError as error:
                    warnings.append(
                        f'Ignoring MCP tool "{remote_tool["name"]}" from "{server_name}": {error.message}'
                    )
                    continue

                description = ". ".join(
                    part
                    for part in (remote_tool.get("title"), remote_tool.get("description") or f"Call the {remote_tool['name']} MCP tool.")
                    if part
                )
                description = _CONTROL_CHARS.sub(" ", description)[:MAX_MCP_TOOL_DESCRIPTION_CHARS]
                safe_server = _CONTROL_CHARS.sub(" ", str(server_name))[:128]
                definition = {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": f"{description} (MCP server: {safe_server}.)",
                        "parameters": parameters,
                    },
                }
                definition_bytes = byte_length(json_stringify(definition))
                if total_definition_bytes + definition_bytes > MAX_MCP_TOTAL_TOOL_BYTES:
                    warnings.append("Ignoring additional MCP tools after reaching the 64 KiB combined schema limit.")
                    total_tool_budget_reached = True
                    break
                tool_definitions.append(definition)
                total_definition_bytes += definition_bytes
                tool_lookup[function_name] = {
                    "client": client,
                    "server_name": server_name,
                    "remote_tool_name": remote_tool["name"],
                }
        except Exception as error:
            if client is not None:
                await client.close()
            warnings.append(f'MCP server "{server_name}" could not be connected: {error}')

    async def close() -> None:
        await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)

    return {
        "clients": clients,
        "tool_definitions": tool_definitions,
        "tool_lookup": tool_lookup,
        "server_guidance": server_guidance,
        "warnings": warnings,
        "close": close,
    }


def _contained_path(root: str, *parts: str) -> str:
    """Resolve ``parts`` under ``root``, refusing anything that escapes it."""
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, *parts))
    relative = os.path.relpath(target, root_real)
    if relative.startswith(f"..{os.sep}") or relative == ".." or os.path.isabs(relative):
        raise AgentError("MCP server files must stay inside the project directory.")
    return target


def _ensure_directory(path: str, root: str) -> None:
    """Create ``path`` under ``root``, refusing symlinked or non-directory components."""
    relative = os.path.relpath(path, os.path.realpath(root))
    current = os.path.realpath(root)
    for segment in [part for part in relative.split(os.sep) if part]:
        current = os.path.join(current, segment)
        try:
            entry = os.lstat(current)
        except OSError as error:
            if not is_missing(error):
                raise AgentError(f"Could not inspect {current}: {error}") from error
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            except OSError as error:
                raise AgentError(f"Could not create {current}: {error}") from error
            entry = os.lstat(current)
        if stat_module.S_ISLNK(entry.st_mode) or not stat_module.S_ISDIR(entry.st_mode):
            raise AgentError("MCP server paths must be regular directories.")


def _write_text_atomic(path: str, text: str) -> None:
    """Write ``text`` through a temporary file, so no reader sees a half-written file."""
    directory = os.path.dirname(path)
    temporary = os.path.join(
        directory, f".{os.path.basename(path)}.minagent-{os.getpid()}-{os.urandom(8).hex()}.tmp"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _stored_server_entry(config: Any) -> dict[str, Any]:
    """Validate one ``mcpServers`` entry and return exactly what will be stored."""
    if not isinstance(config, dict):
        raise AgentError("Server settings must be a JSON object.")
    entry: dict[str, Any] = {}
    url = config.get("url")
    command = config.get("command")
    if isinstance(url, str) and url.strip():
        entry["url"] = url.strip()
        if urlsplit(entry["url"]).scheme not in ("http", "https"):
            raise AgentError("MCP server URLs must use HTTP or HTTPS.")
    elif isinstance(command, str) and command.strip():
        entry["command"] = command.strip()
    else:
        raise AgentError("Set either a stdio server command or an HTTP server URL.")
    args = config.get("args")
    if args is not None:
        if not isinstance(args, list) or any(not isinstance(value, str) for value in args):
            raise AgentError("Server args must be an array of strings.")
        if args:
            entry["args"] = list(args)
    env = config.get("env")
    if env is not None:
        if not isinstance(env, dict) or any(
            not isinstance(name, str) or not _ENV_NAME.match(name) or not isinstance(value, str)
            for name, value in env.items()
        ):
            raise AgentError("Server env must map environment names to string values.")
        if env:
            entry["env"] = dict(env)
    headers = config.get("headers")
    if headers is not None:
        if not isinstance(headers, dict) or any(
            not isinstance(name, str) or not isinstance(value, str) for name, value in headers.items()
        ):
            raise AgentError("Server headers must map names to string values.")
        if headers:
            entry["headers"] = dict(headers)
    cwd = config.get("cwd")
    if cwd is not None:
        if not isinstance(cwd, str) or not cwd.strip():
            raise AgentError("Server cwd must be a non-empty string.")
        entry["cwd"] = cwd
    return entry


def _read_mcp_config(config_path: str) -> dict[str, Any]:
    """Load the configuration for updating, treating a missing file as no servers."""
    try:
        file_entry = os.lstat(config_path)
    except OSError as error:
        if is_missing(error):
            return {"mcpServers": {}}
        raise AgentError(f"Could not inspect the MCP configuration: {error}") from error
    if stat_module.S_ISLNK(file_entry.st_mode) or not stat_module.S_ISREG(file_entry.st_mode) or file_entry.st_nlink > 1:
        raise AgentError("MCP configuration must be a regular, unlinked file.")
    if file_entry.st_size > MAX_MCP_CONFIG_BYTES:
        raise AgentError(f"MCP configuration exceeds {MAX_MCP_CONFIG_BYTES} bytes.")
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AgentError(f"Could not read the MCP configuration: {error}") from error
    if not isinstance(config, dict):
        raise AgentError("MCP configuration must be a JSON object.")
    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise AgentError('MCP configuration must contain an object named "mcpServers".')
    config["mcpServers"] = servers
    return config


def _write_mcp_config(config_path: str, config: dict[str, Any], application_root: str) -> None:
    """Replace the configuration atomically, keeping every other server it declares."""
    directory = os.path.dirname(config_path)
    _ensure_directory(directory, application_root)
    try:
        previous_mode = stat_module.S_IMODE(os.lstat(config_path).st_mode)
    except OSError:
        previous_mode = None
    _write_text_atomic(config_path, f"{json.dumps(config, indent=2, ensure_ascii=False)}\n")
    if previous_mode is not None:
        os.chmod(config_path, previous_mode)


def write_mcp_server(application_root: str, config_path: str, args: dict[str, Any]) -> dict[str, Any]:
    """Author a local MCP server script and register it in ``.minagent/mcp.json``.

    The script goes to ``<project>/.agents/mcp/<name>/`` and the entry keeps every server the
    configuration already declared. Both files are written atomically, an existing
    server of the same name is only replaced with ``overwrite``, and everything stays
    inside the project directory.
    """
    from .skills import slugify_skill_name

    name = slugify_skill_name(str(args.get("name") or ""))
    if not name:
        raise AgentError("MCP server name must contain letters or digits.")
    overwrite = args.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise AgentError("overwrite must be true or false.")

    script = args.get("script")
    filename = ""
    content: Any = ""
    if script is not None:
        if not isinstance(script, dict):
            raise AgentError("script must be an object with filename and content.")
        filename = str(script.get("filename") or "").strip()
        content = script.get("content")
        if not _SAFE_SCRIPT_NAME.match(filename) or os.path.basename(filename) != filename:
            raise AgentError(
                "script filename must be one plain name using letters, digits, dots, dashes, and underscores."
            )
        extension = os.path.splitext(filename)[1].lower()
        if extension not in MCP_SCRIPT_COMMANDS:
            raise AgentError(f"script filename must end in one of: {', '.join(sorted(MCP_SCRIPT_COMMANDS))}.")
        if not isinstance(content, str) or not content.strip():
            raise AgentError("script content must be a non-empty string.")
        if len(content.encode("utf-8")) > MAX_MCP_SCRIPT_BYTES:
            raise AgentError(f"script content exceeds {MAX_MCP_SCRIPT_BYTES} bytes.")

    supplied = {key: args[key] for key in ("url", "command", "args", "env", "headers", "cwd") if args.get(key) is not None}
    if script is not None and not supplied.get("url") and not supplied.get("command"):
        supplied["command"] = MCP_SCRIPT_COMMANDS[os.path.splitext(filename)[1].lower()]
        supplied.setdefault("args", ["{script}"])
    entry = _stored_server_entry(supplied)

    script_path = ""
    if script is not None:
        directory = _contained_path(application_root, MCP_SERVER_DIRECTORY, name)
        script_path = _contained_path(application_root, MCP_SERVER_DIRECTORY, name, filename)
        entry["args"] = [argument.replace("{script}", script_path) for argument in entry.get("args", [])]
        entry.setdefault("cwd", directory)

    config = _read_mcp_config(config_path)
    servers = config["mcpServers"]
    if name in servers and not overwrite:
        raise AgentError(f'MCP server "{name}" is already configured; pass overwrite to replace it.')
    if name not in servers and len(servers) >= MAX_MCP_SERVERS:
        raise AgentError(f"MCP configuration already lists the maximum of {MAX_MCP_SERVERS} servers.")
    servers[name] = entry

    if script is not None:
        _ensure_directory(os.path.dirname(script_path), application_root)
        _write_text_atomic(script_path, content if content.endswith("\n") else f"{content}\n")
    _write_mcp_config(config_path, config, application_root)
    return {"name": name, "entry": entry, "script_path": script_path, "config_path": config_path}


def create_mcp_authoring_tools() -> list[dict[str, Any]]:
    """Tool definitions for authoring local MCP servers, exposed when MCP is enabled."""
    return [
        {
            "type": "function",
            "function": {
                "name": "write_mcp_server",
                "description": (
                    "Create a local MCP server: an optional script written to .agents/mcp/<name>/ plus its entry in "
                    ".minagent/mcp.json, so its tools become usable in this session. Use it when the user needs a "
                    "capability no current tool provides. The user approves the write first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Lowercase name with hyphens; it names the server and its folder",
                        },
                        "script": {
                            "type": "object",
                            "description": "Optional stdio server program to write under .agents/mcp/<name>/",
                            "properties": {
                                "filename": {
                                    "type": "string",
                                    "description": "Plain file name ending in .js, .mjs, .cjs, .ts, .py, or .sh",
                                },
                                "content": {"type": "string", "description": "Complete file contents"},
                            },
                            "required": ["filename", "content"],
                        },
                        "command": {
                            "type": "string",
                            "description": "Program to run; defaults to the interpreter for the script extension",
                        },
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Arguments; {script} is replaced with the written script's absolute path",
                        },
                        "env": {
                            "type": "object",
                            "description": "Extra environment variables as string values",
                        },
                        "cwd": {"type": "string", "description": "Working directory for the server"},
                        "url": {"type": "string", "description": "Streamable HTTP server URL instead of a command"},
                        "overwrite": {
                            "type": "boolean",
                            "description": "Replace a server of the same name; defaults to false",
                        },
                    },
                    "required": ["name"],
                },
            },
        }
    ]


def format_mcp_server_context(server_name: str, instructions: str) -> str:
    """Render one server's instructions, sanitised and bounded.

    Server instructions are untrusted text that ends up in the system prompt, so
    they are stripped of control characters and cut to the shared guidance
    budget. Each server is rendered on its own because a server's guidance now
    travels with that server's capability, not in one blob for all of them.
    """
    safe_name = _CONTROL_CHARS.sub(" ", str(server_name))[:128]
    heading = f"### {json_stringify(safe_name)}\n"
    remaining = MAX_MCP_GUIDANCE_CHARS - len(heading)
    if remaining <= 0:
        return ""
    safe_instructions = _UNSAFE_CHARS.sub(" ", str(instructions))
    if len(safe_instructions) > remaining:
        marker = "\n[truncated]"
        content_limit = max(0, remaining - len(marker))
        return heading + safe_instructions[:content_limit] + (marker if remaining > len(marker) else "")
    return heading + safe_instructions


async def execute_mcp_tool(
    function_name: str,
    args: dict[str, Any],
    tool_lookup: dict[str, dict[str, Any]],
    image_enabled: bool,
) -> dict[str, Any]:
    """Call one MCP tool and normalise its result for the model."""
    entry = tool_lookup.get(function_name)
    if entry is None:
        raise AgentError(f"MCP tool is not available: {function_name}")
    result = await entry["client"].call_tool(entry["remote_tool_name"], args)

    import base64

    text_parts: list[str] = []
    images: list[dict[str, Any]] = []
    content = result.get("content") if isinstance(result, dict) else None
    for item in content if isinstance(content, list) else []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text" and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
        elif item_type == "resource" and isinstance((item.get("resource") or {}).get("text"), str):
            text_parts.append(item["resource"]["text"])
        elif item_type == "resource_link":
            text_parts.append(f"Resource link: {item.get('name') or item.get('uri')}\n{item.get('uri')}")
        elif item_type == "image":
            raw_data = item.get("data")
            encoded = raw_data if isinstance(raw_data, str) else ""
            encoded_limit = -(-MAX_MCP_IMAGE_BYTES // 3) * 4
            valid_base64 = (
                0 < len(encoded) <= encoded_limit and len(encoded) % 4 == 0 and bool(_BASE64.match(encoded))
            )
            can_include = (
                image_enabled
                and len(images) < MAX_MCP_IMAGE_COUNT
                and bool(_IMAGE_MIME.match(item.get("mimeType") or ""))
                and valid_base64
            )
            data = base64.b64decode(encoded) if can_include else b""
            if can_include and 0 < len(data) <= MAX_MCP_IMAGE_BYTES:
                images.append(
                    {
                        "mime_type": item["mimeType"],
                        "data": base64.b64encode(data).decode("ascii"),
                        "path": f"MCP server {entry['server_name']}",
                    }
                )
            else:
                if not image_enabled:
                    reason = "image input is disabled for the model"
                elif len(images) >= MAX_MCP_IMAGE_COUNT:
                    reason = f"the {MAX_MCP_IMAGE_COUNT}-image limit was reached"
                else:
                    reason = "its format, encoding, or size is unsupported"
                text_parts.append(f"[An MCP image was omitted because {reason}.]")
        elif item_type:
            text_parts.append(f"[MCP returned content of type {str(item_type)[:80]}.]")

    structured = result.get("structuredContent") if isinstance(result, dict) else None
    if isinstance(structured, dict):
        text_parts.append(f"Structured result:\n{json_stringify(structured)}")

    tool_text = "\n\n".join(text_parts).strip() or "The MCP tool returned no text content."
    if isinstance(result, dict) and result.get("isError"):
        tool_text = f"MCP tool reported an error.\n{tool_text}"
    if len(tool_text) > MAX_MCP_TEXT_RESULT_CHARS:
        tool_text = f"{tool_text[:MAX_MCP_TEXT_RESULT_CHARS]}\n[Tool result truncated.]"
    return {
        "tool_text": f"[MCP {entry['server_name']}/{entry['remote_tool_name']}]\n{tool_text}",
        "images": images,
    }
