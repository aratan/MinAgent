"""End-to-end test: a clock question through a real streaming endpoint and ``date``.

Unlike the unit tests that stub ``call_chat_completions``, this one exercises the
whole path: the HTTP client, the SSE parser, the app's tool dispatch, the real
shell, and the follow-up request that carries the command output back to the
endpoint.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from minagent.app import MinAgent
from minagent.config import load_configuration


class _FakeOutput:
    """A stdout stand-in that records what was written."""

    def __init__(self, columns: int = 100) -> None:
        self.columns = columns
        self.chunks: list[str] = []

    def write(self, value: str) -> None:
        self.chunks.append(value)

    def isatty(self) -> bool:
        return False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def _stream(*events: dict[str, Any]) -> bytes:
    """Encode events as one SSE body that ends the stream."""
    frames = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return f"{frames}data: [DONE]\n\n".encode()


class _ClockEndpoint(ThreadingHTTPServer):
    """A throwaway OpenAI-compatible endpoint that asks for ``date``, then answers with it."""

    daemon_threads = True

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        super().__init__(("127.0.0.1", 0), _ClockEndpointHandler)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


class _ClockEndpointHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence the test server
        return

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        endpoint: _ClockEndpoint = self.server  # type: ignore[assignment]
        endpoint.requests.append(body)
        tool_result = next(
            (
                message.get("content")
                for message in reversed(body.get("messages", []))
                if message.get("role") == "tool"
            ),
            None,
        )
        if tool_result is None:
            payload = _stream(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_clock",
                                        "type": "function",
                                        "function": {
                                            "name": "run_terminal",
                                            "arguments": json.dumps({"command": "date"}),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            )
        else:
            stamped = str(tool_result).strip().splitlines()[-1]
            payload = _stream(
                {"choices": [{"delta": {"content": f"La hora del sistema es {stamped}"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.wfile.flush()


async def test_a_clock_question_runs_date_against_a_streaming_endpoint(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = _ClockEndpoint()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config = load_configuration(
            application_root=str(tmp_path),
            cwd=str(workspace),
            env={
                "OPENAI_BASE_URL": server.base_url,
                "OPENAI_API_KEY": "test",
                "OPENAI_MODEL": "fake-clock-model",
                "OPENAI_CONTEXT_WINDOW": "8192",
                "WORKSPACE_LIST_LIMIT": "0",
                "TERMINAL_MODE": "auto",
            },
        )
        monkeypatch.setattr("minagent.app.load_configuration", lambda *_args, **_kwargs: config)
        app = MinAgent(stdout=_FakeOutput())
        await app.initialize_configuration()
        await app.initialize_optional_features()
        await app.refresh_workspace_snapshot()
        app.messages.append({"role": "user", "content": "¿Qué hora es? dime la hora del sistema."})
        answer = await app.request_assistant_turn(None)
    finally:
        server.shutdown()
        server.server_close()

    assert len(server.requests) == 2, "expected one tool round and then the final answer"
    sent_tools = [tool["function"]["name"] for tool in server.requests[0]["tools"]]
    assert "run_terminal" in sent_tools

    tool_messages = [message["content"] for message in server.requests[1]["messages"] if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert str(datetime.now().year) in tool_messages[0]
    stamp = re.search(r"\d{2}:\d{2}:\d{2}", tool_messages[0])
    assert stamp, tool_messages[0]
    assert answer == f"La hora del sistema es {tool_messages[0].strip().splitlines()[-1]}"
    assert stamp.group(0) in answer
