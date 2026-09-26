"""Stream robustness and HTTP retry tests for the OpenAI-compatible client."""

from __future__ import annotations

import httpx
import pytest

from minagent.errors import AgentError
from minagent.openai import OpenAiClient, read_streaming_response

ENDPOINT = "http://127.0.0.1:11434/v1/chat/completions"
_TEXT_FRAME = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'


class _Stream:
    """A response double with only the method the parser needs."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def aiter_bytes(self):
        yield self.body


async def test_a_stream_without_a_finish_reason_keeps_the_partial_answer():
    result = await read_streaming_response(_Stream(_TEXT_FRAME))
    assert result["message"]["content"] == "hi"
    assert result["payload"]["finish_reason"] == "incomplete"
    assert result["payload"]["truncated"] is True


async def test_an_empty_stream_without_a_finish_reason_still_errors():
    with pytest.raises(AgentError, match="ended before a complete response"):
        await read_streaming_response(_Stream(b": keep-alive\n\n"))


async def test_a_429_is_retried_then_succeeds():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_TEXT_FRAME + b"data: [DONE]\n\n",
        )

    client = OpenAiClient(ENDPOINT, None, "m", [], timeout_ms=5000, transport=httpx.MockTransport(handler))
    result = await client.complete([{"role": "user", "content": "hi"}])
    assert calls["count"] == 2, "a 429 must be retried once"
    assert result["message"]["content"] == "hi"


async def test_a_connection_error_is_retried_before_failing():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_TEXT_FRAME + b"data: [DONE]\n\n",
        )

    client = OpenAiClient(ENDPOINT, None, "m", [], timeout_ms=5000, transport=httpx.MockTransport(handler))
    result = await client.complete([{"role": "user", "content": "hi"}])
    assert calls["count"] == 2, "a connection error must be retried"
    assert result["message"]["content"] == "hi"


async def test_a_persistent_connection_error_becomes_an_agent_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = OpenAiClient(ENDPOINT, None, "m", [], timeout_ms=5000, transport=httpx.MockTransport(handler))
    with pytest.raises(AgentError, match="Could not connect"):
        await client.complete([{"role": "user", "content": "hi"}])


async def test_a_client_error_is_not_retried():
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, content=b"bad request")

    client = OpenAiClient(ENDPOINT, None, "m", [], timeout_ms=5000, transport=httpx.MockTransport(handler))
    with pytest.raises(AgentError, match="HTTP 400"):
        await client.complete([{"role": "user", "content": "hi"}])
    assert calls["count"] == 1, "a 4xx other than 429 must not be retried"
