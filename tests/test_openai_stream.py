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


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content", "reasoning_summary"])
async def test_every_spelling_of_the_reasoning_channel_is_read(field):
    """A thinking model that answers only in this channel must not look empty.

    llama.cpp, some gateways and Ollama each spell it differently, and a
    spelling the client does not know turns a real answer into "the endpoint
    returned nothing".
    """
    body = (
        f'data: {{"choices":[{{"delta":{{"{field}":"pensando"}}}}]}}\n\n'.encode()
        + b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    )
    result = await read_streaming_response(_Stream(body))
    assert result["message"]["reasoning_content"] == "pensando"
    assert not result["message"]["content"]


async def test_the_reasoning_channel_is_streamed_to_the_caller():
    streamed: list[str] = []
    body = (
        b'data: {"choices":[{"delta":{"reasoning":"a"}}]}\n\n'
        b'data: {"choices":[{"delta":{"reasoning":"b"}}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    )
    await read_streaming_response(_Stream(body), on_reasoning_delta=streamed.append)
    assert "".join(streamed) == "ab"


async def test_usage_is_requested_while_streaming():
    """Without it the endpoint never counts the prompt, and the meter is a guess."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        bodies.append(_json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_TEXT_FRAME + b"data: [DONE]\n\n",
        )

    client = OpenAiClient(ENDPOINT, None, "m", [], timeout_ms=5000, transport=httpx.MockTransport(handler))
    await client.complete([{"role": "user", "content": "hi"}])
    assert bodies[0]["stream_options"] == {"include_usage": True}


async def test_an_endpoint_that_rejects_the_option_is_asked_again_without_it():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        calls.append(body)
        if "stream_options" in body:
            return httpx.Response(400, json={"error": "unknown field stream_options"})
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_TEXT_FRAME + b"data: [DONE]\n\n",
        )

    client = OpenAiClient(ENDPOINT, None, "m", [], timeout_ms=5000, transport=httpx.MockTransport(handler))
    result = await client.complete([{"role": "user", "content": "hi"}])
    assert result["message"]["content"] == "hi"
    assert len(calls) == 2, "the field is dropped and the request is retried once"


async def test_the_usage_frame_from_the_endpoint_is_read():
    usage_frame = b'data: {"choices":[],"usage":{"prompt_tokens":4321,"completion_tokens":12}}\n\n'
    body = _TEXT_FRAME + usage_frame + b"data: [DONE]\n\n"
    result = await read_streaming_response(_Stream(body))
    assert result["payload"]["usage"] == {"prompt_tokens": 4321, "completion_tokens": 12}


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
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        bodies.append(_json.loads(request.content))
        return httpx.Response(400, content=b"bad request")

    client = OpenAiClient(ENDPOINT, None, "m", [], timeout_ms=5000, transport=httpx.MockTransport(handler))
    with pytest.raises(AgentError, match="HTTP 400"):
        await client.complete([{"role": "user", "content": "hi"}])
    # The one retry is not a retry of the same request: it drops stream_options,
    # so the corrected body is the last one tried and the error still surfaces.
    assert len(bodies) == 2
    assert "stream_options" not in bodies[1]
