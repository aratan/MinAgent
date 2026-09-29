"""OpenAI-compatible Chat Completions client.

Streams the model's response as it arrives, reassembles tool calls that arrive
in fragments, surfaces the optional reasoning channel, and reports endpoint
errors with enough of the body to be actionable.
"""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

import httpx

from .errors import AgentError, CancellationToken, OperationAborted
from .jsutil import is_int

DEFAULT_TIMEOUT_MS = 7 * 60 * 1000
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_TOOL_ARGUMENT_CHARS = 1024 * 1024
# A short budget for metadata calls (model list, context length) that should
# never hold up a turn.
DEFAULT_METADATA_TIMEOUT_MS = 10 * 1000
_OLLAMA_CHAT_SUFFIX = "/v1/chat/completions"

# A tuple: ``except`` rejects a set with a TypeError, which silently disabled
# these retries.
RETRYABLE_NETWORK_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.PoolTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
)

# A busy server is worth retrying; a client error is not.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
REJECTED_STATUSES = {400, 404, 405, 415, 422}
"""Statuses a strict endpoint uses to reject a request body it does not know."""
DEFAULT_RETRY_DELAY_SECONDS = 0.25
MAX_RETRY_AFTER_SECONDS = 30.0
MAX_REQUEST_ATTEMPTS = 3

_FRAME_BOUNDARY = re.compile(r"\r?\n\r?\n")


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The server's ``Retry-After`` in seconds, bounded, or ``None``."""
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_SECONDS)


async def _wait_for_retry_delay(signal: CancellationToken | None, delay: float | None = None) -> None:
    """Pause before a retry, honouring a ``Retry-After`` and any cancellation."""
    import asyncio

    seconds = DEFAULT_RETRY_DELAY_SECONDS if delay is None else delay
    if seconds <= 0:
        return
    if signal is None:
        await asyncio.sleep(seconds)
        return
    if signal.cancelled:
        raise OperationAborted("The operation was aborted.")
    try:
        await asyncio.wait_for(signal.wait(), timeout=seconds)
    except TimeoutError:
        return
    raise OperationAborted("The operation was aborted.")


async def _read_response_prefix(response: httpx.Response, max_bytes: int, signal: CancellationToken | None) -> str:
    """Read at most ``max_bytes`` of a response body for an error message."""
    chunks: list[bytes] = []
    total = 0
    truncated = False
    try:
        async for chunk in response.aiter_bytes():
            if total >= max_bytes:
                truncated = True
                break
            keep = chunk[: max_bytes - total]
            chunks.append(keep)
            total += len(keep)
            if len(keep) < len(chunk):
                truncated = True
                break
    except httpx.HTTPError:
        pass
    if signal is not None and signal.cancelled:
        raise OperationAborted("The operation was aborted.")
    text = b"".join(chunks).decode("utf-8", errors="replace")
    return f"{text} [response excerpt truncated]" if truncated else text


def _consume_frame(
    frame: str,
    tool_calls: dict[int, dict[str, Any]],
    state: dict[str, Any],
    on_text_delta: Callable[[str], None] | None,
    on_reasoning_delta: Callable[[str], None] | None,
) -> None:
    """Fold one SSE frame into the accumulated response state."""
    data_lines = [line[5:].lstrip() for line in re.split(r"\r?\n", frame) if line.startswith("data:")]
    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        if data == "[DONE]":
            state["finished"] = True
        return
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        raise AgentError("Endpoint sent an invalid streaming event.") from None
    if isinstance(event, dict) and event.get("error"):
        error = event["error"]
        detail = error if isinstance(error, str) else json.dumps(error, ensure_ascii=False)
        raise AgentError(f"Endpoint streaming error: {detail[:1000]}")
    if not isinstance(event, dict):
        return
    if event.get("usage"):
        state["usage"] = event["usage"]

    choices = event.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    if not isinstance(choice, dict):
        return
    if choice.get("finish_reason"):
        state["finish_reason"] = choice["finish_reason"]
    delta = choice.get("delta") or {}
    if not isinstance(delta, dict):
        return

    # Three spellings for the same channel: llama.cpp sends reasoning_content,
    # some gateways send reasoning_summary, and Ollama's OpenAI-compatible
    # surface sends reasoning. Missing one of them is not cosmetic: a thinking
    # model that answers only in this channel would look like it returned
    # nothing at all.
    reasoning_delta = ""
    for field in ("reasoning_summary", "reasoning_content", "reasoning"):
        value = delta.get(field)
        if isinstance(value, str) and value:
            reasoning_delta = value
            break
    if reasoning_delta:
        # Keep the reasoning even when nothing renders it: a reasoning model can
        # finish a turn with only this channel filled, and dropping it would look
        # exactly like an endpoint that returned nothing.
        state["reasoning"] += reasoning_delta
        if on_reasoning_delta is not None:
            on_reasoning_delta(reasoning_delta)

    content = delta.get("content")
    if isinstance(content, str) and content:
        state["content"] += content
        if on_text_delta is not None:
            on_text_delta(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                state["content"] += part["text"]
                if on_text_delta is not None:
                    on_text_delta(part["text"])

    for delta_call in delta.get("tool_calls") or []:
        if not isinstance(delta_call, dict):
            continue
        index = delta_call["index"] if is_int(delta_call.get("index")) else len(tool_calls)
        call = tool_calls.setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if delta_call.get("id"):
            call["id"] = delta_call["id"]
        if delta_call.get("type"):
            call["type"] = delta_call["type"]
        function_delta = delta_call.get("function") or {}
        if function_delta.get("name"):
            call["function"]["name"] += function_delta["name"]
        if function_delta.get("arguments"):
            call["function"]["arguments"] += function_delta["arguments"]
            if len(call["function"]["arguments"]) > MAX_TOOL_ARGUMENT_CHARS:
                raise AgentError("Endpoint returned tool arguments larger than the 1 MiB limit.")


async def read_streaming_response(
    response: Any,
    on_text_delta: Callable[[str], None] | None = None,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    on_reasoning_delta: Callable[[str], None] | None = None,
    signal: CancellationToken | None = None,
) -> dict[str, Any]:
    """Parse an SSE response into the assembled assistant message.

    ``response`` only needs an ``aiter_bytes()`` async iterator, which keeps the
    parser testable without a live endpoint.
    """
    tool_calls: dict[int, dict[str, Any]] = {}
    state: dict[str, Any] = {
        "content": "",
        "reasoning": "",
        "usage": None,
        "finish_reason": None,
        "finished": False,
    }
    buffer = ""
    bytes_read = 0
    interrupted = False
    stream_incomplete = False
    decoder = codecs.getincrementaldecoder("utf-8")()

    try:
        async for chunk in _iter_chunks(response):
            if signal is not None and signal.cancelled:
                interrupted = True
                break
            bytes_read += len(chunk)
            if bytes_read > max_response_bytes:
                raise AgentError(f"Endpoint streaming response exceeds the {max_response_bytes} byte limit.")
            buffer += decoder.decode(chunk)
            while True:
                match = _FRAME_BOUNDARY.search(buffer)
                if match is None:
                    break
                frame = buffer[: match.start()]
                buffer = buffer[match.end():]
                _consume_frame(frame, tool_calls, state, on_text_delta, on_reasoning_delta)
                if state["finished"]:
                    break
            if state["finished"]:
                break
    except httpx.HTTPError:
        if signal is None or not signal.cancelled:
            raise
        interrupted = True

    if not interrupted:
        try:
            buffer += decoder.decode(b"", True)
            if buffer.strip():
                _consume_frame(buffer, tool_calls, state, on_text_delta, on_reasoning_delta)
            if not state["finished"] and not state["finish_reason"]:
                if state["content"] or state["reasoning"] or tool_calls or state["usage"]:
                    # The server closed the stream without a finish reason. Keep
                    # what arrived and mark it incomplete, rather than discarding
                    # a partial answer the user already saw stream by.
                    state["finish_reason"] = "incomplete"
                    stream_incomplete = True
                else:
                    raise AgentError("Endpoint stream ended before a complete response was received.")
        except httpx.HTTPError:
            if signal is None or not signal.cancelled:
                raise
            interrupted = True

    if interrupted:
        return {
            "payload": {"usage": state["usage"], "finish_reason": "aborted"},
            "message": {"role": "assistant", "content": state["content"] or None, "interrupted": True},
        }

    content = state["content"]
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if state["reasoning"]:
        message["reasoning_content"] = state["reasoning"]
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    finish_reason = state["finish_reason"]
    if finish_reason == "content_filter":
        raise AgentError("Endpoint stopped the response because of its content filter.")
    for call in message.get("tool_calls", []):
        if not call.get("id") or not call["function"].get("name"):
            raise AgentError("Endpoint returned an incomplete tool call.")
        try:
            arguments = json.loads(call["function"]["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise AgentError(f"Endpoint returned invalid arguments for tool {call['function']['name']}.") from None
    payload: dict[str, Any] = {"usage": state["usage"], "finish_reason": finish_reason}
    if state["reasoning"]:
        payload["reasoning_tokens"] = len(state["reasoning"]) // 4
    if finish_reason == "length" or stream_incomplete:
        # Keep the partial text instead of discarding it; the caller continues it.
        payload["truncated"] = True
    return {"payload": payload, "message": message}


def _models_url(endpoint: str) -> str:
    """Derive the OpenAI-compatible ``/models`` URL from the chat endpoint."""
    if endpoint.endswith("/chat/completions"):
        return f"{endpoint[: -len('/chat/completions')]}/models"
    return f"{endpoint.rstrip('/')}/models"


def _looks_like_ollama(endpoint: str) -> bool:
    """Heuristic: the endpoint is the local Ollama server.

    Only Ollama serves ``/api/show``, so this avoids sending a metadata request
    to a cloud endpoint or an unrelated OpenAI-compatible server.
    """
    try:
        parts = urlsplit(endpoint)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    return port == 11434 or "ollama" in host


def _context_length_from_show(payload: Any) -> int | None:
    """Read the real context length from an Ollama ``/api/show`` payload.

    ``parameters`` carries the ``num_ctx`` the server will actually use, which
    is the limit that matters: ``*.context_length`` states the model's maximum,
    often far larger than the runtime window. ``num_ctx`` wins when present.
    """
    if not isinstance(payload, dict):
        return None
    parameters = payload.get("parameters")
    if isinstance(parameters, str):
        match = re.search(r"(?m)^\s*num_ctx\s+(\d+)\s*$", parameters)
        if match and int(match.group(1)) > 0:
            return int(match.group(1))
    model_info = payload.get("model_info")
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if isinstance(key, str) and key.endswith(".context_length") and is_int(value) and int(value) > 0:
                return int(value)
    return None


async def _iter_chunks(response: Any) -> AsyncIterator[bytes]:
    """Iterate a response body as bytes, accepting httpx or a plain test double."""
    if hasattr(response, "aiter_bytes"):
        async for chunk in response.aiter_bytes():
            yield chunk
        return
    async for chunk in response:
        yield chunk


class OpenAiClient:
    """Minimal client for an OpenAI-compatible streaming Chat Completions API."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None,
        model: str,
        tools: Sequence[dict[str, Any]],
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not endpoint or not model:
            raise AgentError("OpenAI client configuration is incomplete.")
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.tools = list(tools)
        self.timeout_ms = timeout_ms
        self.max_response_bytes = max_response_bytes
        self.transport = transport

    async def fetch_model_context(self, model: str | None = None) -> int | None:
        """Return the model's real context length when the endpoint publishes it.

        Ollama serves ``POST /api/show``; other OpenAI-compatible servers do not,
        and simply return ``None`` so the caller falls back to its own estimate.
        """
        name = model or self.model
        if not name or not self.endpoint.endswith(_OLLAMA_CHAT_SUFFIX) or not _looks_like_ollama(self.endpoint):
            return None
        base = self.endpoint[: -len(_OLLAMA_CHAT_SUFFIX)]
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = httpx.Timeout(DEFAULT_METADATA_TIMEOUT_MS / 1000)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False, transport=self.transport
            ) as client:
                response = await client.post(f"{base}/api/show", headers=headers, json={"model": name})
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        return _context_length_from_show(payload)

    async def list_models(self) -> list[str]:
        """Return the model identifiers the endpoint advertises.

        Uses the OpenAI-compatible ``/models`` listing, which Ollama serves at
        ``/v1/models`` and llama.cpp serves at ``/v1/models`` too.
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = httpx.Timeout(self.timeout_ms / 1000) if self.timeout_ms > 0 else httpx.Timeout(None)
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False, transport=self.transport
            ) as client:
                response = await client.get(_models_url(self.endpoint), headers=headers)
        except httpx.HTTPError as error:
            raise AgentError(f"Could not reach the endpoint to list models: {error}") from error
        if response.status_code >= 400:
            raise AgentError(f"Endpoint returned HTTP {response.status_code} when listing models.")
        try:
            payload = response.json()
        except ValueError:
            raise AgentError("Endpoint returned an invalid model list.") from None
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise AgentError("Endpoint returned an unexpected model list.")
        models: list[str] = []
        for item in items:
            identifier = item.get("id") if isinstance(item, dict) else None
            if isinstance(identifier, str) and identifier:
                models.append(identifier)
        return models

    async def complete(
        self,
        request_messages: Sequence[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST one streaming completion and return the assembled message."""
        options = options or {}
        signal: CancellationToken | None = options.get("signal")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": list(request_messages),
            "stream": True,
        }
        # Token usage is opt-in while streaming, and without it the context
        # meter can never be checked against what the endpoint really read.
        request_body["stream_options"] = {"include_usage": True}
        if options.get("with_tools"):
            request_body["tools"] = options.get("available_tools", self.tools)
            request_body["tool_choice"] = options.get("tool_choice", "auto")
        if options.get("max_tokens"):
            request_body["max_tokens"] = options["max_tokens"]
        # Optional fields a caller knows about and the endpoint may not. They
        # are tracked together so one rejection can drop all of them and the
        # retry is a plain request every endpoint understands.
        optional: list[str] = ["stream_options"]
        for key, value in (options.get("extra_body") or {}).items():
            request_body[key] = value
            optional.append(key)

        timeout = httpx.Timeout(self.timeout_ms / 1000) if self.timeout_ms > 0 else httpx.Timeout(None)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, transport=self.transport
        ) as client:
            response: httpx.Response | None = None
            stream: Any = None
            for attempt in range(MAX_REQUEST_ATTEMPTS):
                last_attempt = attempt == MAX_REQUEST_ATTEMPTS - 1
                try:
                    stream = client.stream(
                        "POST",
                        self.endpoint,
                        headers=headers,
                        json=request_body,
                    )
                    response = await stream.__aenter__()
                except RETRYABLE_NETWORK_ERRORS as error:
                    if signal is not None and signal.cancelled:
                        raise OperationAborted("The operation was aborted.") from error
                    if not last_attempt:
                        await _wait_for_retry_delay(signal, DEFAULT_RETRY_DELAY_SECONDS * (attempt + 1))
                        continue
                    raise AgentError(
                        f"Could not connect to the OpenAI-compatible endpoint: {error}"
                    ) from error
                if response.status_code in RETRYABLE_STATUSES and not last_attempt:
                    retry_after = _retry_after_seconds(response)
                    await stream.__aexit__(None, None, None)
                    stream = None
                    response = None
                    await _wait_for_retry_delay(
                        signal, retry_after if retry_after is not None else DEFAULT_RETRY_DELAY_SECONDS * (attempt + 1)
                    )
                    continue
                if response.status_code in REJECTED_STATUSES and any(
                    field in request_body for field in optional
                ):
                    # A strict endpoint may refuse a field rather than ignore
                    # it. Drop them all and ask again: the usage frame and the
                    # caller's hints are both worth asking for, and neither is
                    # worth failing the turn over.
                    await stream.__aexit__(None, None, None)
                    stream = None
                    response = None
                    for field in optional:
                        request_body.pop(field, None)
                    if last_attempt:
                        break
                    continue
                break

            assert response is not None and stream is not None
            try:
                if response.status_code >= 400:
                    body_text = await _read_response_prefix(response, 8 * 1024, signal)
                    raise AgentError(f"Endpoint returned HTTP {response.status_code}: {body_text}")
                content_type = response.headers.get("content-type", "").lower()
                if "text/event-stream" in content_type:
                    return await read_streaming_response(
                        response,
                        on_text_delta=options.get("on_text_delta"),
                        max_response_bytes=self.max_response_bytes,
                        on_reasoning_delta=options.get("on_reasoning_delta"),
                        signal=signal,
                    )
                body_text = await _read_response_prefix(response, 2 * 1024, signal)
                raise AgentError(
                    "The endpoint did not return a streaming response "
                    f"(Content-Type: {content_type or 'unknown'}). Response: {body_text}"
                )
            finally:
                await stream.__aexit__(None, None, None)
