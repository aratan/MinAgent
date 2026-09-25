"""Tests for the ``/model`` command: listing endpoint models and switching."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from minagent.app import MinAgent
from minagent.errors import AgentError
from minagent.openai import OpenAiClient

ENDPOINT = "http://127.0.0.1:11434/v1/chat/completions"


class _FakeOutput:
    def __init__(self, columns: int = 80) -> None:
        self.columns = columns
        self.chunks: list[str] = []

    def write(self, value: str) -> None:
        self.chunks.append(value)

    def isatty(self) -> bool:
        return False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def _client(payload: Any, status: int = 200, captured: list[httpx.Request] | None = None) -> OpenAiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(status, json=payload)

    return OpenAiClient(ENDPOINT, "test-key", "llama3:latest", [], transport=httpx.MockTransport(handler))


async def test_list_models_reads_the_openai_compatible_listing():
    captured: list[httpx.Request] = []
    client = _client(
        {"object": "list", "data": [{"id": "llama3:latest"}, {"id": "qwen2.5:7b"}, {"nope": 1}]},
        captured=captured,
    )
    assert await client.list_models() == ["llama3:latest", "qwen2.5:7b"]
    assert captured[0].headers["authorization"] == "Bearer test-key"
    assert captured[0].url.path == "/v1/models"


async def test_list_models_rejects_a_non_list_payload():
    with pytest.raises(AgentError, match="unexpected model list"):
        await _client({"unexpected": True}).list_models()


async def test_list_models_reports_http_errors():
    with pytest.raises(AgentError, match="HTTP 500"):
        await _client({}, status=500).list_models()


def _app() -> MinAgent:
    app = MinAgent(stdout=_FakeOutput())
    app.model = "llama3:latest"
    app.context_window = 32768
    return app


async def test_model_command_lists_models_and_marks_the_current_one():
    app = _app()
    app.open_ai_client = _client({"data": [{"id": "llama3:latest"}, {"id": "qwen2.5:7b"}]})
    await app.handle_model_command("")
    text = app._stdout.text
    assert "MODELS" in text and "llama3:latest" in text and "current" in text and "qwen2.5:7b" in text


async def test_model_command_reports_when_the_endpoint_cannot_be_reached():
    app = _app()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    app.open_ai_client = OpenAiClient(ENDPOINT, None, "llama3:latest", [], transport=httpx.MockTransport(handler))
    await app.handle_model_command("")
    assert "Could not list models" in app._stdout.text
    assert "/model <name>" in app._stdout.text


async def test_model_command_with_an_argument_switches_without_listing():
    app = _app()
    await app.handle_model_command("qwen2.5:7b")
    assert app.model == "qwen2.5:7b"


def test_selecting_a_model_switches_the_client_and_resets_usage():
    app = _app()
    app.last_prompt_tokens = 123.0
    app.open_ai_client = OpenAiClient(ENDPOINT, None, "llama3:latest", [])
    app.select_model("qwen2.5:7b")
    assert app.model == "qwen2.5:7b"
    assert app.open_ai_client.model == "qwen2.5:7b"
    assert app.last_prompt_tokens is None


def test_selecting_the_same_model_is_a_no_op():
    app = _app()
    app.open_ai_client = OpenAiClient(ENDPOINT, None, "llama3:latest", [])
    app.select_model("llama3:latest")
    assert "Already using" in app._stdout.text
