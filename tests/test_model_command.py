"""Tests for the ``/model`` command: listing endpoint models and switching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from minagent.app import MinAgent
from minagent.editor import Key, build_autocomplete_state, handle_autocomplete_keypress
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


def test_model_picker_lists_matching_models_while_typing():
    state = build_autocomplete_state(
        "/model qw",
        len("/model qw"),
        [],
        [],
        ["llama3:latest", "qwen2.5:7b"],
        "llama3:latest",
    )
    assert state is not None and state["kind"] == "model"
    assert [candidate["value"] for candidate in state["candidates"]] == ["qwen2.5:7b"]


def test_model_picker_completes_the_chosen_name_in_place():
    state = build_autocomplete_state(
        "/model qw",
        len("/model qw"),
        [],
        [],
        ["qwen2.5:7b"],
        "llama3:latest",
    )
    app_line = "/model qw"

    class _Editor:
        line = app_line
        cursor = len(app_line)

    editor = _Editor()
    action = handle_autocomplete_keypress(state, Key(name="enter"), editor)
    assert action == {"kind": "complete", "selected_file": None}
    assert editor.line == "/model qwen2.5:7b"


def test_persist_model_replaces_the_env_entry(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_BASE_URL=http://x/v1\nOPENAI_MODEL=old\n")
    app = _app()
    app.application_root = str(tmp_path)
    assert app.persist_model("new-model") == str(env)
    assert env.read_text() == "OPENAI_BASE_URL=http://x/v1\nOPENAI_MODEL=new-model\n"


def test_persist_model_appends_when_missing(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_BASE_URL=http://x/v1\n")
    app = _app()
    app.application_root = str(tmp_path)
    app.persist_model("fresh-model")
    assert env.read_text().endswith("OPENAI_MODEL=fresh-model\n")


async def test_switching_a_model_persists_and_warms_it(tmp_path: Path, monkeypatch):
    app = _app()
    app.application_root = str(tmp_path)
    app.open_ai_client = OpenAiClient(ENDPOINT, None, "llama3:latest", [])
    calls: list[Any] = []

    async def fake_call(messages, options=None):
        calls.append(options)
        return {"payload": {"usage": {}}, "message": {"content": "pong", "tool_calls": []}}

    async def no_context() -> None:
        return None

    monkeypatch.setattr(app, "call_chat_completions", fake_call)
    monkeypatch.setattr(app, "fetch_model_context_length", no_context)
    await app.switch_model("qwen2.5:7b")
    assert app.model == "qwen2.5:7b"
    assert (tmp_path / ".env").read_text().strip().endswith("OPENAI_MODEL=qwen2.5:7b")
    assert calls == [{"max_tokens": 1}], "the model was not warmed before the first turn"
    assert "Model ready." in app._stdout.text


async def test_fetch_model_context_prefers_num_ctx_over_the_model_maximum():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(
            200,
            json={
                "parameters": "stop '<|end|>'\nnum_ctx                        8192\ntemperature 0.7",
                "model_info": {"qwen35.context_length": 262144},
            },
        )

    client = OpenAiClient(ENDPOINT, None, "m", [], transport=httpx.MockTransport(handler))
    assert await client.fetch_model_context() == 8192


async def test_fetch_model_context_falls_back_to_model_info():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model_info": {"llama.context_length": 32768}})

    client = OpenAiClient(ENDPOINT, None, "m", [], transport=httpx.MockTransport(handler))
    assert await client.fetch_model_context() == 32768


async def test_fetch_model_context_skips_an_endpoint_that_is_not_ollama():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a non-Ollama endpoint must not be called")

    # A local OpenAI-compatible server on another port, e.g. llama.cpp on 8080.
    client = OpenAiClient(
        "http://127.0.0.1:8080/v1/chat/completions", None, "m", [], transport=httpx.MockTransport(handler)
    )
    assert await client.fetch_model_context() is None
    cloud = OpenAiClient(
        "https://api.openai.com/v1/chat/completions", "k", "m", [], transport=httpx.MockTransport(handler)
    )
    assert await cloud.fetch_model_context() is None


def test_the_endpoint_window_beats_the_name_hint():
    app = _app()
    app.context_window = 32768
    app.model = "bonsai27b-8k:latest"  # the name claims 8k
    assert app.model_window_estimate() == 8192
    app.model_context_length = 4096  # the endpoint says otherwise
    assert app.model_window_estimate() == 4096
    assert app.effective_context_window() == 4096
