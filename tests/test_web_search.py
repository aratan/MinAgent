"""Web search tests: the Ollama client, its formatters, config, and app dispatch."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from minagent.app import MinAgent
from minagent.config import load_configuration
from minagent.errors import AgentError
from minagent.web_search import (
    WebSearchClient,
    create_web_search_tools,
    format_fetch_result,
    format_search_results,
)
from minagent.workspace import WorkspaceAccess


class _FakeOutput:
    def write(self, value: str) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _client(payload: Any, status: int = 200, captured: list[httpx.Request] | None = None) -> WebSearchClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(status, json=payload)

    return WebSearchClient(
        "https://ollama.com/api", "test-key", 30, transport=httpx.MockTransport(handler)
    )


async def test_search_normalizes_results_and_sends_the_key():
    captured: list[httpx.Request] = []
    client = _client(
        {
            "results": [
                {"title": "  Ollama  ", "url": "https://ollama.com", "content": "line one\nline two"},
                {"title": "", "url": "https://example.com", "content": "snippet"},
                "not-an-object",
            ]
        },
        captured=captured,
    )
    results = await client.search("what is ollama?")
    assert results[0] == {"title": "Ollama", "url": "https://ollama.com", "content": "line one line two"}
    assert results[1]["title"] == "https://example.com"
    assert len(results) == 2
    assert captured[0].headers["authorization"] == "Bearer test-key"
    assert captured[0].url.path == "/api/web_search"


async def test_search_without_a_key_is_an_error():
    client = WebSearchClient("https://ollama.com/api", None, 30, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(AgentError, match="OLLAMA_API_KEY"):
        await client.search("anything")


async def test_search_rejects_a_query_that_is_empty_or_too_long():
    client = _client({"results": []})
    with pytest.raises(AgentError, match="non-empty query"):
        await client.search("   ")
    with pytest.raises(AgentError, match="exceeds"):
        await client.search("x" * 500)


async def test_search_rejects_a_payload_without_results():
    client = _client({"unexpected": True})
    with pytest.raises(AgentError, match="no usable results"):
        await client.search("anything")


async def test_fetch_returns_the_page_and_its_links():
    client = _client({"title": "Docs", "content": "Hello", "links": ["https://a", 5, "https://b"]})
    page = await client.fetch("https://docs.example.com")
    assert page["title"] == "Docs"
    assert page["content"] == "Hello"
    assert page["links"] == ["https://a", "https://b"]


async def test_fetch_requires_an_http_url():
    client = _client({})
    with pytest.raises(AgentError, match="http:// or https://"):
        await client.fetch("docs.example.com")


async def test_http_errors_become_agent_errors():
    client = _client({"error": "nope"}, status=401)
    with pytest.raises(AgentError, match="HTTP 401.*OLLAMA_API_KEY"):
        await client.search("anything")


def test_formatters_render_results_and_pages():
    text = format_search_results("ollama", [{"title": "T", "url": "U", "content": "C"}])
    assert 'Web results for "ollama"' in text and "1. T" in text and "untrusted" in text
    assert format_search_results("ollama", []) == 'No web results for "ollama".'

    page = format_fetch_result({"title": "T", "url": "U", "content": "Body", "links": ["https://a"]})
    assert "Fetched T (U)" in page and "Body" in page and "Links: https://a" in page


def test_tool_schemas_expose_both_web_tools():
    names = {tool["function"]["name"] for tool in create_web_search_tools()}
    assert names == {"web_search", "web_fetch"}


def test_config_enables_web_search_with_ollama_defaults(tmp_path):
    config = load_configuration(
        str(tmp_path),
        cwd=str(tmp_path),
        env={"OPENAI_MODEL": "m", "WEB_SEARCH_ENABLED": "on", "OLLAMA_API_KEY": "secret"},
    )
    assert config.web_search_enabled is True
    assert config.ollama_api_key == "secret"
    assert config.web_search_base_url == "https://ollama.com/api"
    assert config.web_search_timeout_seconds == 120


def test_config_leaves_web_search_off_by_default(tmp_path):
    config = load_configuration(str(tmp_path), cwd=str(tmp_path), env={"OPENAI_MODEL": "m"})
    assert config.web_search_enabled is False
    assert config.ollama_api_key is None


def _web_app(tmp_path) -> MinAgent:
    app = MinAgent(stdout=_FakeOutput())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "Test"
    app.web_search_enabled = True
    app.web_search_client = _client({"results": [{"title": "R", "url": "https://r", "content": "snippet"}]})
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    app.ensure_web_search_tools()
    return app


async def test_app_dispatches_web_search(tmp_path):
    app = _web_app(tmp_path)
    names = {tool["function"]["name"] for tool in app.tools}
    assert not {"web_search", "web_fetch"} & names, "the web tools wait to be loaded"
    assert "load_capability('web')" in app._unloaded_tool_hint()
    app.load_capabilities(["web"])
    names = {tool["function"]["name"] for tool in app.tools}
    assert {"web_search", "web_fetch"} <= names
    result = await app.execute_tool("web_search", {"query": "how do I do this"})
    assert "R" in result and "untrusted" in result


async def test_web_search_without_a_client_raises(tmp_path):
    app = MinAgent(stdout=_FakeOutput())
    app.web_search_enabled = False
    with pytest.raises(AgentError, match="not enabled"):
        await app.run_web_search({"query": "anything"})


def test_web_search_nudge_mentions_the_tools(tmp_path):
    app = MinAgent(stdout=_FakeOutput())
    note = app.web_search_nudge()
    assert "web_search" in note and "Three or more" in note
