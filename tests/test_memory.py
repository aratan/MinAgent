"""Memory tests: the SQLite store, its tools, config, hints, and auto-capture."""

from __future__ import annotations

import pytest

from minagent.app import MinAgent
from minagent.config import load_configuration
from minagent.errors import AgentError
from minagent.memory import (
    MemoryStore,
    format_memory_hints,
    format_recall,
    match_ratio,
    query_tokens,
)
from minagent.workspace import WorkspaceAccess


class _FakeOutput:
    """A stdout stand-in that records what was written."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, value: str) -> None:
        self.chunks.append(value)

    def isatty(self) -> bool:
        return False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


async def _store(tmp_path) -> MemoryStore:
    store = MemoryStore(str(tmp_path / "memory.db"))
    await store.initialize()
    return store


def _memory_app(tmp_path) -> MinAgent:
    app = MinAgent(stdout=_FakeOutput())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "Test"
    app.memory_enabled = True
    app.memory_db_path = str(tmp_path / ".minagent" / "memory.db")
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    return app


async def test_remember_then_recall_returns_the_procedure(tmp_path):
    store = await _store(tmp_path)
    saved = await store.remember("procedure", "Run the tests with uv", "Run `uv run pytest -q`.", ["tests"])
    assert saved["status"] == "created"
    assert saved["id"] == 1

    found = await store.recall("how do I run the tests")
    assert [memory["title"] for memory in found] == ["Run the tests with uv"]

    text = format_recall(found)
    assert "Run the tests with uv" in text and "#1" in text


async def test_remember_same_title_reinforces_instead_of_duplicating(tmp_path):
    store = await _store(tmp_path)
    first = await store.remember("procedure", "Deploy steps", "step one")
    second = await store.remember("procedure", "deploy   STEPS", "step one and two")
    assert second["status"] == "reinforced"
    assert second["id"] == first["id"]

    memories = await store.recent()
    assert len(memories) == 1
    assert memories[0]["content"] == "step one and two"
    assert memories[0]["success_count"] == 1


async def test_record_outcome_reinforces_and_degrades(tmp_path):
    store = await _store(tmp_path)
    saved = await store.remember("procedure", "Try the cache first", "read the cache")
    good = await store.record_outcome(saved["id"], True, "worked")
    assert good["success_count"] == 1
    assert good["confidence"] > 0.5
    bad = await store.record_outcome(saved["id"], False)
    assert bad["failure_count"] == 1
    assert bad["confidence"] < good["confidence"]


async def test_record_outcome_rejects_an_unknown_id(tmp_path):
    store = await _store(tmp_path)
    with pytest.raises(AgentError, match="no memory with id"):
        await store.record_outcome(99, True)


async def test_hints_respect_confidence_and_stay_bounded(tmp_path):
    store = await _store(tmp_path)
    saved = await store.remember("procedure", "Bounded hint", "x" * 400)
    hints = await store.hints("bounded hint")
    assert len(hints) == 1
    assert "Bounded hint" in format_memory_hints(hints, max_chars=2000)

    await store.record_outcome(saved["id"], False)
    await store.record_outcome(saved["id"], False)
    assert await store.hints("bounded hint") == []


async def test_forget_removes_a_memory(tmp_path):
    store = await _store(tmp_path)
    saved = await store.remember("fact", "Port", "8080")
    assert await store.forget(saved["id"]) is True
    assert await store.forget(saved["id"]) is False
    assert await store.recent() == []


async def test_statistics_report_counts(tmp_path):
    store = await _store(tmp_path)
    await store.remember("fact", "One", "1")
    saved = await store.remember("fact", "Two", "2")
    await store.record_outcome(saved["id"], True)
    stats = await store.statistics()
    assert stats["total"] == 2
    assert stats["successes"] == 1
    assert stats["uses"] == 1


def test_config_enables_memory_and_defaults_its_path(tmp_path):
    config = load_configuration(
        str(tmp_path), cwd=str(tmp_path), env={"OPENAI_MODEL": "m", "MEMORY_ENABLED": "on"}
    )
    assert config.memory_enabled is True
    assert config.memory_db_path == str(tmp_path / ".agents" / "memory" / "memoria.db")
    assert config.memory_direct_answer is True


def test_config_leaves_memory_off_by_default(tmp_path):
    config = load_configuration(str(tmp_path), cwd=str(tmp_path), env={"OPENAI_MODEL": "m"})
    assert config.memory_enabled is False


def test_config_can_disable_direct_answers(tmp_path):
    config = load_configuration(
        str(tmp_path), cwd=str(tmp_path), env={"OPENAI_MODEL": "m", "MEMORY_DIRECT_ANSWER": "off"}
    )
    assert config.memory_direct_answer is False


def test_match_ratio_counts_query_tokens_present_in_a_memory():
    memory = {"title": "Run the tests", "content": "uv run pytest", "tags": "tests"}
    assert match_ratio(query_tokens("run the tests"), memory) == 1.0
    assert match_ratio(query_tokens("deploy build publish"), memory) == 0.0


async def test_lookup_only_answers_strong_confident_memories(tmp_path):
    store = await _store(tmp_path)
    saved = await store.remember("procedure", "Run tests", "uv run pytest -q", ["tests"])
    # Confidence starts below the direct-answer floor, so the model is still needed.
    assert await store.lookup("uv run pytest") is None

    await store.record_outcome(saved["id"], True)
    await store.record_outcome(saved["id"], True)
    found = await store.lookup("uv run pytest")
    assert found is not None and found["id"] == saved["id"]
    assert found["match_ratio"] >= 0.6

    # A weakly related request must not harvest the memory on partial overlap.
    assert await store.lookup("run the production deployment now") is None


async def test_memory_tools_are_exposed_and_callable(tmp_path):
    app = _memory_app(tmp_path)
    assert await app.initialize_optional_features() == []
    assert not {"recall", "remember"} & {tool["function"]["name"] for tool in app.tools}
    app.load_capabilities(["memory"])
    names = {tool["function"]["name"] for tool in app.tools}
    assert {"recall", "remember", "record_outcome"} <= names

    saved = await app.execute_tool(
        "remember", {"title": "Use uv", "content": "uv run pytest", "tags": ["tests"]}
    )
    assert "Saved memory #1" in saved
    recalled = await app.execute_tool("recall", {"query": "uv pytest"})
    assert "Use uv" in recalled


async def test_hints_are_injected_into_the_system_prompt(tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.load_capabilities(["memory"])
    await app.execute_tool("remember", {"title": "Deploy checklist", "content": "run make deploy"})

    await app.refresh_memory_hints("what is the deploy checklist")
    assert "Deploy checklist" in app.memory_hint_context
    assert "Memory hints" in [section["name"] for section in app._current_system_prompt_sections]


async def test_successful_turn_is_captured_as_experience(tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app._current_user_request = "List the workspace root"
    app._tools_used_this_turn = ["list_directory"]
    app._steps_this_turn = ["list_directory(path=.)"]
    app._tool_error_this_turn = False
    app._memory_remembered_this_turn = False

    await app.capture_experience("The root holds src and tests.")
    memories = await app.memory_store.recent()
    assert len(memories) == 1
    assert memories[0]["kind"] == "experience"
    assert "List the workspace root" in memories[0]["content"]
    # The concrete step is stored, so a later session can repeat how it was done.
    assert "Steps: list_directory(path=.)" in memories[0]["content"]


async def test_direct_answer_reuses_a_confident_memory_without_the_model(tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    saved = await app.memory_store.remember("procedure", "Deploy", "run make deploy", ["deploy"])

    assert await app.answer_from_memory("run make deploy") is None
    await app.memory_store.record_outcome(saved["id"], True)
    await app.memory_store.record_outcome(saved["id"], True)

    answer = await app.answer_from_memory("run make deploy")
    assert answer == "run make deploy"
    assert app.messages[-1] == {"role": "assistant", "content": "run make deploy"}


async def test_direct_answer_can_be_disabled(tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.memory_direct_answer = False
    saved = await app.memory_store.remember("procedure", "Deploy", "run make deploy", ["deploy"])
    await app.memory_store.record_outcome(saved["id"], True)
    await app.memory_store.record_outcome(saved["id"], True)

    assert await app.answer_from_memory("run make deploy") is None


async def test_turn_with_a_tool_error_is_not_captured(tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app._current_user_request = "Do the thing"
    app._tools_used_this_turn = ["read_file"]
    app._tool_error_this_turn = True

    await app.capture_experience("done")
    assert await app.memory_store.recent() == []


async def test_recall_without_memory_enabled_raises(tmp_path):
    app = MinAgent(stdout=_FakeOutput())
    app.memory_enabled = False
    with pytest.raises(AgentError, match="not enabled"):
        await app.recall_memory({"query": "anything"})
