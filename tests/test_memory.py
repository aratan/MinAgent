"""Memory tests: the SQLite store, its tools, config, hints, and auto-capture."""

from __future__ import annotations

import pytest

from minagent.app import MinAgent
from minagent.config import load_configuration
from minagent.errors import AgentError
from minagent.memory import (
    AUTO_CAPTURE_SOURCE,
    MemoryStore,
    format_memory_hints,
    format_recall,
    format_remember_result,
    match_ratio,
    query_tokens,
    says_the_same_thing,
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


# ----------------------------------------------------------------- deduplicar


async def test_the_same_thing_stored_twice_becomes_one_memory(tmp_path):
    store = await _store(tmp_path)
    first = await store.remember(
        "procedure",
        "Instalar y ejecutar el proyecto",
        "Las dependencias se instalan con uv pip install y los tests se lanzan con "
        "uv run pytest -q desde la raíz del repositorio.",
    )
    # A different title, the same knowledge: the title is not the identity.
    again = await store.remember(
        "procedure",
        "Cómo correr la suite",
        "Usar uv: instalar dependencias con uv pip install y correr la suite con "
        "uv run pytest -q en la raíz.",
    )

    assert again["status"] == "duplicate"
    assert again["id"] == first["id"]
    assert len(await store.recent()) == 1


async def test_a_merge_keeps_the_newest_wording_and_counts_as_a_reinforcement(tmp_path):
    store = await _store(tmp_path)
    first = await store.remember("fact", "transformers 4.49.0", "La versión que funciona es la 4.49.0")
    before = (await store.recent())[0]["confidence"]

    await store.remember("fact", "Versión de transformers", "transformers 4.49.0 es la que funciona")

    entry = (await store.recent())[0]
    assert entry["id"] == first["id"]
    assert entry["title"] == "Versión de transformers"
    assert entry["success_count"] == 1
    assert entry["confidence"] > before


async def test_a_near_neighbour_on_the_same_subject_stays_its_own_memory(tmp_path):
    store = await _store(tmp_path)
    await store.remember(
        "fact", "Voz de Kokoro", "La voz por defecto no puede ser ef_heart porque no existe; en español solo hay ef_dora."
    )
    # Same domain, same kind, nothing in common: merging these would lose one.
    await store.remember("fact", "Ocupación de la GPU", "llama-server ocupa 5.5 GB de los 8188 MiB de la tarjeta.")

    assert len(await store.recent()) == 2


async def test_the_log_of_turns_is_never_merged(tmp_path):
    store = await _store(tmp_path)
    content = "Request: correr los tests\nTools used: run_terminal\nSteps: run_terminal(command=uv run pytest -q)"
    # Real turns are titled by the request, so the same work asked twice lands
    # as two entries with different titles and the same steps.
    await store.remember("experience", "correr los tests", content, None, AUTO_CAPTURE_SOURCE)
    await store.remember("experience", "vuelve a correr los tests", content, None, AUTO_CAPTURE_SOURCE)

    assert len(await store.recent()) == 2


def test_a_restatement_in_another_language_is_not_merged():
    # The limit of a lexical check, pinned so it is a known behaviour: the same
    # fact in two languages shares no words to compare, and a store that merged
    # on a guess would lose the memory instead of duplicating it.
    assert not says_the_same_thing(
        "Compute Environment Execution Pattern|Run computation scripts using .venv-compute/bin/python. "
        "Install dependencies within this specific compute venv.",
        "Configurar el render de vídeo|Para ejecutar el renderizado usar .venv-compute/bin/python e "
        "instalar las dependencias en el venv de cómputo.",
    )


def test_a_technical_identifier_counts_once_not_as_its_parts():
    # Split into four tokens it shares almost nothing with the other memory,
    # and two memories about the same interpreter stop looking alike.
    assert says_the_same_thing(
        "Render|Los backends corren con .venv-compute/bin/python, no con el venv del agente.",
        "Intérprete de cómputo|El render se ejecuta con .venv-compute/bin/python porque el venv del "
        "agente no tiene la pila de generación.",
    )


def test_the_remember_tool_says_when_it_merged_instead_of_saving():
    said = format_remember_result(
        {"status": "duplicate", "id": 7, "kind": "procedure", "title": "T", "confidence": 0.6, "success_count": 2}
    )
    # Telling the model is the point: it should learn the store already had it.
    assert "already said this" in said
    assert "merged into that one" in said


# ------------------------------------------------------- de-duplicar por sentido

# One memory said twice, in words that share a single distinctive token - below
# the floor the lexical rules need - so only an embedding can see it.
RESTATED = "Instalar dependencias\nEl agente instala con uv desde pyproject.toml."
RESTATEMENT = "Puesta al día de librerías\nSe usa uv reading del manifiesto de dependencias."
# A different fact about a different subject.
UNRELATED = "Ocupación de la GPU\nllama-server ocupa 5.5 GB de los 8188 MiB de la tarjeta."


class _FakeEmbedder:
    """A stand-in for the embedding model, with a fixed opinion about each text.

    It records what it was asked for, because how many requests a memory write
    makes is half of whether this feature is worth having at all.
    """

    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []
        self.unavailable = False

    async def embed(self, texts):
        self.calls.append(list(texts))
        answered = []
        for text in texts:
            # Keyed on the content, which is what the store wraps with a title
            # and tags before asking.
            match = next((vector for known, vector in self.vectors.items() if known in text), None)
            if match is None:
                return None
            answered.append(match)
        return answered


# Points 0.95 apart in a two-dimensional space: the same sentence twice.
SAME_MEANING = {
    "El agente instala con uv desde pyproject.toml.": (1.0, 0.0),
    "Se usa uv reading del manifiesto de dependencias.": (0.95, 0.31),
    "llama-server ocupa 5.5 GB de los 8188 MiB de la tarjeta.": (0.0, 1.0),
}


async def _embedded_store(tmp_path) -> MemoryStore:
    store = MemoryStore(str(tmp_path / "memory.db"), embed_model="nomic-embed-text")
    await store.initialize()
    store.embedder = _FakeEmbedder(SAME_MEANING)
    return store


async def test_a_restatement_the_words_miss_is_merged_by_the_embedding(tmp_path):
    store = await _embedded_store(tmp_path)
    first = await store.remember("fact", "Instalar dependencias", RESTATED)
    # The premise, pinned: the lexical rules alone would have kept these apart.
    assert not says_the_same_thing(RESTATED, RESTATEMENT)

    again = await store.remember("fact", "Puesta al día de librerías", RESTATEMENT)

    assert again["status"] == "duplicate"
    assert again["id"] == first["id"]
    assert len(await store.recent()) == 1


async def test_a_different_memory_on_another_subject_is_left_alone(tmp_path):
    store = await _embedded_store(tmp_path)
    await store.remember("fact", "Instalar dependencias", RESTATED)

    other = await store.remember("fact", "Ocupación de la GPU", UNRELATED)

    assert other["status"] == "created"
    assert len(await store.recent()) == 2


async def test_the_embedding_model_is_not_asked_about_memories_with_nothing_in_common(tmp_path):
    store = await _embedded_store(tmp_path)
    await store.remember("fact", "Instalar dependencias", RESTATED)

    await store.remember("fact", "Ocupación de la GPU", UNRELATED)

    # A request per stored memory on every save is what this gate exists to
    # avoid: the two texts share no distinctive word, so no request is made.
    assert store.embedder.calls == []


async def test_a_store_with_no_embedding_model_falls_back_to_the_words(tmp_path):
    store = await _store(tmp_path)
    first = await store.remember("fact", "Instalar dependencias", RESTATED)

    again = await store.remember("fact", "Puesta al día de librerías", RESTATEMENT)

    assert again["status"] == "created"
    assert again["id"] != first["id"]
    assert len(await store.recent()) == 2


async def test_an_embedding_model_that_cannot_answer_loses_no_memory(tmp_path):
    store = await _embedded_store(tmp_path)
    store.embedder = _FakeEmbedder({})  # a model that is not pulled answers nothing
    first = await store.remember(
        "procedure",
        "Instalar y ejecutar el proyecto",
        "Las dependencias se instalan con uv pip install y los tests se lanzan con "
        "uv run pytest -q desde la raíz del repositorio.",
    )

    # The save succeeds, and the duplicates the words already caught still are.
    again = await store.remember(
        "procedure",
        "Cómo correr la suite",
        "Usar uv: instalar dependencias con uv pip install y correr la suite con "
        "uv run pytest -q en la raíz.",
    )
    assert again["status"] == "duplicate"
    assert again["id"] == first["id"]
    assert len(await store.recent()) == 1


async def test_the_log_of_turns_is_never_merged_by_the_embedding_either(tmp_path):
    store = await _embedded_store(tmp_path)
    # Two near-identical turns, which is what the raw log is full of.
    content = "Request: renderizar el vídeo\nTools used: generate_video\nSteps: generate_video(prompt=ciberpunk)"
    await store.remember("experience", "renderizar el vídeo", content, None, AUTO_CAPTURE_SOURCE)
    await store.remember("experience", "renderizar el vídeo otra vez", content, None, AUTO_CAPTURE_SOURCE)

    assert len(await store.recent()) == 2
    assert store.embedder.calls == []


def test_the_embedding_model_is_configurable_and_optional(tmp_path):
    off = load_configuration(str(tmp_path), cwd=str(tmp_path), env={"OPENAI_MODEL": "m"})
    assert off.memory_embed_model == "" and off.improvement_model == ""
    on = load_configuration(
        str(tmp_path),
        cwd=str(tmp_path),
        env={
            "OPENAI_MODEL": "m",
            "MEMORY_EMBED_MODEL": "nomic-embed-text",
            "IMPROVEMENT_MODEL": "gemma4:12b-q3km",
        },
    )
    assert on.memory_embed_model == "nomic-embed-text"
    assert on.improvement_model == "gemma4:12b-q3km"
