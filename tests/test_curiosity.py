"""Tests for what the agent asks about while nobody is watching.

Three layers, tested apart because they fail apart: the hats turn an answer
into questions, the curation decides whether to propose any, and the resident
decides how many passes a cycle may fund. None of them may reach the network or
a model, so every test here runs offline.

The property that matters most is the one the split was built for: the agent can
*ask* for something overnight, and what it gets believed still has to walk the
same path a question written by hand would.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from minagent.app import MinAgent
from minagent.curiosity import (
    HATS,
    MAX_CURATED_QUESTIONS,
    already_asked,
    append_questions,
    build_prompt,
    needs_curation,
    open_question_count,
    parse_questions,
)
from minagent.resident import MAX_RESEARCH_PASSES_PER_CYCLE, build_worker
from minagent.workspace import WorkspaceAccess

GOOD_ANSWER = json.dumps(
    [
        {
            "hat": "black",
            "question": "What breaks first when the idle loop runs out of budget mid-cycle?",
            "why": "The cap lives beside the loop, so a mid-cycle exhaustion is the failure nobody has measured.",
        },
        {
            "hat": "green",
            "question": "Has anyone tried folding the six hats into one request instead of six?",
            "why": "It is what this implementation does, and nothing says it is right.",
        },
    ]
)


def test_the_prompt_carries_the_hats_and_what_it_knows() -> None:
    prompt = build_prompt(
        subject="why the nightly loop stopped learning",
        user_model="- Prefers short answers.",
        lessons="- The idle gate failed closed once.",
        already_asked="- What does IdleHint return over ssh?",
    )

    for name, discipline, _ in HATS:
        assert name in prompt
        assert discipline.split(".")[0] in prompt
    assert "why the nightly loop stopped learning" in prompt
    assert "- Prefers short answers." in prompt
    assert "- The idle gate failed closed once." in prompt
    assert "What does IdleHint return over ssh?" in prompt


def test_a_well_formed_answer_becomes_questions_with_reasons() -> None:
    parsed = parse_questions(GOOD_ANSWER)

    assert [hat for hat, _, _ in parsed] == ["black", "green"]
    assert parsed[0][1].startswith("What breaks first")
    assert parsed[0][2]
    assert MAX_CURATED_QUESTIONS == 3


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "not json at all",
        "[]",
        '{"hat": "black"}',
        # A question with no reason is the thing that spends credits without learning.
        json.dumps([{"hat": "black", "question": "Is it slow?", "why": ""}]),
        # An unknown hat is not one of the six disciplines.
        json.dumps([{"hat": "purple", "question": "Is it slow?", "why": "It is."}]),
    ],
)
def test_an_unusable_answer_yields_nothing_rather_than_a_vague_question(reply: str) -> None:
    assert parse_questions(reply) == []


def test_the_same_question_is_not_asked_twice_in_one_reply() -> None:
    reply = json.dumps(
        [
            {"hat": "black", "question": "Same question?", "why": "First."},
            {"hat": "green", "question": "Same question?", "why": "Second."},
        ]
    )

    assert len(parse_questions(reply)) == 1


def test_a_reply_is_capped_so_one_pass_cannot_flood_the_queue() -> None:
    reply = json.dumps(
        [{"hat": "white", "question": f"Question {index}?", "why": "Because."} for index in range(20)]
    )

    assert len(parse_questions(reply)) == MAX_CURATED_QUESTIONS


def test_a_question_that_spans_lines_stays_one_line() -> None:
    reply = json.dumps([{"hat": "white", "question": "Line one\nline two", "why": "Because."}])

    question = parse_questions(reply)[0][1]
    assert "\n" not in question
    assert question == "Line one line two"


def test_curation_only_happens_when_the_queue_is_short() -> None:
    assert needs_curation("")
    assert needs_curation("- [ ] One open question")
    assert not needs_curation("- [ ] One\n- [ ] Two")
    # Questions the person wrote down outrank anything this would add.
    assert not needs_curation("- [ ] One\n- [ ] Two\n- [ ] Three")


def test_answered_questions_are_handed_back_so_they_are_not_repeated() -> None:
    asked = already_asked("- [x] What does IdleHint return?  — 0 always over ssh\n- [ ] Still open")

    assert "What does IdleHint return?" in asked
    assert "0 always over ssh" in asked
    assert "Still open" not in asked


def test_curated_questions_land_in_the_queue_without_touching_what_is_there(tmp_path: Path) -> None:
    queue = tmp_path / "PREGUNTAS.md"
    queue.write_text("# Preguntas\n\n- [ ] What did the user already ask?\n", encoding="utf-8")

    added = append_questions(str(queue), parse_questions(GOOD_ANSWER))

    text = queue.read_text(encoding="utf-8")
    assert added == 2
    assert "What did the user already ask?" in text
    assert text.startswith("# Preguntas\n")
    assert "- [ ] What breaks first" in text
    assert "  why: The cap lives beside the loop" in text
    assert open_question_count(text) == 3


def test_a_question_already_in_the_queue_is_not_written_twice(tmp_path: Path) -> None:
    queue = tmp_path / "PREGUNTAS.md"
    queue.write_text(
        "- [ ] What breaks first when the idle loop runs out of budget mid-cycle?\n",
        encoding="utf-8",
    )

    assert append_questions(str(queue), parse_questions(GOOD_ANSWER)) == 1
    assert queue.read_text(encoding="utf-8").count("What breaks first") == 1


def test_curation_creates_the_queue_when_there_is_none(tmp_path: Path) -> None:
    queue = tmp_path / "nested" / "PREGUNTAS.md"

    assert append_questions(str(queue), parse_questions(GOOD_ANSWER)) == 2
    assert queue.exists()


def test_nothing_to_propose_writes_nothing(tmp_path: Path) -> None:
    queue = _queue(tmp_path)

    assert append_questions(str(queue), []) == 0
    assert not queue.exists()


def _queue(tmp_path: Path) -> Path:
    """Where the app will actually look: application_root/agente/PREGUNTAS.md."""
    return tmp_path / "agente" / "PREGUNTAS.md"


def _app(tmp_path: Path) -> MinAgent:
    app = MinAgent()
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    return app


async def test_a_pass_curates_the_queue_before_it_spends_one_on_answers(tmp_path: Path) -> None:
    """The whole design in one test: propose first, answer second."""
    app = _app(tmp_path)
    app.web_search_client = object()
    app._ask_with_reflection_model = _replying(GOOD_ANSWER)
    answered: list[str] = []

    async def fake_research() -> str:
        answered.append("answered")
        return ""

    app._research_one_question = fake_research

    note = await app.run_research_pass()

    assert "curated 2 question(s)" in note
    assert "What breaks first" in (_queue(tmp_path)).read_text(encoding="utf-8")
    assert answered == ["answered"]


async def test_a_full_queue_is_left_alone_and_no_request_is_made(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.web_search_client = object()
    calls: list[str] = []

    async def refuse(prompt: list[dict[str, str]]) -> str:
        calls.append("asked")
        return GOOD_ANSWER

    app._ask_with_reflection_model = refuse
    queue = _queue(tmp_path)
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text("- [ ] One\n- [ ] Two\n", encoding="utf-8")

    async def fake_research() -> str:
        return "research: answered"

    app._research_one_question = fake_research

    note = await app.run_research_pass()

    assert "curated" not in note
    assert calls == []
    assert (_queue(tmp_path)).read_text(encoding="utf-8") == "- [ ] One\n- [ ] Two\n"


async def test_an_unparsable_reply_spends_the_call_and_writes_nothing(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.web_search_client = object()
    app._ask_with_reflection_model = _replying("I would rather not.")
    app._research_one_question = _silent

    note = await app.run_research_pass()

    assert note == ""
    assert not (_queue(tmp_path)).exists()


async def test_no_web_client_means_no_nightly_curiosity(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.web_search_client = None
    app._research_one_question = _silent

    assert await app.run_research_pass() == ""
    assert not (_queue(tmp_path)).exists()


async def test_a_budget_that_refuses_means_no_request_and_no_queue(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.web_search_client = object()
    app._ask_with_reflection_model = _replying(GOOD_ANSWER)
    app._research_one_question = _silent

    class _Full:
        def reserve_call(self) -> bool:
            return False

        def commit_call(self) -> None:  # pragma: no cover - never reached
            raise AssertionError("committed a call that was never reserved")

    app.resident_worker = type("W", (), {"budget": _Full()})()

    assert await app.run_research_pass() == ""
    assert not (_queue(tmp_path)).exists()


async def test_a_cycle_funds_several_passes_but_not_an_unlimited_number(tmp_path: Path) -> None:
    """Two or three per cycle is the ask; the stop is still the budget."""
    passes = 0

    async def research() -> str:
        nonlocal passes
        passes += 1
        return "research: learned something"

    class _Config:
        improvement_autonomous = True
        improvement_cycle_seconds = 0.01
        improvement_idle_seconds = 0
        improvement_max_cycles = 1
        improvement_model_calls = 24
        improvement_calls_per_cycle = 20

    class _Agent:
        async def reflect_on_session(self, reason: str) -> str:
            return ""

        async def run_research_pass(self) -> str:
            return await research()

    class _Idle:
        def read(self) -> object:
            class _State:
                idle = True
                seconds = 999.0

            return _State()

    worker = build_worker(_Agent(), _Config(), idle_reader=_Idle(), sleep=_no_sleep)
    worker.research = research
    await asyncio.wait_for(worker.run(), timeout=5)

    assert 0 < passes <= MAX_RESEARCH_PASSES_PER_CYCLE
    assert passes == MAX_RESEARCH_PASSES_PER_CYCLE


async def test_the_worker_is_wired_to_the_agents_research_pass(tmp_path: Path) -> None:
    """The hook existed for exactly this and was never connected."""

    class _Config:
        improvement_autonomous = True
        improvement_cycle_seconds = 900.0
        improvement_idle_seconds = 120.0

    class _Agent:
        async def reflect_on_session(self, reason: str) -> str:
            return ""

        async def run_research_pass(self) -> str:
            return "research: something"

    worker = build_worker(_Agent(), _Config())

    assert worker.research is not None
    assert await worker.research() == "research: something"


def _replying(text: str):
    """A stand-in for the model call, returning the given text whatever it is asked."""

    async def ask(prompt: list[dict[str, str]]) -> str:
        return text

    return ask


async def _silent() -> str:
    return ""


async def _no_sleep(_seconds: float) -> None:
    return None
