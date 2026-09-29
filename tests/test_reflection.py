"""Reflection tests: what a session decides to keep, and what the review forgets.

The point these cover is the decision, not the plumbing: a turn is only worth a
model call when it looks like a discovery, an answer that cannot be read keeps
nothing, and a review that forgets the noise has to stop offering it back.
"""

from __future__ import annotations

import pytest

from minagent.config import load_configuration
from minagent.errors import AgentError
from minagent.memory import (
    AUTO_CAPTURE_SOURCE,
    REVIEWED_SOURCE,
    MemoryStore,
    ReviewAction,
)
from minagent.reflection import (
    EUREKA_ERROR_THRESHOLD,
    detect_eureka,
    format_review_result,
    parse_review,
    parse_verdict,
)
from tests.test_memory import _memory_app


class _Silent:
    """A client that answers in the wrapper but with nothing in it."""

    async def complete(self, messages, options=None):
        return {"message": {"role": "assistant", "content": ""}}


class _StubClient:
    """A model stand-in that answers with a fixed text and counts the calls.

    It returns the same wrapper the real client does, ``{"message": ...}``. An
    earlier version of this stub answered with the bare message, which hid a
    reader that looked in the wrong place and judged nothing at all.
    """

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages, options=None):
        self.calls.append(list(messages))
        return {"message": {"role": "assistant", "content": self.answer}}


# --------------------------------------------------------------------- signals


def test_several_failures_before_a_success_is_the_shape_of_a_discovery():
    signal = detect_eureka(tool_errors=EUREKA_ERROR_THRESHOLD, steps=["a", "b"], turn_succeeded=True)
    assert signal is not None and signal.reason == "perseverance"
    # The detail is what the model is told to judge, so it has to say what happened.
    assert "2 tool errors" in signal.detail


def test_a_single_failure_is_an_ordinary_turn():
    assert detect_eureka(tool_errors=1, steps=["read_file"], turn_succeeded=True) is None


def test_a_long_turn_is_worth_judging_even_without_failures():
    signal = detect_eureka(tool_errors=0, steps=[f"step {index}" for index in range(6)], turn_succeeded=True)
    assert signal is not None and signal.reason == "long_work"


def test_a_turn_that_ended_in_an_error_has_nothing_settled_to_remember():
    # Judging it would spend a request on work that is not finished yet.
    assert detect_eureka(tool_errors=5, steps=[f"step {i}" for i in range(9)], turn_succeeded=False) is None


# --------------------------------------------------------------------- verdicts


def test_a_keep_answer_is_read_through_a_fenced_code_block():
    verdict = parse_verdict(
        'Here it is:\n```json\n{"keep": true, "kind": "procedure", "title": "Use uv", '
        '"content": "uv run pytest", "reason": "it works"}\n```\n'
    )
    assert verdict.keep
    assert verdict.kind == "procedure"
    assert verdict.title == "Use uv"


def test_a_drop_answer_keeps_nothing():
    assert parse_verdict('{"keep": false, "reason": "one-off"}').keep is False


def test_an_answer_that_cannot_be_read_keeps_nothing():
    # Storing on a garbled answer is how a memory store fills with guesses.
    assert parse_verdict("I think it is probably worth it.").keep is False


def test_a_keep_without_something_to_recall_is_refused():
    verdict = parse_verdict('{"keep": true, "title": "Use uv", "content": "  "}')
    assert verdict.keep is False


def test_a_kind_the_store_does_not_know_cannot_be_invented():
    verdict = parse_verdict('{"keep": true, "kind": "vibes", "title": "T", "content": "C"}')
    assert verdict.kind == "solution"


# ----------------------------------------------------------------------- review


def test_a_review_keeps_and_forgets_by_id():
    entries = [{"id": 1}, {"id": 2}, {"id": 3}]
    actions = parse_review('{"keep": [1], "forget": [2], "notes": {"1": "the method"}}', entries)
    assert [(action.entry_id, action.keep, action.note) for action in actions] == [
        (1, True, "the method"),
        (2, False, ""),
    ]


def test_a_review_cannot_forget_an_entry_it_invented():
    # Forgetting a memory by a number the model made up would delete the wrong thing.
    actions = parse_review('{"keep": [1], "forget": [99]}', [{"id": 1}])
    assert [action.entry_id for action in actions] == [1]


def test_an_entry_named_twice_is_decided_once():
    actions = parse_review('{"keep": [1], "forget": [1]}', [{"id": 1}])
    assert len(actions) == 1


def test_an_unreadable_review_changes_nothing():
    assert parse_review("they all look fine to me", [{"id": 1}]) == []


def test_the_review_result_counts_what_it_did():
    assert format_review_result([ReviewAction(1, True), ReviewAction(2, False)]) == (
        "Memory review: 1 kept, 1 forgotten."
    )
    assert format_review_result([]) == "Memory review: nothing to change."


# -------------------------------------------------------------------- the store


async def test_the_review_offers_only_entries_nobody_has_judged(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    await store.initialize()
    await store.remember("experience", "raw one", "a", None, AUTO_CAPTURE_SOURCE)
    await store.remember("experience", "raw two", "b", None, AUTO_CAPTURE_SOURCE)
    await store.remember("procedure", "deliberate", "c")

    assert [entry["title"] for entry in await store.reviewable()] == ["raw one", "raw two"]


async def test_a_kept_entry_is_reinforced_and_never_offered_again(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    await store.initialize()
    saved = await store.remember("experience", "raw one", "a", None, AUTO_CAPTURE_SOURCE)
    before = (await store.recent())[0]["confidence"]

    changed = await store.apply_review([ReviewAction(saved["id"], True, "the method")])
    assert changed == 1
    entry = (await store.recent())[0]
    assert entry["confidence"] > before
    assert entry["source"] == REVIEWED_SOURCE
    # A cull that left the survivors looking unreviewed would offer them every pass.
    assert await store.reviewable() == []


async def test_a_forgotten_entry_is_gone_and_the_count_is_honest(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.db"))
    await store.initialize()
    saved = await store.remember("experience", "raw one", "a", None, AUTO_CAPTURE_SOURCE)

    assert await store.apply_review([ReviewAction(saved["id"], False)]) == 1
    assert await store.recent() == []
    # An id that is not in the log changes nothing, and is not counted as if it had.
    assert await store.apply_review([ReviewAction(4242, False)]) == 0


# ------------------------------------------------------------------ the layers


async def _judged_app(tmp_path, answer: str):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.open_ai_client = _StubClient(answer)
    app._current_user_request = "Make the video render"
    return app


async def test_a_discovery_turn_is_judged_and_the_judgement_is_stored(tmp_path):
    app = await _judged_app(
        tmp_path,
        '{"keep": true, "kind": "solution", "title": "Run video with COMPUTE_PYTHON", '
        '"content": "The backends need .venv-compute, not the agent venv."}',
    )
    app._tools_used_this_turn = ["generate_video", "run_terminal"]
    app._steps_this_turn = ["generate_video(...)", "run_terminal(...)"]
    app._tool_error_this_turn = False
    app._tool_errors_this_turn = 2

    await app.capture_experience("The render worked.")

    assert len(app.open_ai_client.calls) == 1
    kinds = {memory["kind"] for memory in await app.memory_store.recent()}
    # The raw log entry and the judged one both exist: the log is what the review culls.
    assert kinds == {"experience", "solution"}


async def test_an_ordinary_turn_costs_no_model_call(tmp_path):
    app = await _judged_app(tmp_path, '{"keep": true, "title": "T", "content": "C"}')
    app._tools_used_this_turn = ["read_file"]
    app._steps_this_turn = ["read_file(path=README.md)"]

    await app.capture_experience("It says hello.")

    assert app.open_ai_client.calls == []


async def test_a_turn_the_model_already_remembered_is_not_judged_again(tmp_path):
    app = await _judged_app(tmp_path, '{"keep": true, "title": "T", "content": "C"}')
    app._tools_used_this_turn = ["generate_video"]
    app._steps_this_turn = ["generate_video(...)", "run_terminal(...)"]
    app._tool_error_this_turn = False
    app._tool_errors_this_turn = 3
    # It called remember with the context in hand and has already made the call.
    app._memory_remembered_this_turn = True

    await app.capture_experience("done")

    assert app.open_ai_client.calls == []


async def test_an_unreadable_judgement_stores_nothing_extra(tmp_path):
    app = await _judged_app(tmp_path, "sure, that seems useful")
    app._tools_used_this_turn = ["generate_video"]
    app._steps_this_turn = ["generate_video(...)", "run_terminal(...)"]
    app._tool_error_this_turn = False
    app._tool_errors_this_turn = 2

    await app.capture_experience("done")

    assert [memory["kind"] for memory in await app.memory_store.recent()] == ["experience"]


async def test_a_judgement_the_model_drops_stores_nothing_extra(tmp_path):
    app = await _judged_app(tmp_path, '{"keep": false, "reason": "one-off"}')
    app._tools_used_this_turn = ["generate_video"]
    app._steps_this_turn = ["generate_video(...)", "run_terminal(...)"]
    app._tool_error_this_turn = False
    app._tool_errors_this_turn = 2

    await app.capture_experience("done")

    assert [memory["kind"] for memory in await app.memory_store.recent()] == ["experience"]


async def test_the_review_waits_for_its_interval(tmp_path):
    app = await _judged_app(tmp_path, '{"keep": [], "forget": []}')
    app.memory_reflection_interval = 3
    app._tools_used_this_turn = ["read_file"]
    app._steps_this_turn = ["read_file(path=README.md)"]

    for _ in range(2):
        await app.capture_experience("read it")

    assert app.open_ai_client.calls == []
    await app.capture_experience("read it")
    assert len(app.open_ai_client.calls) == 1


async def test_the_review_forgets_the_noise_and_keeps_the_method(tmp_path):
    app = await _judged_app(tmp_path, "")
    app.memory_reflection_interval = 1
    entries = [await app.memory_store.remember(
        "experience", title, "content", None, AUTO_CAPTURE_SOURCE
    ) for title in ("noise", "the method")]

    # The review reads the log, so the stub answers the review, not a verdict.
    app.open_ai_client = _StubClient('{"keep": [2], "forget": [1]}')
    app._tools_used_this_turn = ["read_file"]
    app._steps_this_turn = ["read_file(path=README.md)"]

    await app.capture_experience("read it")

    remaining = await app.memory_store.recent()
    # The turn wrote its own raw entry, and the review only ruled on the two the
    # log held, so the noise is gone and the new entry is still waiting its turn.
    assert "noise" not in {memory["title"] for memory in remaining}
    kept = next(memory for memory in remaining if memory["title"] == "the method")
    assert kept["id"] == entries[1]["id"]
    assert kept["source"] == REVIEWED_SOURCE
    assert [entry["title"] for entry in await app.memory_store.reviewable()] == ["Make the video render"]
    assert "1 kept, 1 forgotten" in app._stdout.text


async def test_a_review_with_nothing_to_cull_asks_nobody(tmp_path):
    app = await _judged_app(tmp_path, "{}")
    app.memory_reflection_interval = 1
    app._tools_used_this_turn = ["read_file"]
    app._tool_error_this_turn = True

    await app.capture_experience("it failed")

    assert app.open_ai_client.calls == []


async def test_an_answer_in_the_wrong_shape_judges_nothing(tmp_path):
    app = await _judged_app(tmp_path, "")
    app.open_ai_client = _Silent()
    app._tools_used_this_turn = ["generate_video"]
    app._steps_this_turn = ["generate_video(...)", "run_terminal(...)"]
    app._tool_error_this_turn = False
    app._tool_errors_this_turn = 2

    await app.capture_experience("done")

    # A wrapper with no content in it is an answer that was not given.
    assert [memory["kind"] for memory in await app.memory_store.recent()] == ["experience"]


async def test_a_reflection_failure_never_breaks_the_turn(tmp_path):
    class _Broken:
        async def complete(self, messages, options=None):
            raise OSError("the endpoint went away")


    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.open_ai_client = _Broken()
    app._current_user_request = "Make the video render"
    app._tools_used_this_turn = ["generate_video"]
    app._steps_this_turn = ["generate_video(...)", "run_terminal(...)"]
    app._tool_error_this_turn = False
    app._tool_errors_this_turn = 2

    # The reply is already on screen by now; a failed memory is a missed memory.
    await app.capture_experience("The render worked.")
    assert [memory["kind"] for memory in await app.memory_store.recent()] == ["experience"]


# ---------------------------------------------------------------------- config


def test_the_reflection_settings_have_defaults(tmp_path):
    config = load_configuration(str(tmp_path), cwd=str(tmp_path), env={"OPENAI_MODEL": "m", "MEMORY_ENABLED": "on"})
    assert config.memory_eureka is True
    assert config.memory_reflection_interval == 10


def test_the_reflection_settings_can_be_changed(tmp_path):
    config = load_configuration(
        str(tmp_path),
        cwd=str(tmp_path),
        env={"OPENAI_MODEL": "m", "MEMORY_EUREKA": "off", "MEMORY_REFLECTION_INTERVAL": "25"},
    )
    assert config.memory_eureka is False
    assert config.memory_reflection_interval == 25


def test_a_reflection_interval_of_zero_is_refused(tmp_path):
    with pytest.raises(AgentError, match="MEMORY_REFLECTION_INTERVAL"):
        load_configuration(
            str(tmp_path), cwd=str(tmp_path), env={"OPENAI_MODEL": "m", "MEMORY_REFLECTION_INTERVAL": "0"}
        )
