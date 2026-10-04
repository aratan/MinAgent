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
    turn_title,
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
    # The turn's own entry is named by what it did, not by the request that
    # produced it: a title the next session could search for.
    assert [entry["title"] for entry in await app.memory_store.reviewable()] == ["read_file: README.md"]
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


# --------- turn_title: what a memory is named by, and why

def test_a_turn_is_named_by_what_it_did_not_by_what_was_asked():
    """The title is what a later session searches on, so it has to be the method.

    Measured on a real session: every auto-captured memory in the store was
    titled with the user's own words, so the recall that did fire brought back
    the request being asked instead of the way it was answered. One entry, the
    only method the store held, was titled ``read_file: datos.csv;
    run_terminal`` - which is searchable by the file a later task would be about.
    """
    steps = (
        'read_file(path=datos.csv) -> run_terminal(command=tr -d "\\r" < datos.csv) '
        "-> read_file(path=limpio.csv)"
    )
    title = turn_title(steps, "read_file, run_terminal")

    assert title == "read_file: datos.csv; run_terminal"
    # The words a later task would use, and not the ones that were asked with.
    assert "datos.csv" in title


def test_one_tool_in_the_title_even_when_the_arguments_differ():
    """Three reads are one method; listing all three reads as three memories."""
    title = turn_title("read_file(path=a.txt) -> read_file(path=b.txt) -> read_file(path=c.txt)", "read_file")
    assert title == "read_file: a.txt"


def test_a_title_survives_a_malformed_call_without_raising():
    """The steps come from tool calls the model wrote, so bad syntax is input.

    An unclosed quote or a missing bracket has to cost the argument, not the
    capture: raising here would lose the whole turn from the log the review
    culls, which is the opposite of what a malformed call deserves.
    """
    for steps in (
        'run_terminal(command=echo "sin cerrar) -> read_file(path=a.txt)',
        "read_file(path=a.txt) -> ??? -> run_terminal(command=ls)",
        "-> -> ->",
        "",
    ):
        assert isinstance(turn_title(steps, "read_file, run_terminal"), str)


def test_a_turn_with_no_parsable_step_still_gets_a_title():
    """No title means the entry can never be found, which is worse than a vague one."""
    assert turn_title("", "read_file, run_terminal") == "read_file, run_terminal"


# --------- degrading what was recalled into a turn that failed


async def _hinted_app(tmp_path, query: str = "how do I clean the csv"):
    """An app with one stored memory, offered as a hint for ``query``."""
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    saved = await app.memory_store.remember(
        "procedure", "Use iconv to drop the BOM", "iconv -f utf-8 -t utf-8 < in.csv", ["csv"]
    )
    app._current_user_request = query
    await app.refresh_memory_hints(query)
    assert app.memory_hint_context, "the memory has to actually reach the prompt"
    return app, saved["id"]


async def test_a_hint_offered_into_a_failed_turn_loses_confidence(tmp_path):
    """The negative reinforcement the store could do and nothing ever asked for.

    ``record_outcome`` is a tool, so only the model could lower a memory, and
    the model cannot tell that a hint it followed was the wrong one: the
    traceback names the tool, not the advice. Measured over a real store, 36
    reinforcements and no failures - not a record of things going well, but a
    store where every lesson was permanent.
    """
    app, memory_id = await _hinted_app(tmp_path)
    app._tools_used_this_turn = ["run_terminal"]
    app._tool_error_this_turn = True
    app._tool_errors_this_turn = 2

    await app.capture_experience("The command failed.")

    entry = (await app.memory_store.recent())[0]
    assert entry["id"] == memory_id
    assert entry["failure_count"] == 1
    assert entry["success_count"] == 0
    assert entry["confidence"] < 0.65


async def test_a_successful_turn_leaves_the_hint_it_used_alone(tmp_path):
    """Punishing a memory for a turn that worked teaches the store to hide it."""
    app, memory_id = await _hinted_app(tmp_path)
    app._tools_used_this_turn = ["run_terminal"]
    app._tool_error_this_turn = False

    await app.capture_experience("The command worked.")

    entry = (await app.memory_store.recent())[0]
    assert entry["id"] == memory_id
    assert entry["failure_count"] == 0


async def test_a_failed_turn_punishes_only_the_memories_it_was_given(tmp_path):
    """A memory that was never offered cannot have caused the turn.

    Without this the store would learn to stop volunteering its own contents,
    because everything in it would be charged for whatever happened next.
    """
    app, hinted_id = await _hinted_app(tmp_path)
    bystander = await app.memory_store.remember(
        "fact", "The project is in python", "There is a pyproject.toml", ["project"]
    )
    app._tools_used_this_turn = ["run_terminal"]
    app._tool_error_this_turn = True

    await app.capture_experience("The command failed.")

    entries = {entry["id"]: entry for entry in await app.memory_store.recent()}
    assert entries[hinted_id]["failure_count"] == 1
    assert entries[bystander["id"]]["failure_count"] == 0


async def test_a_failure_is_charged_once_and_not_repeated_by_the_next_turn(tmp_path):
    """The ids are spent with the punishment; a later turn has its own."""
    app, memory_id = await _hinted_app(tmp_path)
    app._tools_used_this_turn = ["run_terminal"]
    app._tool_error_this_turn = True
    await app.capture_experience("failed")

    app._tool_error_this_turn = True
    await app.capture_experience("failed again, with no hint offered")

    entry = (await app.memory_store.recent())[0]
    assert entry["id"] == memory_id
    assert entry["failure_count"] == 1
