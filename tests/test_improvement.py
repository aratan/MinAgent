"""Improvement tests: what a session reflection proposes, and what it may change.

The interesting part is the boundary. Proposals are free; changing a setting is
not, and every rule that keeps a wrong hypothesis cheap - evidence required, a
value bounded, one nudge per setting per cooldown - is a rule a test should pin
down rather than trust.
"""

from __future__ import annotations

import inspect
import json

import pytest

from minagent.compute import VramRelease
from minagent.improvement import (
    ADJUSTMENT_COOLDOWN_SECONDS,
    SAFE_SETTINGS,
    AdjustmentLog,
    Hypothesis,
    append_document,
    apply_adjustments,
    build_session_prompt,
    format_adjustment_report,
    format_document_section,
    load_adjustment_log,
    parse_hypotheses,
    plan_adjustments,
    read_document,
    save_adjustment_log,
)
from tests.test_memory import _FakeOutput, _memory_app

A_HYPOTHESIS = {
    "title": "La cola se atasca",
    "kind": "improvement",
    "target": "agent",
    "statement": "Los renders se acumulan más rápido de lo que terminan",
    "evidence": "3 trabajos rechazados por VRAM y 2 en cola en la última hora",
    "expected": "Menos trabajos de golpe",
    "verify": "Que la cola no supere 2 en espera tras el próximo render",
    "setting": "COMPUTE_QUEUE_LIMIT",
    "value": "2",
    "reason": "La cola baja a dos y el rechazo desaparece",
}


def _answer(*hypotheses: dict) -> str:
    return json.dumps({"hypotheses": list(hypotheses)})


def _set_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, str]) -> None:
    """Pin the settings the planner measures against, undone after the test."""
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _hypothesis(**overrides) -> Hypothesis:
    fields = {
        "title": "T",
        "statement": "S",
        "evidence": "E",
        "expected": "X",
        "verify": "V",
        "target": "agent",
        "kind": "improvement",
        "setting": "COMPUTE_QUEUE_LIMIT",
        "value": "2",
    }
    fields.update(overrides)
    return Hypothesis(**fields)


# ----------------------------------------------------------------- hypotheses


def test_a_hypothesis_comes_back_with_its_evidence():
    found = parse_hypotheses(_answer(A_HYPOTHESIS))
    assert len(found) == 1
    assert found[0].title == "La cola se atasca"
    assert found[0].evidence.startswith("3 trabajos rechazados")
    assert found[0].setting == "COMPUTE_QUEUE_LIMIT"


def test_a_claim_with_nothing_behind_it_is_not_a_hypothesis():
    # The whole safety story starts here: no evidence, no proposal at all.
    without = dict(A_HYPOTHESIS)
    without["evidence"] = ""
    assert parse_hypotheses(_answer(without)) == []


def test_a_setting_the_project_never_defined_cannot_reach_the_file():
    invented = dict(A_HYPOTHESIS, setting="SOME_OTHER_SETTING", value="1")
    found = parse_hypotheses(_answer(invented))
    # The hypothesis is still worth reading; only its setting is dropped.
    assert found[0].setting == "" and found[0].value == ""


def test_evidence_given_as_a_list_is_still_evidence():
    # A model asked for evidence answers with a list of the items it found,
    # unprompted. Reading only strings threw the best hypothesis of a session
    # away and reported that the session had taught nothing.
    listed = dict(A_HYPOTHESIS, evidence=["el venv del agente tiene 5.17", "el de cómputo tiene 4.49"])
    found = parse_hypotheses(_answer(listed))
    assert len(found) == 1
    assert "4.49" in found[0].evidence and "5.17" in found[0].evidence


def test_a_hypothesis_whose_evidence_is_an_empty_list_is_dropped():
    assert parse_hypotheses(_answer(dict(A_HYPOTHESIS, evidence=[]))) == []


def test_an_unreadable_reflection_yields_nothing_rather_than_raising():
    assert parse_hypotheses("creo que podríamos hacer el job timeout más grande") == []
    assert parse_hypotheses("{}") == []


def test_an_insight_is_labelled_apart_from_an_improvement():
    found = parse_hypotheses(_answer(dict(A_HYPOTHESIS, kind="insight", target="project", setting="")))
    assert found[0].kind == "insight"
    assert found[0].target == "project"


# ---------------------------------------------------------------- adjustments


def _current(**overrides) -> dict[str, str]:
    values = {
        "MEMORY_REFLECTION_INTERVAL": "10",
        "COMPUTE_QUEUE_LIMIT": "8",
        "COMPUTE_JOB_TIMEOUT_SECONDS": "1800",
        "COMPUTE_VOICE_TIMEOUT_SECONDS": "120",
    }
    values.update(overrides)
    return values


def test_a_supported_change_is_planned():
    planned = plan_adjustments(
        [_hypothesis()], _current(), AdjustmentLog(), now=1_000_000.0
    )
    assert [(item.name, item.previous, item.proposed) for item in planned] == [
        ("COMPUTE_QUEUE_LIMIT", "8", "2")
    ]


def test_a_setting_within_its_bounds_is_left_alone():
    # Proposing the value it already has is not a change; writing it would
    # spend the setting's cooldown on nothing.
    assert plan_adjustments([_hypothesis(value="8")], _current(), AdjustmentLog(), 1_000.0) == []


def test_a_value_outside_the_bounds_is_refused():
    rule = SAFE_SETTINGS["COMPUTE_QUEUE_LIMIT"]
    too_big = _hypothesis(value=str(rule.maximum + 1))
    assert plan_adjustments([too_big], _current(), AdjustmentLog(), 1_000.0) == []
    too_small = _hypothesis(value=str(rule.minimum - 1))
    assert plan_adjustments([too_small], _current(), AdjustmentLog(), 1_000.0) == []


def test_a_jump_to_the_floor_is_refused():
    # 1800 -> 120 is the setting's minimum. Pinning something to its floor is
    # a reaction to the problem, not a considered value.
    hypothesis = _hypothesis(setting="COMPUTE_JOB_TIMEOUT_SECONDS", value="120")
    assert plan_adjustments([hypothesis], _current(), AdjustmentLog(), 1_000.0) == []


def test_a_nudge_inside_the_range_is_allowed_however_big_the_value():
    # 8 -> 2 is a third of the queue limit; judging a nudge by how large the
    # number is would refuse every change to a setting that happens to be small.
    planned = plan_adjustments(
        [_hypothesis(setting="COMPUTE_QUEUE_LIMIT", value="2")], _current(), AdjustmentLog(), 1_000.0
    )
    assert [item.proposed for item in planned] == ["2"]


def test_a_move_across_more_than_half_the_range_is_refused():
    hypothesis = _hypothesis(setting="MEMORY_REFLECTION_INTERVAL", value="100")
    assert plan_adjustments([hypothesis], _current(), AdjustmentLog(), 1_000.0) == []


def test_a_setting_with_no_known_starting_point_is_left_alone():
    # Without a value to measure from there is no way to tell a nudge from a
    # jump, so the user's setting stays where they put it.
    assert plan_adjustments([_hypothesis()], {}, AdjustmentLog(), 1_000.0) == []


def test_a_setting_is_not_nudged_twice_in_one_pass():
    twice = [_hypothesis(), _hypothesis(value="3")]
    assert len(plan_adjustments(twice, _current(), AdjustmentLog(), 1_000.0)) == 1


def test_a_setting_nudged_recently_is_left_alone():
    # The anti-oscillation rule: a wrong hypothesis in one direction gets
    # corrected by the next one in the other, forever, if the cooldown is not here.
    log = AdjustmentLog()
    log.record("COMPUTE_QUEUE_LIMIT", now=1_000.0)
    assert plan_adjustments([_hypothesis()], _current(), log, 1_000.0) == []
    later = 1_000.0 + ADJUSTMENT_COOLDOWN_SECONDS + 1
    assert plan_adjustments([_hypothesis()], _current(), log, later) != []


def test_a_hypothesis_with_no_evidence_moves_nothing():
    log = AdjustmentLog()
    assert plan_adjustments([_hypothesis(evidence="")], _current(), log, 1_000.0) == []


def test_an_observation_moves_nothing_however_specific_its_number():
    # Measured against the live model: asked to reflect on a session where a
    # render hit the limit, it observed "renders take longer than the timeout"
    # and then proposed a *shorter* one - the opposite of its own evidence. It
    # labelled that observation an insight, and a number is a prescription.
    observed = _hypothesis(kind="insight", setting="COMPUTE_JOB_TIMEOUT_SECONDS", value="600")
    assert plan_adjustments([observed], _current(), AdjustmentLog(), 1_000.0) == []
    # The same number as a prescription does move, which is what makes the rule
    # about the kind and not a blanket refusal of setting changes.
    prescribed = _hypothesis(kind="improvement", setting="COMPUTE_JOB_TIMEOUT_SECONDS", value="600")
    assert plan_adjustments([prescribed], _current(), AdjustmentLog(), 1_000.0) != []


def test_a_setting_for_the_project_moves_nothing():
    hypothesis = _hypothesis(target="project")
    assert plan_adjustments([hypothesis], _current(), AdjustmentLog(), 1_000.0) == []


def test_every_allowed_setting_is_a_bounded_integer():
    for name, rule in SAFE_SETTINGS.items():
        assert rule.minimum < rule.maximum, name
        assert rule.step >= 1, name
        assert rule.why, f"{name} exists without saying what it is for"


# --------------------------------------------------------------------- writing


def test_the_change_lands_in_the_env_file(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_MODEL=m\nCOMPUTE_QUEUE_LIMIT=8\n")

    path, applied = apply_adjustments(str(tmp_path), plan_adjustments(
        [_hypothesis()], _current(), AdjustmentLog(), 1_000.0
    ))

    assert path.endswith(".env")
    assert "COMPUTE_QUEUE_LIMIT=2" in (tmp_path / ".env").read_text()
    assert "COMPUTE_QUEUE_LIMIT: 8 -> 2" in applied


def test_a_setting_that_was_not_there_is_added(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_MODEL=m\n")
    apply_adjustments(str(tmp_path), plan_adjustments([_hypothesis()], _current(), AdjustmentLog(), 1.0))
    assert "COMPUTE_QUEUE_LIMIT=2" in (tmp_path / ".env").read_text()


def test_the_cooldown_survives_a_restart(tmp_path):
    log = AdjustmentLog()
    log.record("COMPUTE_QUEUE_LIMIT", now=42.0)
    save_adjustment_log(str(tmp_path), log)

    assert load_adjustment_log(str(tmp_path)).entries == {"COMPUTE_QUEUE_LIMIT": 42.0}


def test_a_corrupt_cooldown_file_does_not_stop_the_agent(tmp_path):
    (tmp_path / ".minagent").mkdir()
    (tmp_path / ".minagent" / "ajustes.json").write_text("{not json")
    # Worst case is one extra nudge, which is a far better outcome than refusing
    # to run because a log was damaged.
    assert load_adjustment_log(str(tmp_path)).entries == {}


# ------------------------------------------------------------------- document


def test_the_document_keeps_what_was_written_before(tmp_path):
    append_document(str(tmp_path), "## Una\n\nlo primero")
    append_document(str(tmp_path), "## Dos\n\nlo segundo")
    text = (tmp_path / "MEJORAS.md").read_text()
    assert "lo primero" in text and "lo segundo" in text
    # Appended, not rewritten: a proposal nobody has read is not one the agent
    # may quietly delete on the next pass.
    assert text.index("lo primero") < text.index("lo segundo")


def test_the_document_explains_itself_from_the_first_line(tmp_path):
    # It was computed and then dropped, so the file started with a blank line
    # and no header at all - a log with no indication that a person can edit it.
    append_document(str(tmp_path), "## Una\n\nlo primero")
    text = (tmp_path / "MEJORAS.md").read_text()
    assert text.startswith("# MEJORAS")
    assert "Editar" in text


def test_the_document_shows_the_evidence_and_the_way_to_check_it(tmp_path):
    section = format_document_section(parse_hypotheses(_answer(A_HYPOTHESIS)))
    assert "Evidencia" in section
    assert "Cómo comprobarla" in section
    assert "COMPUTE_QUEUE_LIMIT=2" in section


def test_a_nothing_useful_reflection_writes_nothing(tmp_path):
    assert format_document_section([]) == ""
    assert append_document(str(tmp_path), "   ") == ""


def test_an_insight_is_marked_as_one_in_the_document():
    section = format_document_section(parse_hypotheses(_answer(dict(A_HYPOTHESIS, kind="insight"))))
    assert "AJA" in section


def test_the_report_says_what_moved_and_what_did_not():
    assert format_adjustment_report(["A: 1 -> 2"], []) == "autoajustó A: 1 -> 2"
    assert format_adjustment_report([], ["B=3"]) == "propuso B=3"
    assert format_adjustment_report([], []) == ""


def test_a_session_that_taught_something_is_not_reported_as_silence():
    # Concluding two things and changing no number is the common case, and it
    # must not read like a session that taught nothing.
    assert format_adjustment_report([], [], count=2) == "2 hipótesis"
    assert format_adjustment_report(["A: 1 -> 2"], [], count=1) == "1 hipótesis; autoajustó A: 1 -> 2"


def test_the_prompt_names_only_the_settings_that_exist():
    prompt = build_session_prompt(knowledge=[], log=[], stats={})
    text = prompt[0]["content"]
    for name in SAFE_SETTINGS:
        assert name in text
    # An empty list has to be a correct answer, or every pass invents a finding.
    assert "correct answer" in text


def test_reading_a_document_that_is_not_there_is_empty(tmp_path):
    assert read_document(str(tmp_path)) == ""


# ------------------------------------------------------------------- the app


async def test_the_app_reflects_and_writes_both_destinations(monkeypatch, tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.improvement_enabled = True
    app.improvement_auto = True
    app.application_root = str(tmp_path)
    _set_env(monkeypatch, {"COMPUTE_QUEUE_LIMIT": "8", "MEMORY_REFLECTION_INTERVAL": "10"})

    class _Stub:
        async def complete(self, messages, options=None):
            return {"message": {"role": "assistant", "content": _answer(A_HYPOTHESIS)}}

    app.open_ai_client = _Stub()
    (tmp_path / ".env").write_text("COMPUTE_QUEUE_LIMIT=8\n")

    report = await app.reflect_on_session("test")

    assert "COMPUTE_QUEUE_LIMIT: 8 -> 2" in report
    assert "COMPUTE_QUEUE_LIMIT=2" in (tmp_path / ".env").read_text()
    assert (tmp_path / "MEJORAS.md").exists()
    stored = await app.memory_store.recent()
    # The hypothesis lands in the store too, under its own kind, not filed as a
    # procedure that has been proven to work.
    assert any(entry["kind"] == "hypothesis" for entry in stored)


async def test_with_auto_adjustments_off_nothing_is_written(monkeypatch, tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.improvement_enabled = True
    app.improvement_auto = False
    app.application_root = str(tmp_path)
    _set_env(monkeypatch, {"COMPUTE_QUEUE_LIMIT": "8", "MEMORY_REFLECTION_INTERVAL": "10"})

    class _Stub:
        async def complete(self, messages, options=None):
            return {"message": {"role": "assistant", "content": _answer(A_HYPOTHESIS)}}

    app.open_ai_client = _Stub()
    (tmp_path / ".env").write_text("COMPUTE_QUEUE_LIMIT=8\n")

    report = await app.reflect_on_session("test")

    assert "propuso" in report
    assert "COMPUTE_QUEUE_LIMIT=8" in (tmp_path / ".env").read_text()


async def test_a_reflection_that_fails_does_not_raise(tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.improvement_enabled = True

    class _Broken:
        async def complete(self, messages, options=None):
            raise OSError("the endpoint went away")

    app.open_ai_client = _Broken()

    assert await app.reflect_on_session("test") == ""


# ------------------------------------------------------- el modelo de reflexión

SESSION_MODEL = "qwen3.5:9b-q4_K_M"
REFLECTION_MODEL = "gemma4:12b-q3km"


def _fake_vram(monkeypatch: pytest.MonkeyPatch, resident: tuple[str, ...] = ()) -> list[tuple]:
    """Record what the reflection does to the card, without touching it.

    The two models do not fit in 8 GB together, so this is the part of the
    feature with a real cost, and a test that could not see it would pass
    whatever the code did to the user's VRAM.
    """
    events: list[tuple] = []

    async def unload(root_directory):
        events.append(("unload",))
        return VramRelease(models=[SESSION_MODEL])

    async def reload(models):
        events.append(("reload", *models))
        return ""

    async def resident_models():
        return list(resident)

    monkeypatch.setattr("minagent.app.unload_ollama", unload)
    monkeypatch.setattr("minagent.app.reload_ollama", reload)
    monkeypatch.setattr("minagent.app.ollama_resident_models", resident_models)
    return events


async def _reflecting_app(monkeypatch, tmp_path, complete, resident: tuple[str, ...] = ()) -> tuple:
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.improvement_enabled = True
    app.improvement_auto = False
    app.application_root = str(tmp_path)
    app.model = SESSION_MODEL
    app.improvement_model = REFLECTION_MODEL
    _set_env(monkeypatch, {"COMPUTE_QUEUE_LIMIT": "8", "MEMORY_REFLECTION_INTERVAL": "10"})
    app.open_ai_client = _Stub(complete)
    return app, _fake_vram(monkeypatch, resident)


class _Stub:
    """A model stand-in that records which model each request asked for."""

    def __init__(self, reply) -> None:
        self.reply = reply
        self.asked: list[str | None] = []

    async def complete(self, messages, options=None):
        self.asked.append((options or {}).get("model"))
        answer = self.reply((options or {}).get("model"))
        if inspect.isawaitable(answer):
            answer = await answer
        return {"message": {"role": "assistant", "content": answer}}


async def test_the_reflection_runs_on_its_own_model_and_gives_the_card_back(monkeypatch, tmp_path):
    app, events = await _reflecting_app(monkeypatch, tmp_path, lambda model: _answer(A_HYPOTHESIS))

    report = await app.reflect_on_session("test")

    assert app.open_ai_client.asked == [REFLECTION_MODEL]
    assert "COMPUTE_QUEUE_LIMIT" in report
    # 5.5 GB of session model and 5.7 GB of reflection model do not fit on an
    # 8 GB card, so the session one is unloaded first - and the reflection model
    # is unloaded before it comes back, or the reload lands on a full card.
    assert events == [("unload",), ("unload",), ("reload", SESSION_MODEL)]


async def test_a_reflection_model_that_cannot_answer_falls_back_to_the_session_one(monkeypatch, tmp_path):
    async def answer(model):
        if model == REFLECTION_MODEL:
            raise OSError("no such model on this endpoint")
        return _answer(A_HYPOTHESIS)

    app, events = await _reflecting_app(monkeypatch, tmp_path, answer)

    report = await app.reflect_on_session("test")

    # The bigger model is an improvement, not a dependency: a weaker answer from
    # the model that was already loaded beats no reflection at all.
    assert app.open_ai_client.asked == [REFLECTION_MODEL, None]
    assert "COMPUTE_QUEUE_LIMIT" in report
    assert events[-1] == ("reload", SESSION_MODEL)


async def test_the_session_model_comes_back_even_when_the_reflection_model_explodes(monkeypatch, tmp_path):
    # An error type the request helper does not know: the reflection gives up,
    # and the card must not keep the user's model off it because of that.
    def answer(model):
        raise ValueError("something nobody planned for")

    app, events = await _reflecting_app(monkeypatch, tmp_path, answer)

    assert await app.reflect_on_session("test") == ""
    assert events == [("unload",), ("unload",), ("reload", SESSION_MODEL)]


async def test_without_a_reflection_model_nothing_is_unloaded(monkeypatch, tmp_path):
    app, events = await _reflecting_app(monkeypatch, tmp_path, lambda model: _answer(A_HYPOTHESIS))
    app.improvement_model = ""

    await app.reflect_on_session("test")

    assert app.open_ai_client.asked == [None]
    assert events == []


async def test_naming_the_session_model_as_its_own_reflector_changes_nothing(monkeypatch, tmp_path):
    app, events = await _reflecting_app(monkeypatch, tmp_path, lambda model: _answer(A_HYPOTHESIS))
    app.improvement_model = SESSION_MODEL

    await app.reflect_on_session("test")

    # Unloading a model to ask that same model would cost a load for nothing.
    assert events == []


async def test_a_reflection_model_that_is_already_loaded_is_not_cycled(monkeypatch, tmp_path):
    app, events = await _reflecting_app(
        monkeypatch, tmp_path, lambda model: _answer(A_HYPOTHESIS), resident=(REFLECTION_MODEL,)
    )

    await app.reflect_on_session("test")

    # Something else on this machine already had it warm. The swap would evict
    # the model it is about to use, load it again for the request, and evict it
    # a second time to put the other one back.
    assert app.open_ai_client.asked == [REFLECTION_MODEL]
    assert events == []


async def test_thinking_is_only_switched_off_for_the_model_it_was_measured_on(monkeypatch, tmp_path):
    sent: list[dict] = []

    class _Recorder:
        async def complete(self, messages, options=None):
            seen = dict(options or {})
            sent.append(seen)
            return {"message": {"role": "assistant", "content": _answer(A_HYPOTHESIS) if seen.get("model") else "{}"}}

    app, _ = await _reflecting_app(monkeypatch, tmp_path, lambda model: "")
    app.open_ai_client = _Recorder()

    await app._ask_with_reflection_model([{"role": "user", "content": "x"}])
    await app._ask_about([{"role": "user", "content": "x"}])

    # The no-thinking flag is what cut a review on the 9B from 58 s to 1.5 s.
    # It is not a property of the request: it is a measurement of one model, so
    # it stops there rather than being imposed on whatever a user names.
    assert "extra_body" not in sent[0]
    assert sent[1]["extra_body"] == {"reasoning_effort": "none"}


async def test_an_answer_that_is_not_a_hypothesis_list_is_asked_again(monkeypatch, tmp_path):
    # Measured against the real reflection model: it sometimes answers with the
    # object cut off mid-sentence, and the parser cannot read half a hypothesis.
    app, _ = await _reflecting_app(
        monkeypatch,
        tmp_path,
        lambda model: '{"hypotheses": [{"title": "T", "kind": "improvement", "state'
        if model == REFLECTION_MODEL
        else _answer(A_HYPOTHESIS),
    )

    report = await app.reflect_on_session("test")

    assert app.open_ai_client.asked == [REFLECTION_MODEL, None]
    assert "COMPUTE_QUEUE_LIMIT" in report


async def test_the_periodic_reflection_counts_turns_not_reviews(monkeypatch, tmp_path):
    # A session can be full of work and still leave nothing to cull. Tying the
    # reflection to the review would let an empty log postpone it for ever.
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.improvement_enabled = True
    app.improvement_interval = 2
    _set_env(monkeypatch, {"COMPUTE_QUEUE_LIMIT": "8", "MEMORY_REFLECTION_INTERVAL": "10"})

    asked: list[str] = []

    class _Stub:
        async def complete(self, messages, options=None):
            asked.append(messages[0]["content"][:40])
            return {"message": {"role": "assistant", "content": '{"hypotheses": []}'}}

    app.open_ai_client = _Stub()

    # Three turns whose memory work finds nothing to cull.
    for _ in range(3):
        app._tools_used_this_turn = []
        await app.capture_experience("no tools, so nothing captured")

    assert asked, "the reflection never ran because the log was empty"
    assert app._session_reflections == 1


async def test_a_session_that_learned_nothing_says_so(tmp_path):
    app = _memory_app(tmp_path)
    await app.initialize_optional_features()
    app.improvement_enabled = True

    class _Empty:
        async def complete(self, messages, options=None):
            return {"message": {"role": "assistant", "content": '{"hypotheses": []}'}}

    app.open_ai_client = _Empty()
    app._stdout = _FakeOutput()

    await app.handle_improvement_command("now")
    assert "No salió nada aplicable" in app._stdout.text
