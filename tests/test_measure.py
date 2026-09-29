"""Validation tests: whether a change the agent made actually helped.

The verdicts are the whole point of the phase, so each rule gets a case that
would go the wrong way without it: a change that is neutral, a change that buys
fewer failures with far more context, a window too short to judge, and the
revert itself.
"""

from __future__ import annotations

import json

from minagent.measure import (
    MAX_TOLERATED_REGRESSION,
    MIN_GAIN,
    MIN_TURNS_PER_TRIAL,
    Scorecard,
    Trial,
    describe_trial,
    judge,
    load_trial,
    save_trial,
)
from tests.test_memory import _memory_app


def _trial(before: Scorecard, after: Scorecard) -> Trial:
    return Trial(setting="COMPUTE_QUEUE_LIMIT", previous="8", proposed="4", reason="r", before=before, after=after)


# ------------------------------------------------------------------ the rates


def test_a_rate_is_per_turn_not_a_total():
    card = Scorecard(turns=10, tool_errors=2, job_refusals=1, job_failures=1)
    # Four failures over ten turns. The same four over a two-turn session would
    # be a rate of 2.0, which is why a total is never the thing compared.
    assert card.failure_rate() == 0.4


def test_an_empty_window_is_zero_rather_than_a_division_error():
    card = Scorecard()
    assert card.failure_rate() == 0.0
    assert card.cost_rate() == 0.0


def test_a_damaged_scorecard_reads_as_empty():
    # A corrupt trial file must not stop the agent from starting; it only means
    # the measurement is forgotten, not that the agent is stuck.
    assert Scorecard.from_json("{not json").turns == 0
    assert Scorecard.from_json("[]").turns == 0
    assert Scorecard.from_json('{"turns": -4}').turns == 0


# ----------------------------------------------------------------- the verdict


def test_a_change_that_fewers_failures_stays():
    verdict = judge(_trial(Scorecard(turns=10, tool_errors=5), Scorecard(turns=10, tool_errors=1)))
    assert verdict.startswith("keep")
    assert "improved" in verdict


def test_a_change_that_fewers_tokens_stays():
    verdict = judge(_trial(
        Scorecard(turns=10, tool_tokens=5000),
        Scorecard(turns=10, tool_tokens=2000),
    ))
    assert verdict.startswith("keep")
    assert "context cost" in verdict


def test_a_change_that_breaks_things_is_reverted():
    verdict = judge(_trial(Scorecard(turns=10, tool_errors=1), Scorecard(turns=10, tool_errors=6)))
    assert verdict.startswith("revert")
    assert "worse" in verdict


def test_a_change_that_buys_fewer_failures_with_far_more_context_is_reverted():
    # The trade nobody asked for: smoother, and much more expensive. The
    # failure rate improved by 40% and the verdict is still no.
    verdict = judge(_trial(
        Scorecard(turns=10, tool_errors=5, tool_tokens=1000),
        Scorecard(turns=10, tool_errors=3, tool_tokens=9000),
    ))
    assert verdict.startswith("revert")
    assert "trade" in verdict


def test_a_change_that_does_nothing_is_reverted():
    # Leaving it in spends a setting's cooldown on nothing, and every later
    # hypothesis for that setting is blocked while it sits there.
    verdict = judge(_trial(Scorecard(turns=10, tool_errors=2, tool_tokens=1000),
                           Scorecard(turns=10, tool_errors=2, tool_tokens=1000)))
    assert verdict.startswith("revert")
    assert "nothing measurable improved" in verdict


def test_a_change_is_not_judged_on_a_window_too_short():
    verdict = judge(_trial(Scorecard(turns=10, tool_errors=5), Scorecard(turns=2, tool_errors=0)))
    # Two clean turns are not evidence of anything, and reverting on them would
    # be worse than leaving a change in place for a while.
    assert verdict.startswith("undecided")
    assert f"{MIN_TURNS_PER_TRIAL}" in verdict


def test_failures_appearing_where_there_were_none_is_a_revert():
    verdict = judge(_trial(Scorecard(turns=10), Scorecard(turns=10, tool_errors=2)))
    assert verdict.startswith("revert")


def test_the_tolerances_are_not_being_met_by_a_hair():
    # A gain just under the bar is not a gain; the bar is what makes "quiet
    # afternoon" and "real effect" different things. Big enough numbers that
    # rounding is not what is being measured.
    small = Scorecard(turns=1000, tool_errors=100, tool_tokens=100_000)
    barely = Scorecard(
        turns=1000,
        tool_errors=int(100 * (1 - MIN_GAIN / 2)),
        tool_tokens=100_000,
    )
    assert judge(_trial(small, barely)).startswith("revert")
    clearly = Scorecard(turns=1000, tool_errors=int(100 * (1 - MIN_GAIN * 2)), tool_tokens=100_000)
    assert judge(_trial(small, clearly)).startswith("keep")
    assert MAX_TOLERATED_REGRESSION > 0


# ------------------------------------------------------------------ the trial


def test_a_trial_survives_a_restart(tmp_path):
    trial = _trial(Scorecard(turns=10), Scorecard(turns=3))
    save_trial(str(tmp_path), trial)

    loaded = load_trial(str(tmp_path))
    assert loaded is not None
    assert loaded.setting == "COMPUTE_QUEUE_LIMIT"
    assert loaded.after.turns == 3
    # Persisted on every turn, because a measurement that restarted on every
    # start would let a change that was never judged sit there for ever.
    assert (tmp_path / ".minagent" / "prueba.json").exists()


def test_clearing_a_trial_removes_the_file(tmp_path):
    save_trial(str(tmp_path), _trial(Scorecard(), Scorecard()))
    save_trial(str(tmp_path), None)
    assert load_trial(str(tmp_path)) is None
    assert not (tmp_path / ".minagent" / "prueba.json").exists()


def test_a_damaged_trial_file_reads_as_no_trial(tmp_path):
    (tmp_path / ".minagent").mkdir()
    (tmp_path / ".minagent" / "prueba.json").write_text("{not json")
    assert load_trial(str(tmp_path)) is None


def test_the_panel_says_what_is_being_measured(tmp_path):
    assert "No hay ningún cambio" in describe_trial(None)
    open_trial = _trial(Scorecard(turns=10), Scorecard(turns=3))
    assert "en medición" in describe_trial(open_trial)
    done = _trial(Scorecard(turns=10), Scorecard(turns=10))
    done.judged, done.kept, done.verdict = True, False, "revert: worse"
    assert "revertido" in describe_trial(done)


# ------------------------------------------------------------------- the loop


class _Stub:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def complete(self, messages, options=None):
        return {"message": {"role": "assistant", "content": self.answer}}


def _hypothesis_answer(setting: str, value: str) -> str:
    return json.dumps({
        "hypotheses": [{
            "title": "La cola se atasca", "kind": "improvement", "target": "agent",
            "statement": "Se acumulan", "evidence": "3 rechazos y 2 en cola",
            "expected": "Menos de golpe", "verify": "Que no supere 2",
            "setting": setting, "value": value, "reason": "porque se atasca",
        }]
    })


async def _measuring_app(monkeypatch, tmp_path, answer: str):
    app = _memory_app(tmp_path)
    app.improvement_enabled = True
    app.improvement_auto = True
    app.application_root = str(tmp_path)
    monkeypatch.setenv("COMPUTE_QUEUE_LIMIT", "8")
    monkeypatch.setenv("MEMORY_REFLECTION_INTERVAL", "10")
    (tmp_path / ".env").write_text("COMPUTE_QUEUE_LIMIT=8\nMEMORY_REFLECTION_INTERVAL=10\n")
    # Opened before the stub goes in: it asks the endpoint for a context
    # length on the way up, and a stand-in that answers the reflection but not
    # that is a stub that is wrong in a way no test here would catch.
    await app.initialize_optional_features()
    app.open_ai_client = _Stub(answer)
    return app


async def test_a_change_is_applied_first_and_judged_later(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))

    report = await app.reflect_on_session("test")

    # It is on probation, not decided: the word matters more than the number.
    assert "en medición" in report
    assert load_trial(tmp_path) is not None
    assert "COMPUTE_QUEUE_LIMIT=4" in (tmp_path / ".env").read_text()

    # And the window is empty, so nothing has been judged yet.
    assert judge(load_trial(tmp_path)).startswith("undecided")


async def test_a_change_that_helps_is_kept_after_the_window(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    # A baseline that was failing: 5 errors in 10 turns. The baseline is the
    # session before the change, not the empty window that follows it.
    app._session_tool_errors = 5
    app._session_turns = 10
    await app.reflect_on_session("test")

    assert load_trial(tmp_path).before.failure_rate() == 0.5

    for _ in range(10):
        await app.capture_experience("x")

    trial = load_trial(tmp_path)
    assert trial.judged and trial.kept, trial.verdict
    assert "COMPUTE_QUEUE_LIMIT=4" in (tmp_path / ".env").read_text()


async def test_a_change_that_hurts_is_reverted_by_itself(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    # A baseline that was failing a lot, and a window that failed more.
    app._session_tool_errors = 5
    app._session_turns = 10
    await app.reflect_on_session("test")

    said = ""
    for _ in range(10):
        app._session_tool_errors += 1
        said = app._advance_trial() or said

    assert "vuelve a 8" in said
    assert load_trial(tmp_path).kept is False
    # The file is what the next start reads, so it has to be right.
    assert "COMPUTE_QUEUE_LIMIT=8" in (tmp_path / ".env").read_text()


async def test_only_one_change_is_tried_at_a_time(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await app.reflect_on_session("test")

    # Two changes at once would make the verdicts unreadable: if the pair
    # improved, nothing says which of them did. The hypothesis is still
    # recorded - the guard is about moving a second setting, not about silence.
    report = await app.reflect_on_session("otro")
    assert "hipótesis" in report
    assert load_trial(tmp_path).proposed == "4"
    assert "COMPUTE_QUEUE_LIMIT=4" in (tmp_path / ".env").read_text()
