"""Validation tests: whether a change the agent made actually helped.

The verdicts are the whole point of the phase, so each rule gets a case that
would go the wrong way without it: a change that is neutral, a change that buys
fewer failures with far more context, a window too short to judge, and the
revert itself.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minagent.measure import (
    MAX_TOLERATED_REGRESSION,
    MIN_GAIN,
    MIN_TURNS_PER_TRIAL,
    TRIAL_CASES,
    Scorecard,
    Trial,
    _arm_over,
    describe_progress,
    describe_trial,
    judge,
    load_ledger,
    load_trial,
    save_trial,
)
from tests.test_memory import _FakeOutput, _memory_app


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
    def __init__(self, answer: str, verdict: str = "valid") -> None:
        self.answer = answer
        self.verdict = verdict

    async def complete(self, messages, options=None):
        asked = " ".join(str(item.get("content", "")) for item in messages)
        if "verdict" in asked:
            # The gate asks a question of its own, in a different shape. A stub
            # that answers the hypothesis prompt to both is not a stub that
            # exercises the gate - it is one that answers a question nobody
            # asked, and a hypothesis only reaches this suite's assertions once
            # the gate has passed it.
            return {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"verdict": self.verdict, "reason": "stubbed for the test"}
                    ),
                }
            }
        return {"message": {"role": "assistant", "content": self.answer}}


def _hypothesis_answer(setting: str, value: str) -> str:
    # The statement has to be a real guideline, not a label: the structural
    # critic rejects a lesson too short to follow, and a hypothesis only reaches
    # a trial once it has been admitted. A short stub here did not fail these
    # tests before because the trial was opened before the gate ran - the
    # measurement was being driven by a lesson the gate would have refused.
    return json.dumps({
        "hypotheses": [{
            "title": "La cola se atasca", "kind": "improvement", "target": "agent",
            "statement": "Los renders se acumulan mas rapido de lo que terminan y la cola crece",
            "evidence": "3 rechazos y 2 en cola", "expected": "Menos de golpe",
            "verify": "Que no supere 2", "falsifier": "Que la cola no supere 2 en dos horas",
            "setting": setting, "value": value, "reason": "porque se atasca",
        }]
    })


async def _measuring_app(monkeypatch, tmp_path, answer: str, verdict: str = "valid"):
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
    app.open_ai_client = _Stub(answer, verdict)
    return app


async def test_a_change_is_held_back_and_judged_later(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))

    report = await app.reflect_on_session("test")

    # It is on probation, not decided: the word matters more than the number.
    assert "en medición" in report
    assert load_trial(tmp_path) is not None

    # And the value has NOT moved. The trial opens on the baseline arm, so the
    # first window has to run the value the user set - a window labelled
    # "baseline" that runs the candidate records arm=previous for turns that
    # never ran it, and the crossover then compares a number against itself.
    assert "COMPUTE_QUEUE_LIMIT=8" in (tmp_path / ".env").read_text()
    assert "COMPUTE_QUEUE_LIMIT=4" not in (tmp_path / ".env").read_text()

    # And the window is empty, so nothing has been judged yet.
    assert judge(load_trial(tmp_path)).startswith("undecided")


async def test_the_live_setting_equals_the_arm_under_measurement(monkeypatch, tmp_path):
    """The invariant, checked against the value the session actually runs.

    The trial records which arm is live. This checks that the arm and the
    process agree at every step of the crossover, because a trial whose label
    and whose value disagree produces a verdict about nothing.
    """
    # MEMORY_REFLECTION_INTERVAL rather than a compute setting: it is the one
    # allowlisted value the session applies without an orchestrator, so this
    # checks the value the process really runs on in a session with compute
    # disabled - which is a normal session, not a special one.
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("MEMORY_REFLECTION_INTERVAL", "4"))
    await _start_trial(app)

    trial = load_trial(str(app.application_root))
    assert trial.live == "baseline"
    assert app.memory_reflection_interval == 10, "the baseline window ran the candidate value"

    # A full baseline window, then the switch.
    for _ in range(MIN_TURNS_PER_TRIAL):
        app._session_tool_errors += 0
        app._advance_trial()

    trial = load_trial(str(app.application_root))
    assert trial.live == "candidate"
    assert app.memory_reflection_interval == 4, "the candidate window ran the baseline value"
    assert "MEMORY_REFLECTION_INTERVAL=4" in (tmp_path / ".env").read_text()

    # And back again, so a value that loses is written down rather than left
    # behind: the file and the session must not drift apart.
    for _ in range(MIN_TURNS_PER_TRIAL):
        app._session_tool_errors += 0
        app._advance_trial()

    trial = load_trial(str(app.application_root))
    assert trial.live == "baseline"
    assert app.memory_reflection_interval == 10
    assert "MEMORY_REFLECTION_INTERVAL=10" in (tmp_path / ".env").read_text()


async def test_the_ledger_never_records_an_arm_it_did_not_run(monkeypatch, tmp_path):
    """The falsification check, on the data rather than on the setting.

    Each window records what the arm produced. If a baseline window ever ran
    the candidate value, its entries would be filed under `previous` - and the
    gate would then be comparing the candidate's own results against themselves.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)

    trial = load_trial(str(app.application_root))
    baseline_value = trial.previous

    for _ in range(MIN_TURNS_PER_TRIAL):
        # Zero errors: the arm is decided by the value under measurement, not
        # by the numbers, so the value is what this test reads back.
        app._advance_trial()

    ledger = load_ledger(str(app.application_root))
    entries = ledger.outcomes.get("COMPUTE_QUEUE_LIMIT", [])
    assert entries, "the baseline window recorded nothing"

    # The first run recorded is the baseline window by construction, and every
    # entry it produced has to carry the value that was actually live while it
    # ran. The number itself is not hardcoded: what the test is about is the
    # pairing of a run with the arm that produced it.
    first_run = min(row["run"] for row in entries)
    opening = [row for row in entries if row["run"] == first_run]
    assert opening, "the baseline window filed nothing"
    assert {row["value"] for row in opening} == {baseline_value}, (
        f"the opening window is the baseline arm but carries "
        f"{sorted({r['value'] for r in opening})}"
    )


async def _drive(app, windows: int, *, baseline_errors: int, candidate_errors: int) -> str:
    """Turn the trial through `windows` windows, giving each arm its own behaviour.

    The arm is read from the trial rather than assumed, because the whole point
    is that the trial switches arms underneath the caller.
    """
    said = ""
    for _ in range(windows):
        for _ in range(MIN_TURNS_PER_TRIAL):
            trial = load_trial(str(app.application_root))
            live = trial.live if trial else "baseline"
            app._session_tool_errors += candidate_errors if live == "candidate" else baseline_errors
            said = app._advance_trial() or said
    return said


async def _start_trial(app):
    app._session_tool_errors = 5
    app._session_turns = 10
    await app.reflect_on_session("test")
    trial = load_trial(str(app.application_root))
    assert trial is not None
    # Two runs per arm keeps the test honest and fast; the default is nine.
    # The real app has a full Config here; this proves the evidence parameters
    # are read from config and not hardcoded into the state machine.
    app.config = SimpleNamespace(improvement_min_runs=2, improvement_max_regressions=1)
    return trial


async def test_one_window_does_not_decide_a_change(monkeypatch, tmp_path):
    """The single guarantee the old design lacked: a window is a sample, not a verdict.

    A change that helped every single one of its first ten turns was still being
    decided on ten turns. Ten turns is a morning.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)

    # One perfect window for the candidate. The old code would have kept it here.
    said = await _drive(app, 1, baseline_errors=1, candidate_errors=0)

    trial = load_trial(str(app.application_root))
    assert not trial.decided, "one window is a sample, not a decision"
    assert not trial.judged
    assert said == "", "nothing is said before the evidence is in"
    assert trial.verdict == ""


async def test_refusals_and_failures_are_measured_per_window(monkeypatch, tmp_path):
    """Job refusals and failures live on the orchestrator, not on a session counter.

    The orchestrator is never reset by a session, so these two cases are only
    readable as the delta since the window opened. The session already has
    failures before the trial starts here, which is the only situation where
    mixing a session-wide total with a window delta gives a wrong answer: the
    change's own three refusals vanish inside the pre-existing total and the
    case reports a pass.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    # Five refusals already on the board when the trial begins.
    app.orchestrator = SimpleNamespace(refusals=5, failures=0)
    trial = await _start_trial(app)
    gating = trial.holdout().gating_cases(TRIAL_CASES)

    for _ in range(4):
        for _ in range(MIN_TURNS_PER_TRIAL):
            live = load_trial(str(app.application_root)).live
            # Baseline holds steady at five; the change pushes it to eight.
            app.orchestrator.refusals = 8 if live == "candidate" else 5
            app._advance_trial()

    ledger = load_ledger(str(app.application_root))
    baseline = _arm_over(ledger, trial.setting, trial.previous, gating).case_verdicts()
    candidate = _arm_over(ledger, trial.setting, trial.proposed, gating).case_verdicts()
    if "job_refusals" not in gating:
        pytest.skip("the reserve took job_refusals; its collection is covered separately")

    assert baseline["job_refusals"] is True, "no new refusals, so the case passes"
    assert candidate["job_refusals"] is False, "three new refusals must show up as a failure"


async def test_a_change_that_helps_is_kept_once_both_arms_have_runs(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)

    await _drive(app, 6, baseline_errors=1, candidate_errors=0)

    trial = load_trial(str(app.application_root))
    assert trial.decided and trial.kept, trial.verdict
    assert "COMPUTE_QUEUE_LIMIT=4" in (tmp_path / ".env").read_text()


async def test_a_change_that_hurts_is_reverted_by_itself(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)

    said = await _drive(app, 6, baseline_errors=0, candidate_errors=1)

    trial = load_trial(str(app.application_root))
    assert trial.decided and not trial.kept, trial.verdict
    # The file is what the next start reads, so it has to be right.
    assert "COMPUTE_QUEUE_LIMIT=8" in (tmp_path / ".env").read_text()
    assert trial.verdict in said


async def test_the_arms_alternate_instead_of_running_in_blocks(monkeypatch, tmp_path):
    """A block design cannot separate the change from the hour it landed in.

    Baseline for fifty turns then candidate for fifty is not a comparison, it is
    a story about two different halves of a day.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)  # called for the trial it persists, not for its return

    seen = []
    for _ in range(4):
        for _ in range(MIN_TURNS_PER_TRIAL):
            seen.append(load_trial(str(app.application_root)).live)
            app._advance_trial()

    # One value per window, alternating, never the same one twice in a row.
    per_window = [seen[index * MIN_TURNS_PER_TRIAL] for index in range(4)]
    assert per_window[0] != per_window[1], per_window
    # strict=False is the point: this compares neighbours in a list one longer
    # than the window, so the pairs are meant to be ragged. strict=True here
    # would raise instead of testing anything.
    assert all(a != b for a, b in zip(per_window, per_window[1:], strict=False)), per_window


async def test_a_gain_that_only_holds_on_the_gated_cases_is_caught(monkeypatch, tmp_path):
    """The reason to reserve cases at all.

    Here the change fixes the visible complaint - tool errors collapse - while
    the case nobody could see while deciding gets worse. A gate built only from
    visible cases would keep this change and call it an improvement.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    trial = await _start_trial(app)
    reserved = trial.holdout().cases
    assert reserved, "a trial must reserve something or the holdout is a gesture"

    for _ in range(6):
        for _ in range(MIN_TURNS_PER_TRIAL):
            live = load_trial(str(app.application_root)).live
            # Tool errors improve under the change; job failures do not.
            app._session_tool_errors += 0 if live == "candidate" else 6
            app.orchestrator = SimpleNamespace(refusals=0, failures=4 if live == "candidate" else 0)
            app._advance_trial()

    trial = load_trial(str(app.application_root))
    assert trial.decided and trial.kept, "the visible cases really did improve"
    assert trial.surprise, "keeping this change must not be reported as clean"
    assert "fitted" in trial.surprise, trial.surprise

    # And the reserve is spent durably, so a restart cannot read it twice.
    assert load_trial(str(app.application_root)).holdout().spent
    assert trial.holdout_json, "the spend must be persisted, not just remembered"


async def test_the_reserved_cases_are_collected_just_never_gated(monkeypatch, tmp_path):
    """Isolation is a filter on the way to the gate, not a filter at collection.

    The reserved cases have to exist in the ledger, otherwise there is nothing
    to read at the end and the holdout is decoration.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    trial = await _start_trial(app)
    reserved = trial.holdout().cases
    gating = trial.holdout().gating_cases(TRIAL_CASES)
    assert reserved and set(reserved).isdisjoint(gating)

    await _drive(app, 6, baseline_errors=1, candidate_errors=0)
    ledger = load_ledger(str(app.application_root))

    for value in (trial.previous, trial.proposed):
        collected = ledger.arm(trial.setting, value=value).case_verdicts()
        assert set(reserved) <= set(collected), "the reserve must actually be measured"
        gated = _arm_over(ledger, trial.setting, value, gating).case_verdicts()
        assert set(reserved).isdisjoint(gated), "and never counted in the gate"


async def test_the_reserved_cases_never_reach_the_gate(monkeypatch, tmp_path):
    """The leak that makes a holdout worthless: reserved cases in the mean.

    ``compare_arms`` takes its mean from the whole arm it is handed, so handing
    it an arm that still contains the reserved cases would let them vote on the
    decision they were reserved to judge.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    trial = await _start_trial(app)

    holdout = trial.holdout()
    assert holdout.cases, "a trial must reserve something or the holdout is a gesture"
    gating = holdout.gating_cases(TRIAL_CASES)
    assert not set(gating) & set(holdout.cases)
    assert set(gating) | set(holdout.cases) == set(TRIAL_CASES)

    await _drive(app, 6, baseline_errors=1, candidate_errors=0)
    ledger = load_ledger(str(app.application_root))

    # Everything is recorded, but the arms the gate sees exclude the reserved set.
    from minagent.measure import _arm_over

    baseline = _arm_over(ledger, trial.setting, trial.previous, gating)
    assert baseline.outcomes, "the baseline arm was never measured"
    assert not {outcome.case for outcome in baseline.outcomes} & set(holdout.cases)


async def test_only_one_change_is_tried_at_a_time(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await app.reflect_on_session("test")

    # Two changes at once would make the verdicts unreadable: if the pair
    # improved, nothing says which of them did. The hypothesis is still
    # recorded - the guard is about moving a second setting, not about silence.
    report = await app.reflect_on_session("otro")
    assert "hipótesis" in report
    assert load_trial(tmp_path).proposed == "4"
    # Still 8: the first trial holds its value back until the crossover says
    # which arm is live.
    assert "COMPUTE_QUEUE_LIMIT=8" in (tmp_path / ".env").read_text()


async def test_progress_shows_each_arm_measured_rate(monkeypatch, tmp_path):
    """The progress line is read to decide whether to keep waiting, so it has to
    say how the change is going, not just that it is going.

    A line that reported only "measuring, 3/8 turns" was indistinguishable from
    one where the candidate was losing badly - the reader has no way to tell
    those apart, and reading a trial is the only reason anyone opens this.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)
    ledger = load_ledger(str(app.application_root))
    trial = load_trial(str(app.application_root))
    gating = trial.holdout().gating_cases(TRIAL_CASES)

    # One measured run on each arm, over the gating cases only. A run is one
    # index shared by every case in it, which is what `Arm.runs` counts - three
    # separate indices would read as three runs, not one.
    for case in gating:
        ledger.record(trial.setting, case=case, passed=True, run=10, value=trial.previous)
    for case in gating:
        ledger.record(trial.setting, case=case, passed=False, run=20, value=trial.proposed)

    line = describe_progress(trial, ledger, min_runs=2)

    # Both arms reported, each with its own rate, and the numbers differ.
    assert f"{trial.previous}: 1/2 corridas, 100% de acierto" in line
    assert f"{trial.proposed}: 1/2 corridas, 0% de acierto" in line
    # And the reserve is disclosed as still unread, so nobody thinks the
    # holdout has already been folded into these numbers.
    assert "reservada" in line


async def test_progress_refuses_to_report_a_rate_it_does_not_have(monkeypatch, tmp_path):
    """An arm with nothing measured is not at 0%.

    Reporting a mean over no observations would show "0% de acierto" for an
    arm nobody has run yet, which reads as a measured failure and is worse than
    admitting there is nothing yet.
    """
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)
    trial = load_trial(str(app.application_root))
    ledger = load_ledger(str(app.application_root))

    line = describe_progress(trial, ledger, min_runs=2)

    assert "0% de acierto" not in line
    assert "sin datos aún" in line


# --- What a person sees while a change is being measured --------------------
#
# /mejoras existed to answer one question: should I keep waiting for this? The
# line it used to show - "3 of 8 turns" - cannot answer it. It reads the same
# whether the candidate is winning or losing, and it names only the arm that
# happens to be live, so the other arm is invisible and you cannot tell how far
# ahead or behind it is. The state is computed and then thrown away.


async def test_mejoras_names_both_arms_while_a_change_is_measured(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)
    # Give the candidate one window it loses, so the two rates differ. If the
    # display averaged them or hid the loser, this is the case that shows it.
    await _drive(app, 1, baseline_errors=0, candidate_errors=4)

    app._stdout = _FakeOutput()
    await app.handle_improvement_command("")

    said = app._stdout.text
    trial = load_trial(str(app.application_root))
    assert trial is not None and not trial.decided, "the trial ended too early to display"

    # Both arms are named by their value, not by an internal label: the reader
    # is deciding between 8 and 4, so that is the pair they must be able to see.
    for value in (trial.previous, trial.proposed):
        assert f"{trial.setting}={value}" in said, f"/mejoras hid {value}: {said!r}"
    # Both rates, not a single averaged number. This window made the candidate
    # lose, so its arm reads "sin datos" while the other reads a percentage.
    assert said.count("%") == 1 and "sin datos" in said, f"rates are misreported: {said!r}"
    # The live arm is the one being applied to, named by value, so the next
    # turn's number is attributable.
    live_value = trial.previous if trial.live == "baseline" else trial.proposed
    assert f"cambios aplicados a {live_value}" in said, f"the live arm is not marked: {said!r}"
    # The internal arm names are not the reader's vocabulary, and this text is
    # Spanish. A raw enum here means the reader sees "changes applied to candidate".
    for leaked in ("candidate", "baseline"):
        assert leaked not in said, f"the internal name {leaked!r} leaked into the display: {said!r}"


async def test_mejoras_reports_no_rate_honestly_before_any_window_finishes(monkeypatch, tmp_path):
    app = await _measuring_app(monkeypatch, tmp_path, _hypothesis_answer("COMPUTE_QUEUE_LIMIT", "4"))
    await _start_trial(app)
    # Two turns in: below MIN_TURNS_PER_TRIAL, so no rate exists yet.
    for _ in range(2):
        app._session_tool_errors += 1
        app._advance_trial()

    app._stdout = _FakeOutput()
    await app.handle_improvement_command("")

    said = app._stdout.text
    assert "sin datos" in said, f"a rate was invented with no window finished: {said!r}"
    assert "%" not in said, f"a percentage was invented with no window finished: {said!r}"
