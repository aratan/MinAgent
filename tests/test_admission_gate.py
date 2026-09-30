"""The admission gate, where a hypothesis becomes something the agent can read.

These tests are about the write that must not happen. A screen nobody can
inspect is a screen nobody can trust, so every case here checks two things at
once: that the store was left alone, and that the reason it was left alone is
written down and can be read back.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minagent.admission import load_admission_log  # noqa: E402
from minagent.app import MinAgent  # noqa: E402
from minagent.improvement import Hypothesis  # noqa: E402
from minagent.measure import load_trial  # noqa: E402
from minagent.memory import MemoryStore  # noqa: E402
from minagent.workspace import WorkspaceAccess  # noqa: E402


class _FakeOutput:
    def write(self, _text: str) -> None:
        pass


class _Stub:
    """Answers whatever the gate asks, in the shape that question expects."""

    def __init__(self, answer: str | None) -> None:
        self.answer = answer

    async def complete(self, messages, options=None):
        if self.answer is None:
            raise OSError("endpoint is down")
        return {"message": {"role": "assistant", "content": self.answer}}


def _app(tmp_path) -> MinAgent:
    app = MinAgent(stdout=_FakeOutput())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "Test"
    app.memory_enabled = True
    app.memory_db_path = str(tmp_path / ".minagent" / "memory.db")
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    return app


def _hypothesis(**overrides) -> Hypothesis:
    fields = {
        "title": "La cola se atasca con la memoria llena",
        "statement": (
            "Bajar el límite de la cola de cómputo cuando las tareas pesadas "
            "se acumulan evita que la memoria de reintentos crezca sin control."
        ),
        "evidence": "3 rechazos y 2 trabajos esperando en 20 turnos",
        "expected": "Menos rechazos por saturación",
        "verify": "Que el límite no supere 4",
        "target": "agent",
        "kind": "improvement",
    }
    fields.update(overrides)
    return Hypothesis(**fields)


async def _store(tmp_path) -> MemoryStore:
    store = MemoryStore(str(tmp_path / "memory.db"))
    await store.initialize()
    return store


def _verdict(value: str) -> str:
    return json.dumps({"verdict": value, "reason": "porque sí"})


async def _admit(app, tmp_path, hypothesis=None, *, answer='{"verdict": "valid"}', known=0, duplicate=False):
    """Run the gate and report what the store was told.

    ``known`` seeds context with lessons that are deliberately *unrelated*, so
    the deterministic duplicate and contradiction checks have nothing to find
    and the consistency reviewer is genuinely consulted. ``duplicate`` seeds an
    exact restatement instead, which is the case the model never needs to see.
    """
    store = await _store(tmp_path)
    for index in range(known or (1 if duplicate else 0)):
        text = (
            _hypothesis().statement
            if duplicate
            else (
                f"Cuando el disco se llena de capturas conviene archivarlas antes de que "
                f"el índice se degrade, caso {index}, y volver a numerarlas después."
            )
        )
        await store.remember("procedure", f"Regla {index}", text, ["regla"])
    app.open_ai_client = _Stub(answer)
    written: list[dict] = []
    original = store.remember

    async def spy(*args, **kwargs):
        written.append(kwargs or {})
        return await original(*args, **kwargs)

    store.remember = spy
    await app._store_hypothesis(store, hypothesis or _hypothesis())
    return written


# ----------------------------------------------------------------- the refusals


async def test_a_malformed_lesson_never_reaches_the_store(tmp_path):
    # No trigger at all: a guideline with no condition is either always on and
    # ignored, or never on and dead weight. Both are worse than refusing.
    app = _app(tmp_path)
    written = await _admit(app, tmp_path, _hypothesis(verify="", trigger=""), answer=_verdict("valid"))

    assert written == []
    refusals = load_admission_log(str(tmp_path)).refused()
    assert refusals and "structural" in refusals[0]["reason"]


async def test_the_store_is_untouched_when_the_reviewer_cannot_be_reached(tmp_path):
    app = _app(tmp_path)
    written = await _admit(app, tmp_path, answer=None, known=1)

    assert written == []
    refusals = load_admission_log(str(tmp_path)).refused()
    assert refusals
    assert "unavailable" in refusals[0]["reason"] or "could not be reached" in refusals[0]["reason"]


async def test_an_unreadable_answer_is_a_refusal_not_a_pass(tmp_path):
    app = _app(tmp_path)
    written = await _admit(app, tmp_path, answer="creo que sí, pero no estoy seguro", known=1)

    assert written == []
    assert load_admission_log(str(tmp_path)).refused()


async def test_an_unrecognised_verdict_is_refused(tmp_path):
    app = _app(tmp_path)
    written = await _admit(app, tmp_path, answer=json.dumps({"verdict": "tal vez"}), known=1)

    assert written == []
    assert load_admission_log(str(tmp_path)).refused()


@pytest.mark.parametrize("said,recorded", [("invalid", "rejected"), ("redundant", "redundant")])
async def test_the_consistency_critic_can_refuse_a_well_formed_lesson(tmp_path, said, recorded):
    app = _app(tmp_path)
    written = await _admit(app, tmp_path, answer=_verdict(said), known=1)

    assert written == []
    refusals = load_admission_log(str(tmp_path)).refused()
    assert refusals
    # The reviewer's own words on the way in, the gate's words on the way out.
    assert refusals[0]["criticisms"][-1]["verdict"] == recorded


async def test_a_lesson_that_restates_what_is_known_is_refused_without_asking(tmp_path):
    # An exact restatement of a lesson already in context is knowable without
    # asking anyone, and it is a rejection: a duplicate is paid for on every
    # future call and teaches nothing. The reviewer is set to fail, so a model
    # call would have destroyed the reason.
    app = _app(tmp_path)
    written = await _admit(app, tmp_path, answer=None, duplicate=True)

    assert written == []
    refusals = load_admission_log(str(tmp_path)).refused()
    assert "already known" in refusals[0]["reason"]
    assert refusals[0]["criticisms"][-1]["verdict"] == "redundant"


async def test_a_deterministic_refusal_keeps_its_reason_and_spares_the_provider(tmp_path):
    app = _app(tmp_path)
    calls: list[str] = []

    async def spy(messages, **kwargs):
        calls.append(messages[0]["content"])
        return _verdict("valid")

    app._ask_about = spy
    store = await _store(tmp_path)
    await store.remember("procedure", "Regla previa", _hypothesis().statement, ["regla"])

    await app._store_hypothesis(store, _hypothesis())

    assert calls == [], "the model was asked a question that was already answered"
    assert [row["title"] for row in await store.recent(limit=5)] == ["Regla previa"]


# ----------------------------------------------------------------- the passages


async def test_a_lesson_with_nothing_to_contradict_is_admitted_without_a_model_call(tmp_path):
    app = _app(tmp_path)
    calls: list[str] = []

    async def spy(messages, **kwargs):
        calls.append(messages[0]["content"])
        return _verdict("valid")

    app.open_ai_client = _Stub(_verdict("valid"))
    app._ask_about = spy
    store = await _store(tmp_path)
    await app._store_hypothesis(store, _hypothesis())

    assert calls == [], "an empty context has nothing to ask about"
    assert (await store.recent(limit=5))
    assert load_admission_log(str(tmp_path)).refused() == []


async def test_three_passing_critics_admit_the_lesson(tmp_path):
    app = _app(tmp_path)
    written = await _admit(app, tmp_path, answer=_verdict("valid"), known=1)

    assert len(written) == 1
    assert load_admission_log(str(tmp_path)).refused() == []


# ----------------------------------------------------------------- the record


async def test_the_record_keeps_the_refusals_and_not_only_the_promotions(tmp_path):
    app = _app(tmp_path)
    await _admit(app, tmp_path, answer=None, known=1)
    await _admit(app, tmp_path, answer=_verdict("valid"), known=1)

    log = load_admission_log(str(tmp_path))
    assert len(log.entries) == 2
    assert len(log.refused()) == 1
    assert (Path(tmp_path) / ".minagent" / "admisiones.json").exists()


async def test_a_corrupt_record_is_read_as_empty_rather_than_crashing(tmp_path):
    (Path(tmp_path) / ".minagent").mkdir(parents=True, exist_ok=True)
    (Path(tmp_path) / ".minagent" / "admisiones.json").write_text("{not json")

    assert load_admission_log(str(tmp_path)).entries == []


async def test_the_screen_runs_before_the_write_and_needs_no_provider(tmp_path):
    """The order is the whole point: a deterministic refusal stops the write.

    The reviewer is set to fail, so a write would have happened anyway had the
    order been the other way round - the screen reads, then refuses, and the
    store is never reached.
    """
    app = _app(tmp_path)
    store = await _store(tmp_path)
    app.open_ai_client = _Stub(None)
    await store.remember(
        "procedure",
        "Regla previa",
        _hypothesis().statement,
        ["regla"],
    )

    await app._store_hypothesis(store, _hypothesis())

    titles = [row["title"] for row in await store.recent(limit=5)]
    assert titles == ["Regla previa"], "the duplicate was written anyway"


async def test_a_refused_hypothesis_never_opens_a_trial(tmp_path, monkeypatch):
    """The gate must govern the setting change, not just the memory write.

    A trial measures a setting against the session's outcomes, and the setting
    is a number the agent reasons with for the next eight turns. So a lesson the
    screen refused must not be able to move it, even though refusing the *write*
    alone looked sufficient: screening the store and not the plan left the plan
    free to act on what the store had just rejected.
    """
    app = _measuring(tmp_path, monkeypatch)
    await app.initialize_optional_features()
    store = await _store(tmp_path)
    app.memory_store = store
    await store.remember("procedure", "Regla previa", _hypothesis().statement, ["regla"])

    class _Refusing:
        async def complete(self, messages, options=None):
            asked = " ".join(str(m.get("content", "")) for m in messages)
            if "verdict" in asked:
                return {"message": {"role": "assistant", "content": _verdict("invalid")}}
            return {"message": {"role": "assistant", "content": _plan("2")}}

    app.open_ai_client = _Refusing()

    await app.reflect_on_session("test")

    assert load_trial(tmp_path) is None, "a refused hypothesis opened a trial"
    assert "COMPUTE_QUEUE_LIMIT=8" in (tmp_path / ".env").read_text()
    # And the refusal is on the record, with its reason.
    log = load_admission_log(str(tmp_path))
    assert any(not row["promote"] for row in log.entries)


async def test_an_admitted_hypothesis_still_opens_its_trial(tmp_path, monkeypatch):
    """The counter-test: the ordering change did not quietly stop everything.

    A gate that refuses everything is not a gate, it is a wall, and it would pass
    the test above perfectly. So the same path is walked with a lesson that
    passes, and the trial has to be there at the end.
    """
    app = _measuring(tmp_path, monkeypatch)
    await app.initialize_optional_features()
    app.memory_store = await _store(tmp_path)

    class _Approving:
        async def complete(self, messages, options=None):
            asked = " ".join(str(m.get("content", "")) for m in messages)
            if "verdict" in asked:
                return {"message": {"role": "assistant", "content": _verdict("valid")}}
            return {"message": {"role": "assistant", "content": _plan("2")}}

    app.open_ai_client = _Approving()

    await app.reflect_on_session("test")

    trial = load_trial(tmp_path)
    assert trial is not None, "an admitted hypothesis was not measured"
    assert trial.setting == "COMPUTE_QUEUE_LIMIT"
    assert trial.proposed == "2"


async def test_a_refused_hypothesis_is_still_shown_in_the_document(tmp_path, monkeypatch):
    """The document is a record of what the model said, not of what survived.

    A night where the model guessed twelve things and the gate kept one is the
    user's business, and the refusals carry their reasons in the log. Hiding
    them would make the gate look better than it is by making the output look
    narrower than it really was.
    """
    app = _measuring(tmp_path, monkeypatch)
    await app.initialize_optional_features()
    store = await _store(tmp_path)
    app.memory_store = store
    await store.remember("procedure", "Regla previa", _hypothesis().statement, ["regla"])

    class _Refusing:
        async def complete(self, messages, options=None):
            asked = " ".join(str(m.get("content", "")) for m in messages)
            if "verdict" in asked:
                return {"message": {"role": "assistant", "content": _verdict("invalid")}}
            return {"message": {"role": "assistant", "content": _plan("2")}}

    app.open_ai_client = _Refusing()

    report = await app.reflect_on_session("test")

    assert "La cola se atasca" in (tmp_path / "MEJORAS.md").read_text()
    assert "hipótesis" in report


def _measuring(tmp_path, monkeypatch) -> MinAgent:
    """An app that may move a setting, with one pinned in the environment."""
    app = _app(tmp_path)
    app.improvement_enabled = True
    app.improvement_auto = True
    app.application_root = str(tmp_path)
    monkeypatch.setenv("COMPUTE_QUEUE_LIMIT", "8")
    monkeypatch.setenv("MEMORY_REFLECTION_INTERVAL", "10")
    (tmp_path / ".env").write_text("COMPUTE_QUEUE_LIMIT=8\nMEMORY_REFLECTION_INTERVAL=10\n")
    return app


def _plan(value: str) -> str:
    return json.dumps(
        {
            "hypotheses": [
                {
                    "title": "La cola se atasca",
                    "kind": "improvement",
                    "target": "agent",
                    "statement": _hypothesis().statement,
                    "evidence": _hypothesis().evidence,
                    "expected": _hypothesis().expected,
                    "verify": _hypothesis().verify,
                    "setting": "COMPUTE_QUEUE_LIMIT",
                    "value": value,
                    "reason": "la cola se atasca",
                }
            ]
        }
    )


async def test_an_empty_context_admits_because_there_is_nothing_to_contradict(tmp_path):
    """Fail-closed has a floor: refusing everything is not a gate, it is a wall.

    With no prior lesson in context the consistency question has no answer to
    disagree with, so the deterministic screen is sufficient and the provider
    is not consulted. The record still shows the reasoning.
    """
    app = _app(tmp_path)
    calls: list[str] = []
    store = await _store(tmp_path)

    async def spy(messages, **kwargs):
        calls.append(messages[0]["content"])
        return _verdict("valid")

    app._ask_about = spy
    await app._store_hypothesis(store, _hypothesis())

    assert calls == []
    assert [row["title"] for row in await store.recent(limit=5)]
    assert "nothing in context" in json.dumps(load_admission_log(str(tmp_path)).entries)
