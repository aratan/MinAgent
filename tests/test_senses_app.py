"""App-level tests for the two behaviours that involve the rest of the session.

The microphone returning words rather than a path, and a durable module being
refused without an answer, are both decisions a model can act on wrongly. They
are asserted here rather than in the module tests because the behaviour lives
in the app: the module records a file, and the app decides what to do with it.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from minagent.app import MODULE_APPROVAL_PREVIEW_CHARS, MinAgent  # noqa: E402
from minagent.errors import AgentError  # noqa: E402

MODULE_SOURCE = '''from typing import Any


def create_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "saludar",
                "description": "Saluda.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
'''


class _Out:
    def write(self, value: str) -> None:
        print(value, end="")

    def isatty(self) -> bool:
        return False


class _Recorder:
    """A fake editor that records the prompt and answers a fixed way."""

    def __init__(self, answer: str = "y") -> None:
        self.answer = answer
        self.asked: list[str] = []

    async def question(self, prompt: str) -> str:
        self.asked.append(prompt)
        return self.answer


def _app(tmp_path, **flags) -> MinAgent:
    app = MinAgent(stdout=_Out())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "prueba"
    app.tools = []
    app._tool_schemas = {}
    for name, value in flags.items():
        setattr(app, name, value)
    return app


# -- the microphone returning words --------------------------------------


class _Senses:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def record_audio(self, seconds: int, stem: str) -> Path:
        return self.path


class _Orchestrator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    async def transcribe(self, path: str, language: str = "") -> str:
        self.calls.append((path, language))
        return self.text


async def test_the_microphone_returns_the_words_not_just_the_path(tmp_path):
    """The recording was made to be read; a path alone defers the words by a turn."""
    wav = tmp_path / "micro.wav"
    wav.write_bytes(b"x")
    app = _app(tmp_path, senses_enabled=True, compute_enabled=True, orchestrator=_Orchestrator("hola que tal"))
    app.senses_client = _Senses(wav)

    result = await app.run_record_microphone({"seconds": 3})
    assert "hola que tal" in result
    assert "micro.wav" in result, "the audio path must still be named, so the user can check it"


async def test_the_language_hint_reaches_the_engine(tmp_path):
    wav = tmp_path / "micro.wav"
    wav.write_bytes(b"x")
    engine = _Orchestrator("hola")
    app = _app(tmp_path, senses_enabled=True, compute_enabled=True, orchestrator=engine)
    app.senses_client = _Senses(wav)

    await app.run_record_microphone({"seconds": 3, "language": "es"})
    assert engine.calls == [(str(wav), "es")]


async def test_silence_is_reported_as_silence_rather_than_as_a_bug(tmp_path):
    wav = tmp_path / "micro.wav"
    wav.write_bytes(b"x")
    app = _app(tmp_path, senses_enabled=True, compute_enabled=True, orchestrator=_Orchestrator("   "))
    app.senses_client = _Senses(wav)

    result = await app.run_record_microphone({"seconds": 3})
    assert "no words" in result
    assert "micro.wav" in result


async def test_with_the_speech_engine_off_it_says_so_and_keeps_the_audio(tmp_path):
    """Silently returning a path would read as a successful transcription."""
    wav = tmp_path / "micro.wav"
    wav.write_bytes(b"x")
    app = _app(tmp_path, senses_enabled=True, compute_enabled=False, orchestrator=None)
    app.senses_client = _Senses(wav)

    result = await app.run_record_microphone({"seconds": 3})
    assert "COMPUTE_ENABLED=on" in result
    assert "micro.wav" in result


async def test_transcribe_false_returns_only_the_path(tmp_path):
    wav = tmp_path / "micro.wav"
    wav.write_bytes(b"x")
    engine = _Orchestrator("hola")
    app = _app(tmp_path, senses_enabled=True, compute_enabled=True, orchestrator=engine)
    app.senses_client = _Senses(wav)

    result = await app.run_record_microphone({"seconds": 3, "transcribe": False})
    assert "transcribe_audio" in result
    assert engine.calls == []


# -- durable modules needing approval ------------------------------------


async def test_a_durable_module_asks_first_and_writes_after_yes(tmp_path):
    from minagent.subagents import SubagentStore

    app = _app(tmp_path, subagents_enabled=True, editor=_Recorder("y"))
    calls: list[list[str]] = []
    app.subagent_store = SubagentStore(str(tmp_path / "sub"), runner=calls.append)

    result = await app.run_write_module({"name": "saludo", "source": MODULE_SOURCE})
    assert app.editor.asked, "a durable module must not be written without asking"
    assert "branch agent/module-saludo" in result
    assert calls and calls[0][:2] == ["git", "checkout"]


async def test_a_declined_module_writes_nothing_and_leaves_no_branch(tmp_path):
    from minagent.subagents import SubagentStore

    app = _app(tmp_path, subagents_enabled=True, editor=_Recorder("n"))
    calls: list[list[str]] = []
    app.subagent_store = SubagentStore(str(tmp_path / "sub"), runner=calls.append)

    result = await app.run_write_module({"name": "saludo", "source": MODULE_SOURCE})
    assert "denied" in result
    assert calls == [], "no git command may run when the user says no"
    assert not (tmp_path / "sub" / "saludo.py").exists()
    assert "saludo" not in await app.run_list_modules({})


async def test_ephemeral_modules_need_no_approval(tmp_path):
    """It never reaches the repository, so interrupting the user buys nothing."""
    from minagent.subagents import SubagentStore

    app = _app(tmp_path, subagents_enabled=True, editor=_Recorder("n"))
    calls: list[list[str]] = []
    app.subagent_store = SubagentStore(str(tmp_path / "sub"), runner=calls.append)

    result = await app.run_write_module(
        {"name": "saludo", "source": MODULE_SOURCE, "ephemeral": True}
    )
    assert app.editor.asked == []
    assert calls == []
    assert "memory only" in result


async def test_without_a_terminal_no_durable_module_is_written(tmp_path):
    """Approving unseen code is not approval, so the safe reading is to refuse."""
    from minagent.subagents import SubagentStore

    app = _app(tmp_path, subagents_enabled=True, editor=None)
    app.subagent_store = SubagentStore(str(tmp_path / "sub"), runner=lambda c: None)

    with pytest.raises(AgentError, match="ephemeral=true"):
        await app.run_write_module({"name": "saludo", "source": MODULE_SOURCE})


async def test_an_oversized_module_is_refused_rather_than_shown_partially(tmp_path):
    """Approving the first 8000 characters of code that then runs is a guess."""
    from minagent.subagents import SubagentStore

    app = _app(tmp_path, subagents_enabled=True, editor=_Recorder("y"))
    app.subagent_store = SubagentStore(str(tmp_path / "sub"), runner=lambda c: None)
    source = MODULE_SOURCE + "# " + "x" * (MODULE_APPROVAL_PREVIEW_CHARS + 10)

    with pytest.raises(AgentError, match="approval preview"):
        await app.run_write_module({"name": "saludo", "source": source})
    assert app.editor.asked == []
