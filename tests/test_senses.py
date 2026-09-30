"""Senses tests: the ffmpeg command lines, and what is refused.

The command lines are asserted exactly, because ffmpeg is unforgiving about
its device and format arguments and a wrong one fails only at the moment the
camera is actually used.
"""

import asyncio
from pathlib import Path

import pytest

from minagent.errors import AgentError
from minagent.senses import (
    MAX_MICROPHONE_SECONDS,
    SensesClient,
    _safe_stem,
    create_senses_tools,
)


@pytest.fixture
def calls() -> list[list[str]]:
    return []


def _fake_ffmpeg(calls: list[list[str]]):
    """A runner that records the call and writes the file ffmpeg would have.

    Without the file the client would rightly report that nothing was
    captured, so the double has to produce the artefact the real one produces.
    """

    def run(arguments: list[str]) -> str:
        calls.append(arguments)
        Path(arguments[-1]).write_bytes(b"not-really-an-image")
        return ""

    return run


@pytest.fixture
def client(calls: list[list[str]], tmp_path) -> SensesClient:
    calls.clear()
    return SensesClient(
        camera_device="/dev/video0",
        output_directory=str(tmp_path / "salida"),
        runner=_fake_ffmpeg(calls),
    )


async def test_a_frame_is_one_still_not_a_burst(client, calls, tmp_path):
    path, note = await client.capture_frame("documento")
    assert note == ""
    arguments = calls[0]
    assert "-f" in arguments and arguments[arguments.index("-f") + 1] == "v4l2"
    assert "-i" in arguments and arguments[arguments.index("-i") + 1] == "/dev/video0"
    # One frame, and a settling delay: the first frame off a sensor is the one
    # it was exposed for before auto-exposure had a chance to act.
    assert arguments[arguments.index("-frames:v") + 1] == "1"
    assert "-ss" in arguments
    assert (tmp_path / "salida" / "documento.jpg").exists()
    assert path.name == "documento.jpg"


def test_a_black_frame_is_reported_as_unreadable_rather_than_handed_over(tmp_path):
    """The camera here is in a dark room: it returns a valid JPEG carrying nothing.
    A model handed that describes blackness, so the caveat has to be explicit."""
    from PIL import Image

    path = tmp_path / "negra.jpg"
    Image.new("L", (32, 32), color=6).save(path)
    assert SensesClient(output_directory=str(tmp_path))._too_dark(path) is True


def test_a_well_lit_frame_is_not_flagged(tmp_path):
    from PIL import Image

    path = tmp_path / "clara.jpg"
    Image.new("L", (32, 32), color=180).save(path)
    assert SensesClient(output_directory=str(tmp_path))._too_dark(path) is False


async def test_a_missing_camera_is_reported_before_ffmpeg_runs(calls):
    absent = SensesClient(
        camera_device="/dev/video99", output_directory="/tmp/senses-missing", runner=_fake_ffmpeg(calls)
    )
    with pytest.raises(AgentError, match="No camera at /dev/video99"):
        await absent.capture_frame()
    assert calls == []


async def test_a_camera_that_holds_nothing_is_reported_as_a_busy_device(calls, tmp_path):
    """ffmpeg exiting cleanly without writing a file is what a device held by a
    video call looks like, and a blank path would be worse than the message."""
    def writes_nothing(arguments: list[str]) -> str:
        calls.append(arguments)
        return ""

    busy = SensesClient(
        camera_device="/dev/video0", output_directory=str(tmp_path / "salida"), runner=writes_nothing
    )
    with pytest.raises(AgentError, match="produced no image"):
        await busy.capture_frame()
    with pytest.raises(AgentError, match="produced no audio"):
        await busy.record_audio(3)


async def test_audio_is_recorded_through_pipewire_not_alsa(client, calls):
    """Naming a raw ALSA device would ignore the user's mixer and often record
    the wrong microphone, of which this machine has three."""
    await client.record_audio(5, "nota")
    arguments = calls[0]
    assert arguments[arguments.index("-f") + 1] == "pulse"
    assert arguments[arguments.index("-i") + 1] == "default"
    assert arguments[arguments.index("-t") + 1] == "5"


async def test_a_long_recording_is_refused_rather_than_trimmed(client, calls):
    with pytest.raises(AgentError, match="at most 120"):
        await client.record_audio(MAX_MICROPHONE_SECONDS + 1)
    assert calls == []


async def test_a_zero_length_recording_is_refused(client, calls):
    with pytest.raises(AgentError, match="at least 1"):
        await client.record_audio(0)
    assert calls == []


def test_a_model_supplied_label_cannot_escape_the_output_directory():
    assert _safe_stem("../../etc/passwd") == "etc-passwd"
    # Path separators and shell punctuation collapse to single dashes.
    assert "/" not in _safe_stem("mi captura; rm -rf /")
    assert ".." not in _safe_stem("../../etc/passwd")
    assert _safe_stem("") == "captura"


def test_the_tools_describe_capture_as_on_demand():
    schemas = {tool["function"]["name"]: tool["function"]["description"] for tool in create_senses_tools()}
    assert set(schemas) == {"capture_camera", "record_microphone"}
    # The guarantee that matters is in the schema the model reads, not only in
    # the module docstring it never sees.
    for description in schemas.values():
        assert "when called" in description


def test_a_missing_ffmpeg_is_named_with_the_install_command(client):
    client._binary = None
    with pytest.raises(AgentError, match="sudo pacman -S ffmpeg"):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(client.capture_frame())
