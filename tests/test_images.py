"""Tests for images that are noticed but not sent.

The property under test is that a path costs a handful of tokens and the pixels
cost a couple of thousand, so the pixels only exist when the model asked for
them, and stop existing when the context gets tight. The old behaviour is still
reachable, and that has to keep working: ``eager`` mode is the escape hatch.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from minagent.app import MinAgent
from minagent.attachments import DEFERRED_NOTICE, prepare_user_message
from minagent.capabilities import build_builtin_capabilities
from minagent.config import parse_on_demand_images
from minagent.context_budget import SHED_ATTACHED_IMAGES, SHED_STEPS, ContextPolicy
from minagent.errors import AgentError
from minagent.images import run_view_image
from minagent.workspace import WorkspaceAccess


class _FakeOutput:
    """A stdout stand-in that records what was written."""

    def __init__(self, columns: int = 100) -> None:
        self.columns = columns
        self.chunks: list[str] = []

    def write(self, value: str) -> None:
        self.chunks.append(value)

    def isatty(self) -> bool:
        return False

    @property
    def text(self) -> str:
        return "".join(self.chunks)


def _app(tmp_path, **settings: Any) -> MinAgent:
    """An app with the features a test asks for, and nothing else."""
    app = MinAgent(stdout=_FakeOutput())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "Test"
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    for name, value in settings.items():
        setattr(app, name, value)
    app.rebuild_capabilities()
    return app


def _sent_tools(app: MinAgent) -> list[str]:
    return [tool["function"]["name"] for tool in app.tools]


def _png(width: int = 8, height: int = 8) -> bytes:
    """A real PNG, small enough to inline and valid enough to sniff."""
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _workspace_with_image(directory: Path, name: str = "salida/captura.png") -> tuple[WorkspaceAccess, str]:
    target = directory / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_png())
    return WorkspaceAccess(str(directory), "Test", 0), name


def _text_of(message: dict[str, Any]) -> str:
    content = message["content"]
    return content if isinstance(content, str) else content[0]["text"]


def _parts_of(message: dict[str, Any]) -> list[str]:
    content = message["content"]
    return [part["type"] for part in content] if not isinstance(content, str) else []


# ------------------------------------------------------- noticing, not sending


@pytest.mark.asyncio
async def test_on_demand_leaves_the_pixels_out_and_keeps_the_path(tmp_path):
    access, path = _workspace_with_image(tmp_path)
    prepared = await prepare_user_message(
        f"que ves en {path}", [], access, ["text", "image"], on_demand_images=True
    )
    message = prepared["message"]
    assert _parts_of(message) == []
    # The path has to survive: it is the only handle the model has.
    assert path in _text_of(message)
    assert DEFERRED_NOTICE.split("{")[0].strip() in _text_of(message)
    assert prepared["deferred_images"] == [{"path": path, "mime_type": "image/png"}]


@pytest.mark.asyncio
async def test_eager_mode_still_attaches_the_pixels(tmp_path):
    access, path = _workspace_with_image(tmp_path)
    prepared = await prepare_user_message(
        f"que ves en {path}", [], access, ["text", "image"], on_demand_images=False
    )
    assert _parts_of(prepared["message"]) == ["text", "image_url"]
    assert prepared["deferred_images"] == []
    assert path not in _text_of(prepared["message"])


@pytest.mark.asyncio
async def test_a_model_without_image_input_still_reports_the_problem(tmp_path):
    access, path = _workspace_with_image(tmp_path)
    prepared = await prepare_user_message(f"mira {path}", [], access, ["text"], on_demand_images=True)
    assert prepared["deferred_images"] == []
    assert any(event["kind"] == "error" for event in prepared["events"])


@pytest.mark.asyncio
async def test_the_same_image_is_only_mentioned_once(tmp_path):
    access, path = _workspace_with_image(tmp_path)
    prepared = await prepare_user_message(
        f"{path} y otra vez {path}", [], access, ["text", "image"], on_demand_images=True
    )
    assert len(prepared["deferred_images"]) == 1


# --------------------------------------------------------------- the tool side


@pytest.mark.asyncio
async def test_view_image_hands_back_pixels_for_the_next_request(tmp_path):
    access, path = _workspace_with_image(tmp_path)
    result = await run_view_image(path, access)
    assert result["image"]["mime_type"] == "image/png"
    assert result["image"]["data"]
    assert path in result["tool_text"]


@pytest.mark.asyncio
async def test_view_image_refuses_a_file_that_is_not_an_image(tmp_path):
    access = WorkspaceAccess(str(tmp_path), "Test", 0)
    (tmp_path / "notas.txt").write_text("esto no es una imagen")
    with pytest.raises(AgentError):
        await run_view_image("notas.txt", access)


@pytest.mark.asyncio
async def test_view_image_needs_a_path(tmp_path):
    access = WorkspaceAccess(str(tmp_path), "Test", 0)
    with pytest.raises(AgentError):
        await run_view_image("   ", access)


# ------------------------------------------------------------ the capability


def test_the_images_capability_is_not_eager():
    """The whole point: no image schema in a request that has not asked for one."""
    entries = build_builtin_capabilities(terminal_mode="off", images_enabled=True)
    images = {entry.name: entry for entry in entries}["images"]
    assert images.eager is False
    assert images.tool_names == ("view_image",)


def test_no_images_capability_for_a_model_that_cannot_see():
    entries = build_builtin_capabilities(terminal_mode="off", images_enabled=False)
    assert "images" not in {entry.name for entry in entries}


def test_view_image_stays_out_of_the_request_until_the_capability_loads(tmp_path):
    app = _app(tmp_path, input_modalities=["text", "image"], on_demand_images=True)
    app.ensure_image_tools()
    assert "view_image" not in _sent_tools(app)
    app.load_capabilities("images")
    assert "view_image" in _sent_tools(app)


def test_the_main_model_does_the_looking_with_vision_off(tmp_path):
    """No second model, but the tool is still there: view_image needs no Ollama."""
    app = _app(tmp_path, input_modalities=["text", "image"], on_demand_images=True, vision_enabled=False)
    app.ensure_image_tools()
    assert app.vision_client is None
    app.load_capabilities("images")
    assert "view_image" in _sent_tools(app)
    assert "describe_image" not in app._tool_schemas


def test_no_image_tools_for_a_model_that_cannot_see(tmp_path):
    app = _app(tmp_path, input_modalities=["text"], on_demand_images=True, vision_enabled=True)
    app.ensure_image_tools()
    assert "view_image" not in app._tool_schemas
    assert "describe_image" not in app._tool_schemas


# ------------------------------------------------------------- letting go


def _with_image(message: dict[str, Any], path: str) -> None:
    message["content"] = [
        {"type": "text", "text": f"Image(s) now visible, loaded by a tool: {path}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png().hex()}"}},
    ]


def test_releasing_the_pixels_keeps_the_path_and_frees_the_room(tmp_path):
    app = _app(tmp_path, input_modalities=["text", "image"], on_demand_images=True)
    app.ensure_image_tools()
    app.messages = [{"role": "user", "content": "hola"}]
    _with_image(app.messages[0], "salida/captura.png")
    app._turn_first_message_index = 1  # the image belongs to a finished turn

    released = app.release_attached_images()

    assert released > 0
    parts = app.messages[0]["content"]
    assert [part["type"] for part in parts] == ["text", "text"]
    assert "image_url" not in app.messages[0]["content"][1]["text"]
    # The text part is what tells the model the picture is still there.
    assert "salida/captura.png" in parts[0]["text"]


def test_the_turn_in_flight_keeps_its_image(tmp_path):
    """A task in progress must not have the thing it is reading vanish under it."""
    app = _app(tmp_path, input_modalities=["text", "image"], on_demand_images=True)
    app.messages = [{"role": "user", "content": "turno viejo"}, {"role": "assistant", "content": "ok"}]
    _with_image(app.messages[1], "salida/actual.png")
    app._turn_first_message_index = 1

    assert app.release_attached_images() == 0
    assert app.messages[1]["content"][1]["type"] == "image_url"


def test_releasing_twice_gives_nothing_the_second_time(tmp_path):
    app = _app(tmp_path, input_modalities=["text", "image"], on_demand_images=True)
    old_turn = {"role": "user", "content": "turno viejo"}
    _with_image(old_turn, "salida/a.png")
    app.messages = [old_turn]
    app._turn_first_message_index = 1
    assert app.release_attached_images() > 0
    assert app.release_attached_images() == 0


def test_images_are_shed_after_tool_results_and_before_memory():
    assert SHED_STEPS.index(SHED_ATTACHED_IMAGES) == SHED_STEPS.index("old tool results") + 1


def test_a_full_window_reaches_the_image_step(tmp_path):
    """The shed step has to be reachable from the cascade, not just defined."""
    app = _app(tmp_path, input_modalities=["text", "image"], on_demand_images=True)
    old_turn = {"role": "user", "content": "turno viejo"}
    _with_image(old_turn, "salida/captura.png")
    app.messages = [old_turn]
    app._turn_first_message_index = 1
    # A tiny window with a high watermark: the cascade is over the mark from the
    # first turn, which is the situation the step exists for.
    app.context_policy = ContextPolicy(high_watermark=0.01, low_watermark=0.005)
    app.effective_context_window = lambda: 1  # type: ignore[method-assign]

    applied = app.regulate_context()

    assert SHED_ATTACHED_IMAGES in applied
    # Later steps may compact the message further; all that matters here is that
    # no encoding is left in it.
    assert "image_url" not in str(old_turn["content"])


# ------------------------------------------------------- the whole way round


@pytest.mark.asyncio
async def test_a_path_costs_nothing_until_the_model_asks_for_it(tmp_path):
    """The property, end to end: notice, ask, see, release."""
    access, path = _workspace_with_image(tmp_path)
    app = _app(tmp_path, input_modalities=["text", "image"], on_demand_images=True)
    app.ensure_image_tools()

    message = await app.prepare_user_message(f"que se ve en {path}?", [])
    app.messages.append(message)
    app._turn_first_message_index = len(app.messages) - 1
    assert _parts_of(message) == []
    assert path in _text_of(message)

    # The model loads the capability and calls the tool.
    app.load_capabilities("images")
    result = await app.run_view_image({"path": path})
    assert result["image"]["data"]

    # The turn loop does what it already did for any tool with an image.
    app.messages.append({"role": "tool", "tool_call_id": "1", "content": result["tool_text"]})
    app.messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Image(s) now visible, loaded by a tool: {path}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{result['image']['data']}"},
                },
            ],
        }
    )

    # Next turn: the picture is released and the path is still there to reload.
    app._turn_first_message_index = len(app.messages)
    assert app.release_attached_images() > 0
    released = app.messages[-1]["content"][-1]
    assert released["type"] == "text"
    assert path in app.messages[-1]["content"][0]["text"]


# ------------------------------------------------------------------- settings


def test_images_are_on_demand_unless_something_else_is_asked_for():
    assert parse_on_demand_images(None) is True
    assert parse_on_demand_images("on_demand") is True
    assert parse_on_demand_images("  ON-DEMAND ") is True
    assert parse_on_demand_images("eager") is False


def test_a_bad_image_mode_is_rejected_rather_than_guessed():
    with pytest.raises(AgentError):
        parse_on_demand_images("maybe")
