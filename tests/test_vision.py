"""Vision tests: the Ollama chat client, image encoding, config, and app dispatch."""

from __future__ import annotations

import base64
from typing import Any

import httpx
import pytest

from minagent.app import MinAgent
from minagent.config import load_configuration
from minagent.errors import AgentError
from minagent.vision import (
    VisionClient,
    create_vision_tools,
    encode_image,
    format_image_result,
)
from minagent.workspace import WorkspaceAccess

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class _FakeOutput:
    def write(self, value: str) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _client(payload: Any, status: int = 200, captured: list[httpx.Request] | None = None) -> VisionClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured.append(request)
        return httpx.Response(status, json=payload)

    return VisionClient("http://localhost:11434", "test-model", 30, transport=httpx.MockTransport(handler))


def _config(tmp_path: Any, env: dict[str, str]) -> Any:
    """A configuration with the keys load_configuration insists on."""
    base = {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "test-model"}
    return load_configuration(str(tmp_path), str(tmp_path), {**base, **env})


def _agent(tmp_path: Any, env: dict[str, str]) -> MinAgent:
    agent = MinAgent(stdout=_FakeOutput())
    agent.root_directory = str(tmp_path)
    agent.application_root = str(tmp_path)
    agent.workspace_name = "Test"
    agent.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    agent.vision_enabled = _config(tmp_path, env).vision_enabled
    if agent.vision_enabled:
        agent.vision_client = VisionClient("http://localhost:11434", "test-model", 30)
        agent.ensure_vision_tools()
    return agent


# ----------------------------------------------------------------- encoding


def test_encode_image_returns_base64_and_media_type(tmp_path):
    path = tmp_path / "foto.png"
    path.write_bytes(PNG_BYTES)
    payload, media_type = encode_image(str(path))
    assert media_type == "image/png"
    assert base64.b64decode(payload) == PNG_BYTES


@pytest.mark.parametrize(
    ("name", "expected"),
    [("a.jpg", "image/jpeg"), ("a.JPEG", "image/jpeg"), ("a.webp", "image/webp")],
)
def test_encode_image_detects_type_by_suffix(tmp_path, name, expected):
    path = tmp_path / name
    path.write_bytes(PNG_BYTES)
    _, media_type = encode_image(str(path))
    assert media_type == expected


def test_encode_image_rejects_a_non_image(tmp_path):
    path = tmp_path / "notas.txt"
    path.write_text("hola", encoding="utf-8")
    with pytest.raises(AgentError, match="Not an image format"):
        encode_image(str(path))


def test_encode_image_rejects_an_empty_file(tmp_path):
    path = tmp_path / "vacia.png"
    path.write_bytes(b"")
    with pytest.raises(AgentError, match="empty"):
        encode_image(str(path))


def test_encode_image_reports_a_missing_file(tmp_path):
    with pytest.raises(AgentError, match="Cannot read image"):
        encode_image(str(tmp_path / "no-existe.png"))


# ----------------------------------------------------------------- the call


async def test_describe_sends_the_image_inline_and_returns_the_text(tmp_path):
    path = tmp_path / "captura.png"
    path.write_bytes(PNG_BYTES)
    captured: list[httpx.Request] = []
    client = _client({"message": {"role": "assistant", "content": "  Un gato.  "}}, captured=captured)

    answer = await client.describe(str(path), "  ¿Qué   ves? ")

    assert answer == "Un gato."
    request = captured[0]
    assert request.url.path == "/api/chat"
    body = request.read().decode()
    assert base64.b64encode(PNG_BYTES).decode() in body
    # The question is normalised: runs of whitespace collapse to one space.
    assert '"content":"\\u00bfQu\\u00e9 ves?"' in body or "¿Qué ves?" in body


async def test_describe_works_with_the_flat_response_field(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(PNG_BYTES)
    client = _client({"response": "plano"})
    assert await client.describe(str(path), "q") == "plano"


async def test_describe_requires_a_question(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(PNG_BYTES)
    client = _client({"message": {"content": "x"}})
    with pytest.raises(AgentError, match="non-empty question"):
        await client.describe(str(path), "   ")


async def test_describe_rejects_an_empty_answer_with_a_useful_hint(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(PNG_BYTES)
    client = _client({"message": {"content": "  "}})
    with pytest.raises(AgentError, match="vision"):
        await client.describe(str(path), "q")


async def test_missing_model_names_the_pull_command(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(PNG_BYTES)
    client = _client({"error": "not found"}, status=404)
    with pytest.raises(AgentError, match="ollama pull"):
        await client.describe(str(path), "q")


async def test_a_rejected_request_points_at_the_vision_capability(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(PNG_BYTES)
    client = _client({"error": "bad"}, status=400)
    with pytest.raises(AgentError, match="vision"):
        await client.describe(str(path), "q")


async def test_an_unreachable_server_says_how_to_start_ollama(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(PNG_BYTES)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = VisionClient(
        "http://localhost:11434", "m", 5, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(AgentError, match="ollama serve"):
        await client.describe(str(path), "q")


# ---------------------------------------------------------------- rendering


def test_the_result_warns_the_model_it_is_a_reading():
    text = format_image_result("salida/a.png", "qwen3.5", "Un gato")
    assert "salida/a.png" in text
    assert "qwen3.5" in text
    # A model that reads a plate or a face out of a picture will invent one,
    # so the payload has to say its output is not verified.
    assert "not" in text and "verified" in text
    assert text.endswith("Un gato")


def test_the_tool_schema_requires_a_path_and_a_question():
    tool = create_vision_tools()[0]
    assert tool["function"]["name"] == "describe_image"
    assert set(tool["function"]["parameters"]["required"]) == {"path", "question"}


# -------------------------------------------------------------------- config


def test_vision_is_off_by_default(tmp_path):
    config = _config(tmp_path, {})
    assert config.vision_enabled is False
    assert config.vision_model == "qwen3.5:9b-q4_K_M"
    assert config.vision_base_url == "http://localhost:11434"


def test_vision_settings_come_from_the_environment(tmp_path):
    config = _config(
        tmp_path,
        {
            "VISION_ENABLED": "on",
            "VISION_MODEL": "qwen2.5vl:3b",
            "VISION_BASE_URL": "http://gpu-box:11434/",
            "VISION_TIMEOUT_SECONDS": "900",
        },
    )
    assert config.vision_enabled is True
    assert config.vision_model == "qwen2.5vl:3b"
    assert config.vision_base_url == "http://gpu-box:11434"
    assert config.vision_timeout_seconds == 900


# ----------------------------------------------------------------- dispatch


async def test_the_app_resolves_a_workspace_relative_image_path(tmp_path):
    (tmp_path / "salida").mkdir()
    (tmp_path / "salida" / "a.png").write_bytes(PNG_BYTES)
    agent = _agent(tmp_path, {"VISION_ENABLED": "on"})

    seen: dict[str, str] = {}

    async def fake_describe(image_path: str, question: str) -> str:
        seen["path"] = image_path
        seen["question"] = question
        return "Un gato"

    agent.vision_client.describe = fake_describe  # type: ignore[method-assign]
    result = await agent.run_describe_image({"path": "salida/a.png", "question": "¿Qué ves?"})

    assert seen["path"] == str(tmp_path / "salida" / "a.png")
    assert seen["question"] == "¿Qué ves?"
    assert "Un gato" in result


async def test_the_app_refuses_a_path_outside_the_workspace(tmp_path):
    agent = _agent(tmp_path, {"VISION_ENABLED": "on"})
    with pytest.raises(AgentError, match="outside the current workspace"):
        await agent.run_describe_image({"path": "/etc/passwd", "question": "q"})


async def test_the_app_reports_a_missing_file_before_calling_the_model(tmp_path):
    agent = _agent(tmp_path, {"VISION_ENABLED": "on"})

    async def should_not_run(image_path: str, question: str) -> str:  # pragma: no cover
        raise AssertionError("the model must not be called for a missing file")

    agent.vision_client.describe = should_not_run  # type: ignore[method-assign]
    with pytest.raises(AgentError, match="No image file"):
        await agent.run_describe_image({"path": "salida/nada.png", "question": "q"})


async def test_the_app_says_when_vision_is_disabled(tmp_path):
    agent = _agent(tmp_path, {})
    with pytest.raises(AgentError, match="not enabled"):
        await agent.run_describe_image({"path": "a.png", "question": "q"})


def test_the_tool_is_registered_only_when_enabled(tmp_path):
    off = _agent(tmp_path, {})
    assert "describe_image" not in off._tool_schemas

    on = _agent(tmp_path, {"VISION_ENABLED": "on"})
    # Registered in the catalogue, but the vision capability is on demand, so
    # it is not published until something pulls it in.
    assert "describe_image" in on._tool_schemas
    index = on.capability_index_section()
    assert index is not None
    assert "describe_image" in index["content"]
    assert "vision" in index["content"]


def test_the_note_offers_vision_proactively(tmp_path):
    # The reminder is what the model gets when it starts to answer a picture
    # question without calling the tool, so that is where it has to be.
    agent = _agent(tmp_path, {"VISION_ENABLED": "on"})
    note = agent.missing_capability_note()
    assert "describe_image" in note
    assert "cannot see images" in note

    without = _agent(tmp_path, {})
    assert "describe_image" not in without.missing_capability_note()


def test_the_capability_is_offered_and_not_eager(tmp_path):
    from minagent.capabilities import build_builtin_capabilities

    entries = build_builtin_capabilities(terminal_mode="off", vision_enabled=True)
    vision = next(entry for entry in entries if entry.name == "vision")
    assert vision.tool_names == ("describe_image",)
    # Loading it costs a request, so it must not be paid for on every turn.
    assert vision.eager is False

    without = build_builtin_capabilities(terminal_mode="off", vision_enabled=False)
    assert not any(entry.name == "vision" for entry in without)
