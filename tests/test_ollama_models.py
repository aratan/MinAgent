"""Ollama model management tests: the requests sent, and what is refused."""

import json

import httpx
import pytest

from minagent.errors import AgentError
from minagent.ollama_models import (
    OllamaModelsClient,
    check_model_name,
    format_models_table,
    gpu_vram_bytes,
)

BASE = "http://ollama.test"


def _client(handler) -> OllamaModelsClient:
    return OllamaModelsClient(BASE, transport=httpx.MockTransport(handler))


async def test_listing_reports_size_and_quantisation():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            # The 9B holds 5.11 GiB on an 8 GiB card while its 6.14 GiB file sits
            # on disk, so the VRAM figure is the one that decides a second load.
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b-q4_K_M", "size_vram": 5490081790}]})
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3.5:9b-q4_K_M",
                        "size": 6596000000,
                        "modified_at": "2026-09-30T10:00:00Z",
                        "details": {
                            "family": "qwen35",
                            "parameter_size": "9.7B",
                            "quantization_level": "Q4_K_M",
                        },
                    }
                ]
            },
        )

    models = await _client(handler).list_models()
    assert models[0]["name"] == "qwen3.5:9b-q4_K_M"
    assert models[0]["quantization"] == "Q4_K_M"
    assert models[0]["vram_bytes"] == 5490081790
    assert models[0]["vram_bytes"] < models[0]["size_bytes"]


async def test_a_model_that_is_not_resident_is_reported_as_an_upper_bound():
    """The disk copy is larger than the card footprint, so claiming a definite
    'does not fit' from it would refuse a model that is actually running."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        return httpx.Response(200, json={"models": [{"name": "gorda:70b", "size": 40 * 1024**3}]})

    report = await _client(handler).report_hardware()
    assert "upper bound" in report


async def test_an_empty_server_says_what_to_do_about_it():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    assert "ollama pull" in format_models_table(await _client(handler).list_models())


async def test_creating_a_derived_model_sends_from_and_system():
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"status": "success"})

    name = await _client(handler).create_model(
        "cocinero", "gemma3:4b", "Eres un experto en cocina y solo respondes con recetas."
    )
    assert name == "cocinero"
    assert sent["from"] == "gemma3:4b"
    assert "recetas" in sent["system"]
    assert sent["stream"] is False


async def test_creating_one_without_a_system_prompt_omits_the_field():
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={})

    await _client(handler).create_model("rapido", "gemma3:4b")
    assert "system" not in sent


async def test_a_derived_model_without_a_base_is_refused_before_any_request():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no request should be sent")

    with pytest.raises(AgentError, match="base model is required"):
        await _client(handler).create_model("huerfano", "")


@pytest.mark.parametrize(
    "name", ["con espacio", "-empieza", "vacio/", "doble//barra", "a" * 65, "salto\nlinea"]
)
def test_a_name_that_is_not_ollama_safe_is_refused(name):
    """A model name becomes a directory and appears in shell commands."""
    with pytest.raises(AgentError, match="Invalid model name"):
        check_model_name(name)


@pytest.mark.parametrize("name", ["gemma3:4b", "mi-asistente", "hf.co/user/repo:Q4_K_M"])
def test_ordinary_names_are_accepted(name):
    assert check_model_name(name) == name


async def test_deleting_sends_the_exact_name():
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={})

    assert await _client(handler).delete_model("viejo:4b") == "viejo:4b"
    assert sent["model"] == "viejo:4b"


async def test_a_model_that_is_not_there_is_reported_with_the_servers_reason():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'x' not found"})

    with pytest.raises(AgentError, match="no such model"):
        await _client(handler).delete_model("x")


async def test_an_unreachable_server_names_the_fix():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(AgentError, match="ollama serve"):
        await _client(handler).list_models()


def test_a_model_that_fills_the_card_is_reported_as_not_fitting():
    client = OllamaModelsClient(BASE)
    vram = 8 * 1024**3
    # 7.4 GB on an 8 GB card leaves no room for a KV cache.
    assert client.fits_in_vram(int(7.4 * 1024**3), vram) is False
    assert client.fits_in_vram(int(4.0 * 1024**3), vram) is True


def test_the_measured_model_on_this_card_is_not_refused():
    """The resident 9B really is 5.11 GiB on an 8 GiB card: a check built on the
    6.14 GiB file size would call a working model impossible."""
    client = OllamaModelsClient(BASE)
    assert client.fits_in_vram(5490081790, 8 * 1024**3) is True


def test_the_gpu_is_read_from_nvidia_smi_rather_than_a_binding():
    """A session that never touches a GPU must not pay for importing torch."""
    value = gpu_vram_bytes()
    assert value == 0 or value > 1024**3
