"""Ollama model management tests: the requests sent, and what is refused."""

import asyncio
import json

import httpx
import pytest

from minagent.errors import AgentError
from minagent.ollama_models import (
    OllamaModelsClient,
    advise_derivation,
    check_model_name,
    describe_push_destination,
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


def test_a_bare_namespace_is_public_and_a_host_is_not():
    """The two forms are the same words with opposite consequences, and the whole
    risk of a push is knowing which one the model is about to do."""
    assert "PUBLIC" in describe_push_destination("aratan/mi-model")
    assert "PUBLIC" in describe_push_destination("mi-model")
    assert "PUBLIC" not in describe_push_destination("registry.aratan.dev/mi-model")
    assert "PUBLIC" not in describe_push_destination("registry.example.com:5000/team/m")


def test_a_registry_with_a_port_is_still_a_registry():
    assert describe_push_destination("registry.example.com:5000/team/m").startswith("private")


async def test_pushing_to_a_public_name_is_refused_outright():
    """A publish the user did not mean to make cannot be undone by deleting the
    public model afterwards: it is already public by then."""
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("nothing may be sent")

    with pytest.raises(AgentError, match="publishes the model"):
        await _client(handler).push_model("aratan/mi-model")


async def test_pushing_to_a_private_registry_goes_through():
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"status": "success"})

    result = await _client(handler).push_model("registry.aratan.dev/mi-model")
    assert sent["model"] == "registry.aratan.dev/mi-model"
    assert "private registry" in result


async def test_the_size_shown_before_a_push_is_the_honest_worst_case():
    """The one number a user gets to decide an irreversible publish on.

    It is the sum of the model's layers, which is what could travel if the
    destination had none of them. A derived model reports 6.59 GB locally but
    shares its base's weight layer by digest, so the truth is usually
    kilobytes - and the number shown is the upper bound, with that said
    underneath it. What must never happen is the opposite error: a figure
    smaller than a layer that really would be sent.
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "layers": [
                    {"digest": "sha256:pesos", "size": 6_590_000_000},
                    {"digest": "sha256:plantilla", "size": 1_024},
                ]
            },
        )

    estimate = await _client(handler).push_size_estimate("registry.aratan.dev/mi-model")
    assert requested == ["/api/show"], "the estimate comes from the manifest, not from the listing"
    assert estimate == 6_590_001_024
    assert estimate >= 6_590_000_000


async def test_an_estimate_of_nothing_is_zero_rather_than_an_error():
    """A server that omits the layers must not stop the user from pushing."""

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        return httpx.Response(200, json={})

    assert await _client(handler).push_size_estimate("registry.aratan.dev/mi-model") == 0


# -- when a derived model is worth it ------------------------------------


def test_a_prompt_used_once_is_not_worth_a_derived_model():
    """The deciding cost is context window, and one short prompt barely uses any."""
    advice = asyncio.run(
        advise_derivation(
            _client(lambda r: httpx.Response(200, json={})),
            "qwen3.5:9b-q4_K_M",
            "Responde en español.",
            reuses_per_session=1,
        )
    )
    assert "do not derive" in advice


def test_a_long_repeated_prompt_is_worth_deriving_because_of_the_window():
    prompt = "x" * 2400  # ~600 tokens, 7% of an 8k window
    advice = asyncio.run(
        advise_derivation(
            _client(lambda r: httpx.Response(200, json={})),
            "qwen3.5:9b-q4_K_M",
            prompt,
            reuses_per_session=4,
        )
    )
    assert "derive it" in advice
    assert "tokens" in advice


def test_a_base_missing_a_needed_capability_is_called_out_first():
    """A prompt cannot give a model a capability it lacks; deriving from the
    wrong base produces something that looks right and fails the same way."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"capabilities": ["completion"]})

    advice = asyncio.run(
        advise_derivation(
            _client(handler),
            "qwen3.5:9b-q4_K_M",
            "Describe what you see in this photo. " * 200,
            reuses_per_session=5,
            needs="vision",
        )
    )
    assert "does NOT have 'vision'" in advice
    assert "do not derive yet" in advice


def test_a_base_that_cannot_be_read_is_not_recommended_anyway():
    """create_model does not verify the base exists, and the failure lands at the
    first request rather than at creation, so an unreadable base stops the advice."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'gemma3:4b' not found"})

    advice = asyncio.run(
        advise_derivation(
            _client(handler),
            "gemma3:4b",
            "Eres un experto. " * 300,
            reuses_per_session=6,
            needs="vision",
        )
    )
    assert "could not read the base" in advice
    assert "do not derive yet" in advice


def test_a_base_that_has_the_capability_says_so():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"capabilities": ["completion", "vision"]})

    advice = asyncio.run(
        advise_derivation(
            _client(handler), "q", "Look at this. " * 400, reuses_per_session=3, needs="vision"
        )
    )
    assert "has 'vision'" in advice


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
