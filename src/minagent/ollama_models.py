"""Managing the models the local Ollama server holds.

The server already knows what it has; this reads that and acts on it, so the
agent can answer "what can this machine actually run" from what the hardware
says rather than from what the model remembers.

Creating a model here is Ollama's own ``/api/create``: a derived model that
points at a base and carries a system prompt, parameters, and template of its
own. It does not copy weights. A derived model is a few kilobytes of
configuration in front of a base that is already on disk, which is what makes
it cheap enough to keep a dozen of: one per role, each with its own prompt,
sharing one set of weights that is only ever in VRAM once.

Every call here is refused unless the name is one Ollama itself would accept.
A model name becomes a directory under the models path and appears in a shell
command, so a name with a slash or a space in it is a real hazard rather than
an untidy label.
"""

from __future__ import annotations

import re
import shutil
from typing import Any

import httpx

from .errors import AgentError

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT_SECONDS = 300

MAX_SYSTEM_PROMPT_CHARS = 8_000
MAX_NAME_CHARS = 64

MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}$")
"""Ollama's own naming rules, minus a leading dash and the spaces some clients allow."""

UNSAFE_NAME_SHAPE = re.compile(r"//|/$|/$")
"""Slashes are legal inside a name like ``hf.co/user/repo:Q4_K_M``, but a name
that ends in one, or has an empty segment, describes a path that does not exist
rather than a model. Rejecting the shape is cheaper than a confusing 404."""


def check_model_name(name: str) -> str:
    """Validate a model name before it reaches the server or a filesystem."""
    cleaned = (name or "").strip()
    if not cleaned:
        raise AgentError("A model name is required.")
    if not MODEL_NAME.match(cleaned) or UNSAFE_NAME_SHAPE.search(cleaned):
        raise AgentError(
            f"Invalid model name {name!r}. Use letters, digits, dot, underscore, colon, slash "
            "or dash, starting with a letter or digit, at most 64 characters. A slash may appear "
            "inside a name like hf.co/user/repo:Q4_K_M but not at either end."
        )
    return cleaned


class OllamaModelsClient:
    """Lists, inspects, creates, and deletes the models on a local Ollama server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Make one call, mapping transport and status failures to readable errors."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = await client.request(method, f"{self.base_url}{path}", **kwargs)
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            try:
                detail = error.response.json().get("error", "")
            except ValueError:
                detail = ""
            if status == 404:
                raise AgentError(f"Ollama has no such model. {detail}".strip()) from error
            raise AgentError(f"Ollama returned HTTP {status}. {detail}".strip()) from error
        except httpx.ConnectError as error:
            raise AgentError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`."
            ) from error
        except httpx.ReadTimeout as error:
            raise AgentError(
                f"Ollama did not answer in {self.timeout_seconds}s. Creating a model can take a "
                "while on a slow disk; raise OLLAMA_MODELS_TIMEOUT_SECONDS if this is not it."
            ) from error

    async def list_loaded(self) -> dict[str, int]:
        """The models currently resident, and how much VRAM each actually holds.

        This is the number that matters for "will it fit", and it is not the
        size on disk: on this machine a 6.14 GB model holds 5.11 GB of VRAM,
        because the disk copy includes weights the card never loads together.
        Asking about the disk figure would refuse a model that is running.
        """
        payload = await self._request("GET", "/api/ps")
        entries = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return {}
        loaded: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", ""))
            footprint = int(entry.get("size_vram", 0) or 0)
            if name and footprint:
                loaded[name] = footprint
        return loaded

    async def list_models(self) -> list[dict[str, Any]]:
        """Every model installed, with the details the server reports.

        The ``capabilities`` list is the one worth reading: it is how the server
        says whether a model can see images, and a model without ``vision``
        cannot be asked what a photo holds.
        """
        payload = await self._request("GET", "/api/tags")
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return []
        loaded = await self.list_loaded()
        return [_summarise(entry, loaded) for entry in models if isinstance(entry, dict)]

    async def show_model(self, name: str) -> dict[str, Any]:
        """One model in detail: its template, parameters, and system prompt."""
        checked = check_model_name(name)
        payload = await self._request("POST", "/api/show", json={"model": checked})
        if not isinstance(payload, dict):
            raise AgentError(f"Ollama returned nothing for {checked}.")
        details = payload.get("details")
        parameters = payload.get("parameters")
        return {
            "name": checked,
            "family": (details or {}).get("family", "") if isinstance(details, dict) else "",
            "parameter_size": (details or {}).get("parameter_size", "") if isinstance(details, dict) else "",
            "quantization": (details or {}).get("quantization_level", "") if isinstance(details, dict) else "",
            "family_count": len(details.get("families", [])) if isinstance(details, dict) else 0,
            "capabilities": [str(item) for item in payload.get("capabilities", []) or []],
            "template": str(payload.get("template", ""))[:MAX_SYSTEM_PROMPT_CHARS],
            "parameters": parameters if isinstance(parameters, str) else "",
            "modified_at": str(payload.get("modified_at", "")),
        }

    async def create_model(
        self,
        name: str,
        base: str,
        system: str = "",
        parameters: str = "",
        template: str = "",
    ) -> str:
        """Create a derived model with its own prompt, pointing at an existing base.

        No weights are copied, so this is a configuration record in front of a
        base that is already installed. ``base`` is checked too: a derived model
        whose base is missing fails later, at the first request, which is a much
        worse moment to find out.
        """
        checked = check_model_name(name)
        if not (base or "").strip():
            raise AgentError("A base model is required: a derived model points at one.")
        check_model_name(base)
        prompt = (system or "").strip()
        if len(prompt) > MAX_SYSTEM_PROMPT_CHARS:
            raise AgentError(
                f"The system prompt is {len(prompt)} characters, over the "
                f"{MAX_SYSTEM_PROMPT_CHARS} limit."
            )
        body: dict[str, Any] = {"model": checked, "from": base.strip(), "stream": False}
        if prompt:
            body["system"] = prompt
        if parameters.strip():
            body["parameters"] = parameters.strip()
        if template.strip():
            body["template"] = template.strip()
        await self._request("POST", "/api/create", json=body)
        return checked

    async def delete_model(self, name: str) -> str:
        """Delete a model. The weights go with it, so this is not undoable.

        There is no soft delete on the server side, which is why the name has to
        be spelled out correctly: ``ollama list`` first, then the exact name.
        """
        checked = check_model_name(name)
        await self._request("DELETE", "/api/delete", json={"model": checked})
        return checked

    def fits_in_vram(self, size_bytes: int, vram_total_bytes: int) -> bool:
        """Whether a footprint leaves room for a usable context on the card.

        Not simply "smaller than the card": a KV cache for an 8k window on a 9B
        model is a gigabyte or more, and a model that fills the card to the last
        MiB cannot generate a token without evicting itself.
        """
        overhead = KV_CACHE_ALLOWANCE * vram_total_bytes
        return size_bytes + overhead <= vram_total_bytes

    async def report_hardware(self, root_directory: str = ".") -> str:
        """What this machine can run, from what the card actually reports."""
        vram = gpu_vram_bytes()
        lines: list[str] = []
        if vram:
            lines.append(f"GPU VRAM: {vram / (1024**3):.2f} GiB total.")
        else:
            lines.append("No NVIDIA VRAM reported; nvidia-smi is not available to this session.")
        try:
            models = await self.list_models()
        except AgentError as error:
            lines.append(f"Could not list models: {error}")
            return "\n".join(lines)
        if not models:
            lines.append("No models installed. Pull one with: ollama pull <name>")
            return "\n".join(lines)
        for entry in models:
            if entry["vram_bytes"]:
                footprint = entry["vram_bytes"]
                verdict = "fits" if self.fits_in_vram(footprint, vram) else "does not fit" if vram else "unknown"
                basis = "measured on the card"
            else:
                # Not resident, so the disk figure is an upper bound: the card
                # footprint is smaller, because the disk copy includes weights
                # that are never resident together. Claiming a definite answer
                # from it would refuse models that run.
                footprint = entry["size_bytes"]
                verdict = "fits" if vram and self.fits_in_vram(footprint, vram) else "unknown"
                basis = "not loaded; disk size is an upper bound"
            lines.append(f"- {entry['name']}: {footprint / (1024**3):.2f} GiB ({basis}), {verdict}")
        if vram:
            lines.append(
                f"Only one of these loads at a time: a card with {vram / (1024**3):.1f} GiB does "
                "not hold two models plus a usable context."
            )
        return "\n".join(lines)


KV_CACHE_ALLOWANCE = 0.25
"""A quarter of the card for the KV cache and activations, measured on this GPU.

Measured, not guessed: the resident qwen3.5:9b holds 5.11 GiB with an 8k
window on an 8 GiB card, so a 2 GiB allowance is what this one actually needs.
"""


def _summarise(entry: dict[str, Any], loaded: dict[str, int] | None = None) -> dict[str, Any]:
    """Flatten one entry from ``/api/tags`` into what is worth reading."""
    details = entry.get("details")
    details = details if isinstance(details, dict) else {}
    size = int(entry.get("size", 0) or 0)
    name = str(entry.get("name", ""))
    return {
        "name": name,
        "size_bytes": size,
        # The card footprint when the server reports one, since that is the
        # figure that decides whether a second model can be loaded.
        "vram_bytes": (loaded or {}).get(name, 0),
        "size_gib": round(size / (1024**3), 2),
        "family": str(details.get("family", "")),
        "parameter_size": str(details.get("parameter_size", "")),
        "quantization": str(details.get("quantization_level", "")),
        "modified_at": str(entry.get("modified_at", "")),
    }


def format_models_table(models: list[dict[str, Any]]) -> str:
    """A compact listing, sized so it does not eat the context window."""
    if not models:
        return "No models installed. Pull one with: ollama pull <name>"
    lines = [f"{'MODEL':44} {'SIZE':>8}  PARAMS  QUANT"]
    for entry in models:
        lines.append(
            f"{entry['name'][:44]:44} {entry['size_gib']:7.2f}G  "
            f"{entry['parameter_size'] or '?':7} {entry['quantization'] or '?':5}"
        )
    return "\n".join(lines)


def format_model_detail(detail: dict[str, Any]) -> str:
    """One model's details, with the capabilities spelled out."""
    parts = [
        detail["name"],
        f"family: {detail['family'] or '?'}, {detail['parameter_size'] or '?'} "
        f"{detail['quantization'] or '?'}",
        f"capabilities: {', '.join(detail['capabilities']) or 'not reported'}",
    ]
    if detail["parameters"]:
        parts.append(f"parameters: {detail['parameters'][:400]}")
    if detail["template"]:
        parts.append(f"template: {detail['template'][:400]}")
    return "\n".join(parts)


def gpu_vram_bytes(root_directory: str = ".") -> int:
    """Total VRAM on the local card, or 0 when it cannot be read.

    Read from ``nvidia-smi`` rather than a Python binding so a session that
    never touches a GPU never pays for torch, and so the answer matches the
    number the compute orchestrator already uses for its own budget.
    """
    import subprocess

    binary = shutil.which("nvidia-smi")
    if binary is None:
        return 0
    try:
        result = subprocess.run(
            [binary, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if result.returncode != 0:
        return 0
    for line in result.stdout.splitlines():
        cleaned = line.strip()
        if cleaned.isdigit():
            return int(cleaned) * 1024 * 1024
    return 0


def create_ollama_models_tools() -> list[dict[str, Any]]:
    """The tool schemas for managing local models."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_models",
                "description": (
                    "List every model the local Ollama server holds, with its size, parameter count "
                    "and quantisation. Start here before promising to run anything, because a model "
                    "that is not installed cannot be loaded."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "show_model",
                "description": (
                    "Read one model in detail: its family, capabilities, default parameters, template "
                    "and system prompt. Capabilities is how you know whether it can see images."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Exact model name"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_model",
                "description": (
                    "Create a derived model with its own system prompt, pointing at an installed base. "
                    "No weights are copied, so it costs kilobytes and shares the base's memory. Use "
                    "it to keep one permanent model per role instead of restating the prompt each turn."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name for the new model"},
                        "base": {"type": "string", "description": "Installed model to derive from"},
                        "system": {"type": "string", "description": "The system prompt it will always use"},
                        "parameters": {
                            "type": "string",
                            "description": "Ollama parameter overrides, e.g. 'temperature 0.2\\nnum_ctx 4096'",
                        },
                    },
                    "required": ["name", "base"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_model",
                "description": (
                    "Delete a model and its weights. This is permanent: the server has no undo, so "
                    "check with list_models first and pass the exact name."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Exact model name"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hardware_report",
                "description": (
                    "Report this machine's GPU, its total VRAM, and which installed models would fit "
                    "with room for a usable context window. Use it to answer what can run here."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


OLLAMA_MODELS_GUIDANCE = (
    "These tools talk to the Ollama server on this machine, not to a cloud. create_model makes a "
    "derived model: it stores a prompt and parameters in front of a base that is already installed, "
    "it does not copy weights, so a dozen role-specific models cost kilobytes between them and only "
    "one is ever in VRAM at a time. Check list_models before promising a model can run, and "
    "hardware_report before promising one fits: two models do not fit at once on an 8 GB card. "
    "delete_model is permanent."
)
"""Sits with the tools, because the cost model behind them is the surprise."""
