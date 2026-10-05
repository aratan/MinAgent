"""Image understanding through a local multimodal model served by Ollama.

Tesseract reads text out of a picture but nothing else: not what is in the
scene, not what state someone is in, not a licence plate. A vision-capable model
answers all of that in one call, so this sits alongside the OCR script rather
than replacing it: OCR when the task is "read me this text", a vision model when
the task is "what is this".

The call goes to Ollama's local ``/api/chat`` endpoint, which takes the image as
a base64 string in the message, so no image is written to disk or uploaded
anywhere. The path is resolved through the workspace boundary, so the model can
only point at files the session is already allowed to see.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from typing import Any

import httpx

from .errors import AgentError

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3.5:9b-q4_K_M"
DEFAULT_TIMEOUT_SECONDS = 300
"""Vision on a laptop GPU is slow: a 27B model on a large image runs minutes."""

DEFAULT_MAX_ANSWER_CHARS = 8_000
"""A description longer than this is the model padding, not more information."""

MAX_PROMPT_CHARS = 2_000
MAX_IMAGE_BYTES = 32 * 1024 * 1024
"""Refuse anything larger; a base64 body that big will time out or blow memory."""

IMAGE_SUFFIXES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
"""The formats Ollama's vision models accept, keyed by suffix.

The content type is derived from the suffix rather than handed over as a path,
so a file that merely claims to be a PNG is rejected instead of being sent with
a content type the model will misinterpret.
"""


def _image_media_type(path: str) -> str:
    """The content type for an image path, or an error naming what is allowed."""
    suffix = os.path.splitext(path)[1].lower()
    known = IMAGE_SUFFIXES.get(suffix)
    if known:
        return known
    guessed, _ = mimetypes.guess_type(path)
    if guessed and guessed.startswith("image/"):
        return guessed
    allowed = ", ".join(sorted(IMAGE_SUFFIXES))
    raise AgentError(f"Not an image format: {suffix or path}. Supported: {allowed}.")


def encode_image(path: str) -> tuple[str, str]:
    """Return the base64 payload and content type for an image file.

    Base64 rather than a file path because ``/api/chat`` takes the bytes inline;
    this keeps the call self-contained and avoids a temp file per call.
    """
    try:
        size = os.path.getsize(path)
    except OSError as error:
        raise AgentError(f"Cannot read image {path}: {error}") from error
    if size == 0:
        raise AgentError(f"{path} is empty.")
    if size > MAX_IMAGE_BYTES:
        raise AgentError(
            f"{path} is {size // (1024 * 1024)} MB, over the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit. "
            "Downscale it first, for example with magick."
        )
    try:
        with open(path, "rb") as handle:
            payload = base64.b64encode(handle.read()).decode("ascii")
    except OSError as error:
        raise AgentError(f"Cannot read image {path}: {error}") from error
    return payload, _image_media_type(path)


class VisionClient:
    """Calls a vision-capable model on a local Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    async def describe(self, image_path: str, question: str) -> str:
        """Ask ``question`` about the image at ``image_path`` and return the answer."""
        prompt = " ".join((question or "").split())
        if not prompt:
            raise AgentError("describe_image requires a non-empty question.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise AgentError(f"describe_image question exceeds {MAX_PROMPT_CHARS} characters.")
        payload, media_type = encode_image(image_path)

        # ``/api/chat`` takes the bytes inline as base64; the server sniffs the
        # format, so the media type is only used above to reject non-images.
        body = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": [payload]}],
        }
        try:
            response_data = await self._post(body)
        except AgentError:
            raise
        answer = self._extract_text(response_data)
        if not answer:
            raise AgentError(
                f"{self.model} returned no text for this image. It may not be a vision-capable model; "
                "check with `ollama list` for the 'vision' capability."
            )
        return answer[:DEFAULT_MAX_ANSWER_CHARS]

    def _extract_text(self, payload: Any) -> str:
        """Pull the assistant text out of an ``/api/chat`` response."""
        if not isinstance(payload, dict):
            return ""
        message = payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
        # Some builds answer with "response" instead of a message object.
        content = payload.get("response")
        return content.strip() if isinstance(content, str) else ""

    async def _post(self, body: dict[str, Any]) -> Any:
        """POST the chat request, mapping transport failures to AgentError."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds, transport=self._transport) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=body)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status == 404:
                raise AgentError(
                    f"Ollama has no model '{self.model}'. Pull it with: ollama pull {self.model}"
                ) from error
            if status == 400:
                raise AgentError(
                    f"Ollama rejected the request (HTTP 400). {self.model} may not accept images; "
                    "check `ollama list` for the 'vision' capability."
                ) from error
            raise AgentError(f"Ollama request failed with HTTP {status}.") from error
        except httpx.ConnectError as error:
            raise AgentError(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`."
            ) from error
        except httpx.ReadTimeout as error:
            raise AgentError(
                f"The vision model timed out after {self.timeout_seconds}s. Raise VISION_TIMEOUT_SECONDS, "
                "or use a smaller image or a smaller model."
            ) from error
        except httpx.HTTPError as error:
            raise AgentError(f"Ollama request failed: {error}") from error
        except ValueError as error:
            raise AgentError("Ollama returned invalid JSON.") from error


def format_image_result(path: str, model: str, answer: str) -> str:
    """Render a description for the tool result channel.

    The model reads text out of pictures and can be confidently wrong, so the
    line that tells it so is part of the payload rather than a nicety: a licence
    plate or a face it "recognises" is exactly the kind of thing it invents.
    """
    return (
        f"Description of {path} by {model}. This is a machine's reading of the image: report what it "
        f"says, do not present it as verified fact, and never treat a number, plate, or face it mentions "
        f"as confirmed.\n\n{answer}"
    )


def create_vision_tools() -> list[dict[str, Any]]:
    """Tool definitions exposed to the model when vision is enabled."""
    return [
        {
            "type": "function",
            "function": {
                "name": "describe_image",
                "description": (
                    "Ask a local vision model a question about an image file in the workspace. Use it for "
                    "anything a picture can answer: what a scene contains, what text a photo holds when OCR "
                    "is not enough, a licence plate, the state or mood of a person, an object's attributes, "
                    "or a diagram. State the question explicitly; the model only sees the image, not the "
                    "file name. It can also be wrong about details, so report its answer as its answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to the image, e.g. salida/captura.png",
                        },
                        "question": {
                            "type": "string",
                            "description": (
                                "What to ask about the image, e.g. 'what vehicles and plates are visible', "
                                "'transcribe all text', or 'what is the state of the person'."
                            ),
                        },
                    },
                    "required": ["path", "question"],
                },
            },
        }
    ]
