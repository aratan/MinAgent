"""Image type detection from magic bytes.

The agent trusts the file's own header rather than its extension, so a renamed
executable is never attached to the model as an image.
"""

from __future__ import annotations

from typing import Any

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def detect_image_mime_type(data: bytes) -> str | None:
    """Return the MIME type of a supported image, or ``None``."""
    if len(data) >= 8 and data[:8] == _PNG_MAGIC:
        return "image/png"
    if len(data) >= 3 and data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF:
        return "image/jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_content_part(image: dict[str, Any]) -> dict[str, Any]:
    """Build the OpenAI multimodal content part for an attached image."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{image['mime_type']};base64,{image['data']}"},
    }
