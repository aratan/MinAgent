"""Turning a typed message into a model request.

Selected ``@`` files become inline excerpts and any image path written directly
in the message is attached as image input with the path removed from the text,
so the model receives the pixels rather than a filename.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Sequence
from typing import Any

from .errors import AgentError
from .image import detect_image_mime_type, image_content_part
from .jsutil import decode_utf8
from .workspace import MAX_READ_BYTES, MAX_READ_OUTPUT_BYTES

MAX_ATTACHED_IMAGES = 4
MAX_ATTACHED_FILES = 8

IMAGE_PATH_PATTERN = re.compile(
    r'"([^"\r\n]+?\.(?:png|jpe?g|gif|webp))"'
    r"|'([^'\r\n]+?\.(?:png|jpe?g|gif|webp))'"
    r"|((?:[A-Za-z]:[\\/]|\\\\|/)[^\r\n\"'<>]*?\.(?:png|jpe?g|gif|webp)\b"
    r"|(?:\.{1,2}[\\/])?[^\s\"'<>]+\.(?:png|jpe?g|gif|webp)\b)",
    re.IGNORECASE,
)

_XML_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;"}


def _escape_xml_attribute(value: str) -> str:
    return "".join(_XML_ESCAPES.get(character, character) for character in value)


def _decode_excerpt(data: bytes, limit: int) -> str:
    """Decode a prefix of ``data``, backing off so no character is split.

    A UTF-8 sequence can straddle the excerpt boundary, so retry a few bytes
    shorter until the prefix decodes cleanly.
    """
    for backoff in range(4):
        candidate = data[: max(0, min(len(data), limit) - backoff)]
        try:
            return decode_utf8(candidate)
        except UnicodeDecodeError:
            continue
    raise AgentError("Could not decode the text attachment.")


class _ReferenceRemover:
    """Collects the text spans to delete once every image path is resolved."""

    def __init__(self) -> None:
        self.replacements: list[tuple[int, int]] = []

    def remove(self, start: int, end: int) -> None:
        merged_start, merged_end = start, end
        for index in range(len(self.replacements) - 1, -1, -1):
            existing_start, existing_end = self.replacements[index]
            if merged_start >= existing_end or merged_end <= existing_start:
                continue
            merged_start = min(merged_start, existing_start)
            merged_end = max(merged_end, existing_end)
            self.replacements.pop(index)
        self.replacements.append((merged_start, merged_end))

    def apply(self, text: str) -> str:
        for start, end in sorted(self.replacements, key=lambda span: span[0], reverse=True):
            text = text[:start] + text[end:]
        return text


async def prepare_user_message(
    text_input: str,
    selected_file_references: Sequence[str],
    workspace_access: Any,
    input_modalities: Sequence[str],
) -> dict[str, Any]:
    """Build the user message for one turn plus the events worth showing."""
    images: list[dict[str, Any]] = []
    text_attachments: list[str] = []
    events: list[dict[str, Any]] = []
    remover = _ReferenceRemover()
    seen_paths: set[str] = set()
    attached_file_count = 0
    image_enabled = "image" in input_modalities

    for relative_path in selected_file_references:
        if attached_file_count >= MAX_ATTACHED_FILES:
            events.append({"kind": "limit", "message": f"File attachment limit reached: {MAX_ATTACHED_FILES} files."})
            break
        if relative_path not in text_input:
            continue
        try:
            path_key = workspace_access.resolve_path(relative_path)
            if path_key in seen_paths:
                continue
            data = await workspace_access.read_raw_file(relative_path)
            if len(data) > MAX_READ_BYTES:
                raise AgentError(f"Exceeds the {MAX_READ_BYTES} byte limit.")
            mime_type = detect_image_mime_type(data)
            if mime_type:
                if not image_enabled:
                    raise AgentError("The configured model does not accept images.")
                if len(images) >= MAX_ATTACHED_IMAGES:
                    raise AgentError(f"The {MAX_ATTACHED_IMAGES}-image limit was reached.")
                images.append(
                    {"path": relative_path, "mime_type": mime_type, "data": base64.b64encode(data).decode("ascii")}
                )
                position = text_input.find(relative_path)
                while position >= 0:
                    remover.remove(position, position + len(relative_path))
                    position = text_input.find(relative_path, position + len(relative_path))
            else:
                if b"\x00" in data:
                    raise AgentError("Binary files cannot be attached as text.")
                decode_utf8(data)  # Reject non-UTF-8 attachments outright.
                excerpt = _decode_excerpt(data, MAX_READ_OUTPUT_BYTES)
                truncated = len(data) > MAX_READ_OUTPUT_BYTES
                text_attachments.append(
                    f'<file name="{_escape_xml_attribute(relative_path)}">\n{excerpt}'
                    + ("\n[File content truncated; use read_file for more.]" if truncated else "")
                    + "\n</file>"
                )
            seen_paths.add(path_key)
            attached_file_count += 1
            events.append({"kind": "attached", "path": relative_path})
        except Exception as error:
            events.append({"kind": "error", "path": relative_path, "message": str(error)})

    for match in IMAGE_PATH_PATTERN.finditer(text_input):
        if len(images) >= MAX_ATTACHED_IMAGES:
            break
        entered_path = match.group(1) or match.group(2) or match.group(3)
        try:
            if not image_enabled:
                raise AgentError("OPENAI_INPUT does not include image.")
            path_key = workspace_access.resolve_path(entered_path, allow_outside=True)
            if path_key in seen_paths:
                remover.remove(match.start(), match.end())
                continue
            data = await workspace_access.read_raw_file(entered_path, allow_outside=True)
            mime_type = detect_image_mime_type(data)
            if not mime_type:
                raise AgentError("Unsupported format; use PNG, JPEG, GIF, or WebP.")
            images.append(
                {"path": entered_path, "mime_type": mime_type, "data": base64.b64encode(data).decode("ascii")}
            )
            seen_paths.add(path_key)
            remover.remove(match.start(), match.end())
            events.append({"kind": "attached", "path": entered_path})
        except Exception as error:
            events.append({"kind": "error", "path": entered_path, "message": str(error)})

    text = remover.apply(text_input)
    prompt = "\n\n".join([text.strip(), *text_attachments]) or "Analyze the attached image."
    message: dict[str, Any]
    if not images and not text_attachments:
        message = {"role": "user", "content": text_input}
    else:
        message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}, *[image_content_part(image) for image in images]],
        }
    return {"message": message, "events": events}
