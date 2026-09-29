"""Loading an image on demand, one call at a time.

A vision-capable main model sees the pixels it is given, so an image path in a
message used to mean the whole image went into the next request. On a 1024x1024
PNG that is about 2000 tokens, a quarter of an 8k window, paid whether or not
the model needs the pixels - and once in the transcript it stays there.

So the image is noticed, not sent. ``prepare_user_message`` leaves the path in
the text and hands the list of what it found back to the agent; this module is
the other half. The model has to name the ``images`` capability and call
``view_image`` before any pixel exists, and only then is the image added.

The pixels are not smuggled through the tool result text. The turn loop already
knows how to take an image out of a tool result and put it in the following
request (see the ``pending_images`` branch), so the tool returns the same
``{"tool_text": ..., "image": ...}`` shape any other tool would.
"""

from __future__ import annotations

import base64
from typing import Any

from .errors import AgentError
from .image import detect_image_mime_type

# Reading an image means reading a file, so the same workspace limits apply.
from .workspace import MAX_READ_BYTES

VIEW_IMAGE_TOOL_NAME = "view_image"


def create_image_tools() -> list[dict[str, Any]]:
    """The tool schemas for the ``images`` capability."""
    return [
        {
            "type": "function",
            "function": {
                "name": VIEW_IMAGE_TOOL_NAME,
                "description": (
                    "Load an image from the workspace and actually see it. Use this when the answer "
                    "depends on what an image shows - a screenshot, a photo, a diagram, a licence plate, "
                    "what is on the screen - and not when the path alone answers the question. The "
                    "pixels arrive in the next request, so the following turn is where you can see it. "
                    "Each call puts a couple of thousand tokens in the context, so ask for one image at "
                    "a time and only when it is needed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to the image, e.g. salida/captura.png.",
                        }
                    },
                    "required": ["path"],
                },
            },
        }
    ]


async def run_view_image(path: str, workspace_access: Any, allow_outside: bool = True) -> dict[str, Any]:
    """Read an image and hand back the pixels for the next request.

    Returns the tool-result dict the turn loop already understands. The text half
    is deliberately dull: the model can see the image itself, so anything longer
    is tokens spent describing what is about to be in front of it.
    """
    if not isinstance(path, str) or not path.strip():
        raise AgentError("view_image needs a non-empty path.")
    path = path.strip()

    data = await workspace_access.read_raw_file(path, allow_outside=allow_outside)
    if len(data) > MAX_READ_BYTES:
        raise AgentError(f"Image exceeds the {MAX_READ_BYTES} byte limit.")
    mime_type = detect_image_mime_type(data)
    if not mime_type:
        raise AgentError(f"{path} is not an image; view_image reads PNG, JPEG, GIF, and WebP.")

    image = {
        "path": path,
        "mime_type": mime_type,
        "data": base64.b64encode(data).decode("ascii"),
    }
    return {
        "tool_text": f"Loaded {path} ({mime_type}). You can see it in the next request.",
        "image": image,
    }
