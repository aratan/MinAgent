"""Downloading a file, and where it is allowed to land.

Downloads have one folder to go to. The agent works in a project directory, and
a fetched file dropped at the workspace root is indistinguishable from a source
file the next time somebody lists the directory - it pollutes the inventory the
model reads, and it is easy to lose. So ``salida/`` is not advice given in a
prompt, it is the only place the tool will write: the name is resolved inside
that folder and anything that would escape it is refused.

The tool is a capability of its own rather than part of the file tools, because
it is the one write in the session that reaches the network. On-demand loading
is what keeps an unused network tool out of every request.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from .errors import AgentError

DOWNLOAD_TOOL_NAME = "download_file"

# Every download lands here, relative to the workspace root.
OUTPUT_DIRNAME = "salida"

DEFAULT_TIMEOUT_SECONDS = 120
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_NAME_CHARS = 120
MAX_URL_CHARS = 2000

_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def create_download_tools() -> list[dict[str, Any]]:
    """The tool schema for the ``files.download`` capability."""
    return [
        {
            "type": "function",
            "function": {
                "name": DOWNLOAD_TOOL_NAME,
                "description": (
                    "Download a file from an http(s) URL into the workspace. The file is always written "
                    f"to {OUTPUT_DIRNAME}/ - pass only the file name, not a path. Overwrites nothing: a "
                    "second download of the same name gets a numbered suffix. Use web_fetch to read a "
                    "page's text; this is for keeping the file itself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The http or https URL to download.",
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "File name to save as, e.g. informe.pdf. Optional: taken from the URL when "
                                "it is missing. Extensions in a name are not trusted, the URL decides the real type."
                            ),
                        },
                    },
                    "required": ["url"],
                },
            },
        }
    ]


def _sanitize(name: str) -> str:
    """Reduce a requested name to something that cannot climb out of salida/."""
    name = unquote(name).replace("\\", "/").split("/")[-1]
    name = _UNSAFE_NAME_CHARS.sub("-", name).strip("-.")
    if not name or name in {".", ".."}:
        raise AgentError("download_file needs a file name to save as.")
    if len(name) > MAX_NAME_CHARS:
        suffix = Path(name).suffix[:16]
        name = name[: MAX_NAME_CHARS - len(suffix)] + suffix
    return name


def _name_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    candidate = path.rsplit("/", 1)[-1]
    return _sanitize(candidate) if candidate else ""


def _unique_path(directory: Path, name: str) -> Path:
    """A free path in ``directory``, numbered rather than overwriting."""
    path = directory / name
    if not path.exists():
        return path
    stem, suffix = Path(name).stem, Path(name).suffix
    for counter in range(2, 1000):
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
    raise AgentError(f"Could not find a free name for {name} in {directory}.")


def resolve_destination(root: Path, name: str) -> Path:
    """Where a download of ``name`` goes, and proof it stays under salida/.

    The check is on resolved paths, not on the name, because a name that looks
    harmless can still land elsewhere. ``_sanitize`` has already reduced the
    name to a single safe segment, so the two things left to rule out are a
    symlinked ``salida/`` pointing out of the workspace, and a name that
    resolves to something other than a child of it.
    """
    resolved_root = root.resolve()
    output_dir = (resolved_root / OUTPUT_DIRNAME).resolve()
    if output_dir.parent != resolved_root:
        raise AgentError(
            f"{OUTPUT_DIRNAME}/ resolves to {output_dir}, outside the workspace {resolved_root}. "
            "download_file refuses to write through it."
        )
    candidate = (output_dir / _sanitize(name)).resolve()
    if candidate.parent != output_dir:
        raise AgentError(f"download_file writes only into {OUTPUT_DIRNAME}/.")
    return candidate


async def run_download(
    url: str,
    name: str | None,
    root: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Fetch ``url`` and save it under ``salida/``; return a line to show the user."""
    if not isinstance(url, str) or not url.strip():
        raise AgentError("download_file requires a url.")
    url = url.strip()
    if len(url) > MAX_URL_CHARS:
        raise AgentError(f"The url exceeds {MAX_URL_CHARS} characters.")
    scheme = urlparse(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise AgentError(f"download_file reads http and https urls, not {scheme or 'a bare path'}.")

    chosen = _sanitize(name) if name and str(name).strip() else _name_from_url(url)
    destination = resolve_destination(root, chosen)
    destination.parent.mkdir(parents=True, exist_ok=True)

    async with (
        httpx.AsyncClient(timeout=timeout_seconds, transport=transport, follow_redirects=True) as client,
        client.stream("GET", url) as response,
    ):
        if response.status_code >= 400:
            raise AgentError(f"The server answered {response.status_code} for {url}.")
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
            raise AgentError(
                f"The file is {int(declared) // (1024 * 1024)} MiB, over the "
                f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit."
            )
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                raise AgentError(f"The download passed the {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit.")
            chunks.append(chunk)
        body = b"".join(chunks)
        content_type = response.headers.get("content-type", "").split(";")[0].strip()

    if not body:
        raise AgentError(f"{url} returned an empty body.")
    path = _unique_path(destination.parent, destination.name)
    path.write_bytes(body)
    detail = f" ({content_type})" if content_type else ""
    return f"Downloaded {url} to {OUTPUT_DIRNAME}/{path.name}{detail}, {size} bytes."
