"""Tests for download_file and the one folder it may write to.

The rule this file exists to protect: a download lands in ``salida/``, whatever
name the model asks for. A name is untrusted input that arrives from a model
reading a URL, so the tests are mostly about the ways it tries to point
somewhere else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from minagent.download import (
    DOWNLOAD_TOOL_NAME,
    OUTPUT_DIRNAME,
    create_download_tools,
    resolve_destination,
    run_download,
)
from minagent.errors import AgentError


def _transport(body: bytes = b"contenido", content_type: str = "text/plain") -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: httpx.Response(
            200, content=body, headers={"content-type": content_type}, request=request
        )
    )


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# ------------------------------------------------------------ where it lands


@pytest.mark.asyncio
async def test_a_download_lands_in_salida(tmp_path):
    result = await run_download("https://example.com/datos.csv", "datos.csv", tmp_path, transport=_transport())
    assert _files(tmp_path) == [f"{OUTPUT_DIRNAME}/datos.csv"]
    assert (tmp_path / OUTPUT_DIRNAME / "datos.csv").read_bytes() == b"contenido"
    assert "datos.csv" in result


@pytest.mark.asyncio
async def test_the_name_defaults_to_the_one_in_the_url(tmp_path):
    await run_download("https://example.com/carpeta/informe.pdf", None, tmp_path, transport=_transport())
    assert _files(tmp_path) == [f"{OUTPUT_DIRNAME}/informe.pdf"]


@pytest.mark.asyncio
async def test_a_url_without_a_usable_name_still_needs_one(tmp_path):
    with pytest.raises(AgentError):
        await run_download("https://example.com/", None, tmp_path, transport=_transport())


@pytest.mark.asyncio
async def test_nothing_else_is_ever_written(tmp_path):
    """Not the workspace root, not a subfolder: salida/ and only salida/."""
    await run_download("https://example.com/a.txt", "a.txt", tmp_path, transport=_transport())
    await run_download("https://example.com/b.txt", "b.txt", tmp_path, transport=_transport())
    assert _files(tmp_path) == [f"{OUTPUT_DIRNAME}/a.txt", f"{OUTPUT_DIRNAME}/b.txt"]


# --------------------------------------------------------- refusing to escape


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "landed"),
    [
        ("../fuera.txt", "fuera.txt"),
        ("../../etc/passwd", "passwd"),
        ("sub/carpeta.txt", "carpeta.txt"),
        ("/etc/passwd", "passwd"),
        ("..\\windows.txt", "windows.txt"),
    ],
)
async def test_a_name_with_a_path_in_it_still_lands_in_salida(tmp_path, name, landed):
    """A path in the name is reduced to its last segment, not obeyed."""
    await run_download("https://example.com/x", name, tmp_path, transport=_transport())
    assert _files(tmp_path) == [f"{OUTPUT_DIRNAME}/{landed}"]
    assert not (tmp_path.parent / "fuera.txt").exists()


def test_a_symlinked_salida_cannot_redirect_writes_out_of_the_workspace(tmp_path):
    """Otherwise a link is a way out, and a name check would not see it."""
    workspace = tmp_path / "proyecto"
    workspace.mkdir()
    elsewhere = tmp_path / "fuera-del-proyecto"
    elsewhere.mkdir()
    (workspace / OUTPUT_DIRNAME).symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(AgentError):
        resolve_destination(workspace, "datos.txt")
    assert not (elsewhere / "datos.txt").exists()


def test_a_symlinked_salida_inside_the_workspace_is_fine(tmp_path):
    """The check is about leaving the workspace, not about links themselves."""
    elsewhere = tmp_path / "otro"
    elsewhere.mkdir()
    (tmp_path / OUTPUT_DIRNAME).symlink_to(elsewhere, target_is_directory=True)
    assert resolve_destination(tmp_path, "datos.txt") == elsewhere / "datos.txt"


def test_a_name_that_reduces_to_nothing_is_refused(tmp_path):
    with pytest.raises(AgentError):
        resolve_destination(tmp_path, "///")


# ------------------------------------------------------------------ the basics


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "solo-una-ruta"])
async def test_only_http_and_https(tmp_path, url):
    with pytest.raises(AgentError):
        await run_download(url, "x", tmp_path, transport=_transport())


@pytest.mark.asyncio
async def test_a_second_download_does_not_overwrite_the_first(tmp_path):
    await run_download("https://example.com/datos.csv", "datos.csv", tmp_path, transport=_transport(b"uno"))
    await run_download("https://example.com/datos.csv", "datos.csv", tmp_path, transport=_transport(b"dos"))
    assert _files(tmp_path) == [f"{OUTPUT_DIRNAME}/datos-2.csv", f"{OUTPUT_DIRNAME}/datos.csv"]
    assert (tmp_path / OUTPUT_DIRNAME / "datos.csv").read_bytes() == b"uno"


@pytest.mark.asyncio
async def test_an_error_status_is_reported_not_written(tmp_path):
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    with pytest.raises(AgentError):
        await run_download("https://example.com/no.txt", "no.txt", tmp_path, transport=httpx.MockTransport(failing))
    assert _files(tmp_path) == []


@pytest.mark.asyncio
async def test_an_empty_body_is_not_a_file(tmp_path):
    with pytest.raises(AgentError):
        await run_download("https://example.com/vacio", "vacio", tmp_path, transport=_transport(b""))


@pytest.mark.asyncio
async def test_a_declared_size_over_the_limit_is_refused_before_the_body(tmp_path):
    def big(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 10, headers={"content-length": "999999999"}, request=request)

    with pytest.raises(AgentError):
        await run_download("https://example.com/x", "x", tmp_path, transport=httpx.MockTransport(big))


@pytest.mark.asyncio
async def test_a_url_longer_than_the_limit_is_refused(tmp_path):
    with pytest.raises(AgentError):
        await run_download("https://example.com/" + "a" * 3000, "x", tmp_path, transport=_transport())


# --------------------------------------------------------------- the tool side


def test_the_tool_says_where_the_file_lands():
    """The model has to be able to find the file without being told twice."""
    (schema,) = create_download_tools()
    assert schema["function"]["name"] == DOWNLOAD_TOOL_NAME
    assert OUTPUT_DIRNAME in schema["function"]["description"]


def test_the_schema_asks_for_a_name_and_a_url():
    (schema,) = create_download_tools()
    parameters: dict[str, Any] = schema["function"]["parameters"]
    assert parameters["required"] == ["url"]
    assert set(parameters["properties"]) == {"url", "name"}
