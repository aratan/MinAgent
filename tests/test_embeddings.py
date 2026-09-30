"""Embedding tests: the local client, its cache, and what it does when it cannot answer.

The failure paths matter more than the happy one here. This runs inside
``remember``, on the path of every memory the agent writes, so a client that
raises, hangs or answers with the wrong shape would cost memories rather than
one failed comparison.
"""

from __future__ import annotations

import json

import httpx
import pytest

from minagent.embeddings import CACHE_LIMIT, Embedder, cosine


def _embedder(handler, **kwargs) -> Embedder:
    return Embedder("nomic-embed-text", base_url="http://ollama.test", transport=httpx.MockTransport(handler))


def test_the_cosine_of_a_vector_with_itself_is_one():
    assert cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    # A vector of a different length, or of nothing, cannot be compared. It
    # scores zero - not a duplicate - instead of raising in the middle of a save.
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine([], [1.0]) == 0.0


async def test_a_batch_is_one_request_and_the_answer_is_cached():
    asked: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        asked.append(body)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in body["input"]]})

    embedder = _embedder(handler)
    first = await embedder.embed(["uno", "dos"])
    second = await embedder.embed(["uno", "dos", "tres"])

    assert first == [(1.0, 0.0), (1.0, 0.0)]
    assert len(second) == 3
    # The comparison runs on every memory write, and most of what it compares
    # against has already been embedded this session.
    assert len(asked) == 2
    assert asked[1]["input"] == ["tres"]


async def test_the_cache_is_bounded_so_a_long_session_does_not_grow_one():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in body["input"]]})

    embedder = _embedder(handler)
    for index in range(CACHE_LIMIT + 40):
        await embedder.embed([f"texto {index}"])

    assert len(embedder._cache) == CACHE_LIMIT


async def test_a_server_that_refuses_is_not_asked_again():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, json={"error": "model not found"})

    embedder = _embedder(handler)

    assert await embedder.embed(["uno"]) is None
    assert await embedder.embed(["otro"]) is None
    # A model that is not pulled is a setup problem, not a transient one: the
    # caller falls back to comparing words and stops paying for this.
    assert calls == 1
    assert embedder.unavailable is True


async def test_a_server_that_is_not_there_is_not_asked_again():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    embedder = _embedder(handler)

    assert await embedder.embed(["uno"]) is None
    assert await embedder.embed(["otro"]) is None
    assert embedder.unavailable is True


async def test_an_answer_of_the_wrong_shape_is_answered_with_nothing():
    shapes = [
        {"embeddings": [[1.0, 0.0]]},  # fewer vectors than texts
        {"embeddings": [[1.0, 0.0], "not a vector"]},
        {"embeddings": [[1.0, 0.0], []]},
        {"embeddings": "no"},
        {},
    ]
    for payload in shapes:
        embedder = _embedder(lambda request, payload=payload: httpx.Response(200, json=payload))
        assert await embedder.embed(["uno", "dos"]) is None, payload


async def test_a_batch_answered_with_mixed_dimensions_is_refused():
    # The store compares vectors to each other, and a batch of different sizes
    # cannot be compared. Asking again with the server's cache warm usually
    # fixes it, so this is not a permanent "unavailable".
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [1.0, 0.0, 0.0]]})
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [1.0, 0.0]]})

    embedder = _embedder(handler)

    assert await embedder.embed(["uno", "dos"]) is None
    assert embedder.unavailable is False
    assert await embedder.embed(["uno", "dos"]) == [(1.0, 0.0), (1.0, 0.0)]


async def test_a_body_that_is_not_json_leaves_the_words_to_the_caller():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    embedder = _embedder(handler)

    assert await embedder.embed(["uno"]) is None


async def test_without_a_server_configured_nothing_is_asked():
    embedder = Embedder("nomic-embed-text", base_url="")
    embedder._url = lambda: None  # type: ignore[method-assign]

    assert await embedder.embed(["uno"]) is None
    assert embedder.unavailable is True
