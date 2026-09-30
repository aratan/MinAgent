"""Meaning, not vocabulary: local embeddings for comparing two memories.

Two memories that say the same thing do not have to share a word. "Use uv to
install" and "La instalación se hace con uv" are one memory said twice, and a
word-by-word comparison calls them two, which fills the hint block with one
answer in two voices and makes both look weaker than they are.

An embedding model settles it. ``nomic-embed-text`` is 274 MB, answers in
milliseconds, and runs on the CPU as happily as on a card that already has the
session model resident - which matters, because this runs inside ``remember``,
on the path of every memory the agent writes.

Two things are deliberate here:

* **The request is batched and cached.** A save compares against what is already
  stored, so the same stored memory is embedded over and over. The cache is keyed
  by the exact text and is bounded, so a long session does not grow one.
* **Every failure is silent and returns ``None``.** No model pulled, no server
  running, a request that times out, a vector of the wrong length: the caller
  falls back to comparing words, which is worse and already measured. A
  comparison that cannot be had must not cost a memory.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import httpx

from .compute import ollama_base_url

DEFAULT_EMBED_MODEL = "nomic-embed-text"
"""Small, fast, and multilingual enough not to matter; the store is mostly Spanish."""

EMBED_TIMEOUT_SECONDS = 30.0
"""Short on purpose: this blocks a memory write, and a slow answer is a lost memory."""

CACHE_LIMIT = 512
"""Bounded because the store can hold two thousand memories and the session does not last for ever.

Enough that the same stored memory is embedded once per session rather than once
per save; the oldest entry goes out first, which costs nothing, since a
re-embedding is one fast request.
"""


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """How much two vectors point the same way, from 0 to 1.

    Zero-length or mismatched vectors score 0 rather than raising: a vector that
    cannot be compared is not a duplicate, and the caller is about to fall back
    to a lexical comparison that answers the question anyway.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for first, second in zip(left, right, strict=True):
        dot += first * second
        left_norm += first * first
        right_norm += second * second
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / math.sqrt(left_norm * right_norm)))


class Embedder:
    """One embedding model on the local Ollama server, with a cache in front of it."""

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        base_url: str | None = None,
        timeout: float = EMBED_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model.strip() or DEFAULT_EMBED_MODEL
        self.base_url = base_url
        self.timeout = timeout
        self.transport = transport
        self._cache: dict[str, tuple[float, ...]] = {}
        # Set once the server has proved it cannot answer, so a memory write is
        # not repeated against a dead endpoint for the rest of the session.
        self.unavailable = False

    def _url(self) -> str | None:
        return self.base_url or ollama_base_url()

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]] | None:
        """Vectors for ``texts``, in the same order, or ``None`` if they cannot be had.

        Only the texts not already cached are asked for, in one request: the
        server batches, and a save that compares against a thousand stored
        memories would otherwise be a thousand round trips.
        """
        if self.unavailable or not texts:
            return None
        base = self._url()
        if not base:
            self.unavailable = True
            return None
        wanted = [text for text in texts if text not in self._cache]
        if wanted:
            vectors = await self._request(base, wanted)
            if vectors is None:
                return None
            for text, vector in zip(wanted, vectors, strict=True):
                self._cache[text] = vector
                while len(self._cache) > CACHE_LIMIT:
                    self._cache.pop(next(iter(self._cache)))
        return [self._cache[text] for text in texts if text in self._cache]

    async def _request(self, base: str, texts: Sequence[str]) -> list[tuple[float, ...]] | None:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout), transport=self.transport
            ) as client:
                response = await client.post(
                    f"{base}/api/embed",
                    json={"model": self.model, "input": list(texts)},
                )
            if response.status_code >= 400:
                self.unavailable = True
                return None
            payload: Any = response.json()
        except (httpx.HTTPError, OSError, ValueError):
            self.unavailable = True
            return None
        vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            return None
        cleaned: list[tuple[float, ...]] = []
        for vector in vectors:
            if not isinstance(vector, list) or not vector:
                return None
            try:
                cleaned.append(tuple(float(value) for value in vector))
            except (TypeError, ValueError):
                return None
        if len({len(vector) for vector in cleaned}) != 1:
            # A batch answered with mixed dimensions cannot be compared; asking
            # again with the server's cache warm usually fixes it, and if it does
            # not the caller falls back to words.
            return None
        return cleaned
