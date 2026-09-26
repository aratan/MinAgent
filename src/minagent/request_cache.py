"""Replay cache for byte-identical requests.

A conversational turn almost always changes the prompt, so this cache earns
nothing on the happy path. It earns its keep on the paths that deliberately
resend the exact same request: the retry after an empty or truncated response,
a resubmitted prompt from the history ring, and a nudge that the model answered
badly. Replaying the answer that already came back for that identical input is
both faster and more consistent than paying for a second call that would
differ only in the model's mood.

Only the assistant message is stored, and only for requests that completed
normally. A cancelled, interrupted, or failed request is never replayed, because
the user is entitled to a fresh attempt.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

# Enough for the retry-after-empty case plus a short scroll back through history,
# without letting the cache grow into a second transcript.
DEFAULT_MAX_ENTRIES = 24


class RequestCache:
    """A small LRU of completions keyed by the exact request sent."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max(1, max_entries)
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> str:
        """Fingerprint everything the endpoint sees, so a hit means a true repeat.

        The model and the tool schemas are part of the key because changing
        either changes the answer even when the conversation is untouched.
        """
        digest = hashlib.sha256()
        digest.update(model.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\x00tools\x00")
        digest.update(_canonical(tools).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\x00messages\x00")
        digest.update(_canonical(messages).encode("utf-8", errors="surrogatepass"))
        return digest.hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a replayable copy of a cached completion, or ``None``.

        The whole completion is replayed, not just the text: the caller branches
        on ``truncated`` and on the usage figures, and a replay that dropped
        those flags would take a different path than the original response.
        """
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        payload = dict(entry["payload"])
        message = dict(entry["message"])
        if isinstance(message.get("tool_calls"), list):
            message["tool_calls"] = [dict(call) for call in message["tool_calls"]]
        # Flagged so the caller can tell a replay from a fresh round trip.
        payload["replayed"] = True
        return {"payload": payload, "message": message}

    def put(self, key: str, completion: dict[str, Any]) -> None:
        """Remember one completion, evicting the oldest when full."""
        message = completion.get("message")
        payload = completion.get("payload")
        if not isinstance(message, dict) or not isinstance(payload, dict):
            return
        if message.get("interrupted") or payload.get("finish_reason") == "aborted":
            # A stopped response is not an answer; the user gets a fresh attempt.
            return
        kept = {
            name: payload[name]
            for name in ("usage", "finish_reason", "truncated")
            if name in payload
        }
        self._entries[key] = {
            "payload": kept,
            "message": {name: value for name, value in message.items() if name != "interrupted"},
        }
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry, for a new conversation or a model switch."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses, "max": self._max_entries}


def _canonical(value: Any) -> str:
    """A stable serialisation, so key order never causes a false miss."""
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)
