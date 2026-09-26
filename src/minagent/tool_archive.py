"""Off-window archive for oversized tool output.

A single command can emit far more text than the context window can hold. The
agent used to keep the head and the tail and drop the middle for good, which is
exactly the part that holds the evidence. This module stores that text, zlib
compressed, beside the session state and hands the model a short reference it
can read back on demand with ``recall_tool_output``.

Compression is what makes this affordable, not what makes it effective: the
saved bytes only matter because the full text now lives *outside* the window
instead of being discarded. Nothing here is lossy, so a recall returns exactly
the text the model would have seen had the result fit.
"""

from __future__ import annotations

import os
import re
import zlib

from .errors import AgentError

# Archive state lives beside the other per-installation agent files.
ARCHIVE_DIRECTORY = os.path.join(".minagent", "tool-outputs")

# A single result larger than this is not worth archiving; the caller keeps the
# old head-and-tail truncation instead.
MAX_ARCHIVED_CHARS = 8 * 1024 * 1024
# Total on-disk budget. Oldest entries are pruned once it is exceeded.
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
# A recall returns at most this many characters, so one call cannot refill the
# window the archive just freed.
MAX_ARCHIVE_RECALL_CHARS = 40_000
DEFAULT_ARCHIVE_RECALL_CHARS = 8_000
# Guard against a decompression bomb in a corrupted or tampered archive.
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024

# A reference becomes part of the conversation and is used to build a path, so it
# is restricted to characters that cannot escape the archive directory.
_VALID_REFERENCE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


class ToolArchive:
    """Store oversized tool results on disk and read them back by reference."""

    def __init__(self, root_directory: str) -> None:
        # An archive with no installation root would write relative to the
        # current directory. Staying disabled keeps an unconfigured session from
        # scattering files next to wherever it happens to be run; the caller
        # falls back to plain truncation.
        self._root = os.path.join(root_directory, ARCHIVE_DIRECTORY) if root_directory else ""

    def store(self, text: str) -> str | None:
        """Compress ``text`` into the archive and return its reference.

        Returns ``None`` when the text cannot or should not be archived, so the
        caller can fall back to plain truncation rather than failing a turn.
        """
        if not self._root or not text or len(text) > MAX_ARCHIVED_CHARS:
            return None
        payload = zlib.compress(text.encode("utf-8"), 9)
        try:
            os.makedirs(self._root, exist_ok=True)
            for _attempt in range(8):
                reference = os.urandom(6).hex()
                path = self._path(reference)
                try:
                    # O_EXCL: never overwrite another entry if references collide.
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                except FileExistsError:
                    continue
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                self._prune()
                return reference
        except OSError:
            # A full or read-only disk must not break the turn; the caller keeps
            # the truncated preview it would have had anyway.
            return None
        return None

    def read(self, reference: str, offset: int = 0, limit: int = DEFAULT_ARCHIVE_RECALL_CHARS) -> str:
        """Return a character slice of an archived result.

        The slice is reported with the full size so the model can tell whether
        it is looking at a fragment.
        """
        path = self._validated_path(reference)
        if not self._root:
            raise AgentError("No tool output archive is configured for this session.")
        try:
            with open(path, "rb") as handle:
                compressed = handle.read()
        except FileNotFoundError as error:
            raise AgentError(
                f"No archived tool output with id {reference!r}. It may have been pruned after a "
                "restart; rerun the command instead."
            ) from error
        except OSError as error:
            raise AgentError(f"Could not read the archived tool output {reference!r}: {error}") from error

        # Bounded decompression: a crafted archive must not be able to exhaust memory.
        decompressor = zlib.decompressobj()
        try:
            raw = decompressor.decompress(compressed, MAX_DECOMPRESSED_BYTES)
        except zlib.error as error:
            raise AgentError(f"The archived tool output {reference!r} is corrupt: {error}") from error

        text = raw.decode("utf-8", errors="replace")
        total = len(text)
        start = max(0, offset)
        stop = min(total, start + max(1, min(limit, MAX_ARCHIVE_RECALL_CHARS)))
        return f"[archived tool output {reference}: characters {start}-{stop} of {total}]\n" + text[start:stop]

    def stored_bytes(self) -> int:
        """Total size of the archive on disk."""
        return sum(entry[1] for entry in self._entries())

    def _path(self, reference: str) -> str:
        return os.path.join(self._root, f"{reference}.z")

    def _validated_path(self, reference: str) -> str:
        if not isinstance(reference, str) or not _VALID_REFERENCE.match(reference):
            raise AgentError("An archived tool output id may only contain letters, digits, dashes, and underscores.")
        return self._path(reference)

    def _entries(self) -> list[tuple[str, int, float]]:
        if not self._root:
            return []
        try:
            names = os.listdir(self._root)
        except OSError:
            return []
        entries: list[tuple[str, int, float]] = []
        for name in names:
            if not name.endswith(".z") or not _VALID_REFERENCE.match(name[:-2]):
                continue
            path = os.path.join(self._root, name)
            try:
                info = os.lstat(path)
            except OSError:
                continue
            if os.path.isfile(path) and not os.path.islink(path):
                entries.append((path, info.st_size, info.st_mtime))
        return entries

    def _prune(self) -> None:
        """Delete the oldest entries once the archive exceeds its budget."""
        entries = self._entries()
        total = sum(size for _path, size, _mtime in entries)
        if total <= MAX_ARCHIVE_BYTES:
            return
        for path, size, _mtime in sorted(entries, key=lambda entry: entry[2]):
            if total <= int(MAX_ARCHIVE_BYTES * 0.85):
                break
            try:
                os.unlink(path)
            except OSError:
                continue
            total -= size
