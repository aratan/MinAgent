"""Persistent knowledge memory backed by SQLite.

MinAgent records what worked - verified procedures, facts, and experiences - in a
local database. Before a request the app recalls the most relevant entries as a
short hint block, and the model can search, save, or reinforce memories with
tools. A later session therefore starts from accumulated knowledge instead of
from zero.

The database is opened per operation and every call runs in a worker thread, so
the event loop is never blocked and no connection crosses threads.
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from .embeddings import Embedder, cosine
from .errors import AgentError
from .reflection import ReviewAction

MAX_TITLE_CHARS = 160
MAX_CONTENT_CHARS = 8000
MAX_SOURCE_CHARS = 200
MAX_TAGS = 12
MAX_TAG_CHARS = 40
MAX_MEMORIES = 2000
DEFAULT_RECALL_LIMIT = 5
MAX_RECALL_LIMIT = 8
MAX_HINT_MEMORIES = 4
MAX_HINT_CHARS = 1400
MIN_HINT_CONFIDENCE = 0.35
DEFAULT_CONFIDENCE = 0.5
# A memory is trusted to answer on its own only when it is both strong on
# confidence and a close match for the request, so a stale entry never wins.
DIRECT_ANSWER_MIN_CONFIDENCE = 0.75
DIRECT_ANSWER_MIN_RATIO = 0.6
DIRECT_ANSWER_MIN_TOKENS = 2
SUCCESS_CONFIDENCE_STEP = 0.15
FAILURE_CONFIDENCE_STEP = 0.2
REINFORCE_CONFIDENCE_STEP = 0.05

# ``hypothesis`` is what a session reflection leaves behind: something the
# evidence points at but that has not happened yet. It is listed here rather
# than folded into the others because ``_clean_kind`` turns anything unknown
# into ``procedure``, which would file every conjecture as a method that works.
ALLOWED_KINDS = (
    "procedure",
    "solution",
    "fact",
    "preference",
    "experience",
    "hypothesis",
)

# The two sources that make a reviewable log visible. What the turn wrote by
# itself is still up for judgement; what the review kept is not, and is never
# offered to a later cull.
AUTO_CAPTURE_SOURCE = "auto-captured after a successful turn"
REVIEWED_SOURCE = "kept by the memory review"
# A review is a second opinion on something that already succeeded once, so it
# moves confidence less than recording the same memory again would.
REVIEW_REINFORCE_STEP = 0.1
# Two memories that say the same thing are one memory said twice. The share has
# to be high, because merging two genuinely different lessons loses one of them
# and there is no way to get it back. Calibrated against paraphrases of the same
# memory and against near neighbours on the same subject: the paraphrases score
# 0.55 to 1.0 on overlap, the neighbours 0.0, so the two populations do not meet.
DUPLICATE_OVERLAP = 0.55
# A paraphrase can use mostly different words, so overlap alone misses some. The
# symmetric measure catches those, at a lower bar because it already insists that
# neither side has much the other lacks.
DUPLICATE_SIMILARITY = 0.35
# Below this many shared distinctive tokens the overlap is vocabulary rather than
# meaning. Two is low on purpose and is the most a technical subject shares by
# accident: a version number and a library name.
MIN_SHARED_TOKENS = 2
# The log of what was done is never merged. Two turns can run the same tools
# and still be two separate things that happened.
DEDUPED_KINDS = ("procedure", "solution", "fact", "preference")
# The vector check catches the restatement that shares no distinctive word with
# what is already stored, which is exactly the case the rules above miss: two
# memories about the same fact, phrased from scratch. Calibrated with
# nomic-embed-text on this project's own store, in Spanish:
#
# * a paraphrase of a stored memory scores 0.89, two short ones 0.91, and a
#   restatement about the same subsystem 0.96;
# * two different memories on the same subject score 0.52, two on adjacent
#   subjects 0.65;
# * and the closest pair of genuinely different memories in the store - a
#   Himalaya command reference next to a Himalaya read-mail procedure - scores
#   0.79.
#
# 0.80 therefore sits above the worst real pair and below every measured
# duplicate, and a little below it is where the merging starts eating real
# memories. It is a narrow gap: it is one number and it is load-bearing, so
# raising the embedding model's own similarity is the way to widen it, not
# lowering this.
#
# It only ever adds a merge; it never removes one. A cosine below the threshold
# leaves the decision to the token rules, so a model that is unsure costs one
# duplicate memory and nothing else. That is also what happens on memories too
# short to tell apart: a one-line restatement of a one-line memory scores 0.69,
# the same 0.69 as a different fact about a different subject, and the check
# stays quiet rather than guessing between them. It pays off on memories with
# real content, which is what this store holds.
#
# The other known limit is the one the token rules have too: the same fact
# written in two languages. This model was trained on English text and reads a
# Spanish restatement of an English memory as a near neighbour only at 0.66, so
# that duplicate survives rather than being wrongly merged.
DUPLICATE_COSINE = 0.80
# The gate that decides which stored memories are worth one embedding request.
# Comparing every row on every save would be a request per memory in the store,
# so anything sharing no distinctive word with the new text is dropped before any
# request is made - and that costs nothing, because the measured negatives score
# zero shared tokens anyway. It is a filter, not a decision: the cosine above
# still has the final word.
MIN_CANDIDATE_TOKENS = 1

_QUERY_TOKEN = re.compile(r"[0-9A-Za-z_]{2,}")
_WORD = re.compile(r"[^a-z0-9]+")
# Unicode-aware, unlike ``_QUERY_TOKEN``: the store holds Spanish, and an ASCII
# pattern turns "video" into "deo", which quietly breaks any comparison that
# depends on whole words. It also keeps technical identifiers whole, so
# ".venv-compute/bin/python" is one token instead of four - splitting it left two
# memories about the same interpreter sharing almost nothing.
_DEDUP_TOKEN = re.compile(
    r"[^\W_]*[0-9][^\W_./+-]*[./+-][^\W]*|[.~/][\w./-]{3,}|[\w-]{4,}",
    re.UNICODE,
)

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    title_key TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    uses INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_used_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS memories_kind_title ON memories(kind, title_key);
CREATE INDEX IF NOT EXISTS memories_confidence ON memories(confidence DESC);
CREATE INDEX IF NOT EXISTS memories_updated ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS memories_last_used ON memories(last_used_at);
CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    outcome TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS outcomes_memory ON outcomes(memory_id);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    title, content, tags, content='memories', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
    VALUES ('delete', old.id, old.title, old.content, old.tags);
    INSERT INTO memories_fts(rowid, title, content, tags)
    VALUES (new.id, new.title, new.content, new.tags);
END;
"""


def _compact_whitespace(value: str) -> str:
    """Collapse any run of whitespace to a single space."""
    return " ".join(value.split())


def _now() -> str:
    """The current UTC time as a stable ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _title_key(title: str) -> str:
    """Normalized title used for de-duplication and reinforcement."""
    return _compact_whitespace(title).casefold()[:MAX_TITLE_CHARS]


def query_tokens(text: str) -> list[str]:
    """Word tokens worth searching for, in order and without duplicates."""
    tokens: list[str] = []
    for match in _QUERY_TOKEN.finditer(text or ""):
        token = match.group(0).casefold()
        if token and token not in tokens:
            tokens.append(token)
        if len(tokens) >= 12:
            break
    return tokens


def match_ratio(tokens: Sequence[str], memory: dict[str, Any]) -> float:
    """Share of the query tokens that a memory actually contains."""
    if not tokens:
        return 0.0
    haystack = " ".join(
        (
            memory.get("title") or "",
            memory.get("content") or "",
            memory.get("tags") or "",
        )
    ).casefold()
    hits = sum(1 for token in tokens if token in haystack)
    return hits / len(tokens)


def content_tokens(text: str) -> set[str]:
    """Whole words worth comparing two memories by."""
    return {match.group(0).casefold() for match in _DEDUP_TOKEN.finditer(text or "")}


def similarity(left: str, right: str) -> float:
    """How much two texts say the same thing, symmetrically, from 0 to 1.

    Jaccard rather than containment: containment would call a short memory a
    duplicate of any long one that happens to contain its words, and the short
    memory is usually the precise new one. This only calls it a duplicate when
    neither side has much the other lacks.
    """
    first, second = content_tokens(left), content_tokens(right)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def overlap(left: str, right: str) -> float:
    """How much of the smaller text the larger one already says, from 0 to 1.

    This is the measure that survives a paraphrase, where the same fact comes
    back with a different sentence around it: the two texts share the subject's
    words without sharing their length.
    """
    first, second = content_tokens(left), content_tokens(right)
    if not first or not second:
        return 0.0
    return len(first & second) / min(len(first), len(second))


def says_the_same_thing(left: str, right: str) -> bool:
    """Whether two memories are one memory said twice.

    The shared-token floor is checked first because it is what actually
    separates the two populations: a paraphrase shares the subject's distinctive
    words, a near neighbour on the same subject shares none. The two ratios then
    only have to agree once that has been established.
    """
    first, second = content_tokens(left), content_tokens(right)
    if len(first & second) < MIN_SHARED_TOKENS:
        return False
    return (
        overlap(left, right) >= DUPLICATE_OVERLAP
        or similarity(left, right) >= DUPLICATE_SIMILARITY
    )


def _clean_text(value: Any, limit: int, field: str) -> str:
    """Trim one model-supplied string and enforce its size limit."""
    if not isinstance(value, str):
        raise AgentError(f"{field} must be a string.")
    cleaned = value.strip()
    if not cleaned:
        raise AgentError(f"{field} cannot be empty.")
    if len(cleaned) > limit:
        raise AgentError(f"{field} exceeds {limit} characters.")
    return cleaned


def _clean_kind(value: Any) -> str:
    """Accept one of the known kinds; anything else becomes ``procedure``."""
    if value is None or value == "":
        return "procedure"
    if not isinstance(value, str):
        raise AgentError("kind must be a string.")
    cleaned = _WORD.sub("-", value.strip().lower()).strip("-")
    return cleaned if cleaned in ALLOWED_KINDS else "procedure"


def _clean_tags(value: Any) -> str:
    """Flatten a tag list into one searchable, bounded string."""
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        items: Sequence[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise AgentError("tags must be a string or an array of strings.")
    tags: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise AgentError("each tag must be a string.")
        tag = _compact_whitespace(item)[:MAX_TAG_CHARS].strip()
        if tag and tag not in tags:
            tags.append(tag)
        if len(tags) >= MAX_TAGS:
            break
    return " ".join(tags)


class MemoryStore:
    """A local SQLite store of learned procedures, facts, and experiences."""

    def __init__(self, path: str, *, embed_model: str = "") -> None:
        self.path = path
        self.fts_enabled = False
        # Empty model name means no embedding model is configured, and the store
        # compares words only - the behaviour that works with nothing installed.
        self.embedder: Embedder | None = Embedder(embed_model) if embed_model.strip() else None

    # ---------------------------------------------------------- connections

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a fresh connection for one operation, committing on success."""
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            # NORMAL is durable with WAL and much faster than the FULL default.
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA temp_store=MEMORY")
            yield connection
            connection.commit()
        finally:
            connection.close()

    # ---------------------------------------------------------- lifecycle

    async def initialize(self) -> None:
        """Create the schema, falling back to plain columns when FTS5 is absent."""
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as error:
                raise AgentError(f"Could not create the memory directory: {error}") from error
        try:
            with self._connect() as connection:
                connection.executescript(_BASE_SCHEMA)
                try:
                    connection.executescript(_FTS_SCHEMA)
                    self.fts_enabled = True
                except sqlite3.OperationalError:
                    self.fts_enabled = False
            self._maintenance_sync()
        except sqlite3.Error as error:
            raise AgentError(f"Could not open the memory database: {error}") from error

    def _maintenance_sync(self) -> None:
        """Refresh planner statistics and reclaim space when the file is bloated."""
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            connection.execute("PRAGMA optimize")
            page_count = connection.execute("PRAGMA page_count").fetchone()[0]
            free_pages = connection.execute("PRAGMA freelist_count").fetchone()[0]
            if page_count and free_pages / page_count > 0.2:
                connection.execute("VACUUM")
        except sqlite3.Error:
            pass
        finally:
            connection.close()

    # ------------------------------------------------------------- search

    async def recall(self, query: str, limit: int = DEFAULT_RECALL_LIMIT) -> list[dict[str, Any]]:
        """Search memories and mark the returned entries as used."""
        memories = await asyncio.to_thread(self._search_sync, query, limit)
        if memories:
            await asyncio.to_thread(self._mark_used_sync, [memory["id"] for memory in memories])
        return memories

    async def hints(self, text: str, limit: int = MAX_HINT_MEMORIES) -> list[dict[str, Any]]:
        """Search for prompt hints without touching usage counters."""
        memories = await asyncio.to_thread(self._search_sync, text, limit)
        return [memory for memory in memories if memory["confidence"] >= MIN_HINT_CONFIDENCE]

    def _search_sync(self, query: str, limit: int) -> list[dict[str, Any]]:
        tokens = query_tokens(query)
        if not tokens:
            return []
        bounded = max(1, min(int(limit), MAX_RECALL_LIMIT))
        with self._connect() as connection:
            if self.fts_enabled:
                match = " OR ".join(f'"{token}"*' for token in tokens)
                try:
                    rows = connection.execute(
                        "SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts f "
                        "JOIN memories m ON m.id = f.rowid "
                        "WHERE memories_fts MATCH ? ORDER BY rank, m.confidence DESC LIMIT ?",
                        (match, bounded),
                    ).fetchall()
                    return [dict(row) for row in rows]
                except sqlite3.OperationalError:
                    pass
            clauses = " OR ".join("(title LIKE ? OR content LIKE ? OR tags LIKE ?)" for _ in tokens)
            parameters: list[str] = []
            for token in tokens:
                like = f"%{token}%"
                parameters.extend((like, like, like))
            parameters.append(bounded)  # type: ignore[arg-type]
            rows = connection.execute(
                f"SELECT m.*, 0 AS rank FROM memories m WHERE {clauses} "
                "ORDER BY confidence DESC, success_count DESC, updated_at DESC LIMIT ?",
                parameters,
            ).fetchall()
            return [dict(row) for row in rows]

    async def lookup(
        self,
        query: str,
        min_confidence: float = DIRECT_ANSWER_MIN_CONFIDENCE,
        min_ratio: float = DIRECT_ANSWER_MIN_RATIO,
    ) -> dict[str, Any] | None:
        """Return one strongly matching memory, or ``None`` when nothing is close enough.

        This is the cheap question the app asks before spending a model call:
        does MinAgent already know this well enough to answer without the endpoint?
        """
        memory = await asyncio.to_thread(self._lookup_sync, query, min_confidence, min_ratio)
        if memory is not None:
            await asyncio.to_thread(self._mark_used_sync, [memory["id"]])
        return memory

    def _lookup_sync(
        self, query: str, min_confidence: float, min_ratio: float
    ) -> dict[str, Any] | None:
        tokens = query_tokens(query)
        if len(tokens) < DIRECT_ANSWER_MIN_TOKENS:
            return None
        best: dict[str, Any] | None = None
        best_ratio = 0.0
        for memory in self._search_sync(query, MAX_RECALL_LIMIT):
            if memory["confidence"] < min_confidence:
                continue
            ratio = match_ratio(tokens, memory)
            if ratio > best_ratio:
                best, best_ratio = memory, ratio
        if best is None or best_ratio < min_ratio:
            return None
        result = dict(best)
        result["match_ratio"] = best_ratio
        return result

    def _mark_used_sync(self, identifiers: list[int]) -> None:
        if not identifiers:
            return
        placeholders = ",".join("?" for _ in identifiers)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE memories SET uses = uses + 1, last_used_at = ? WHERE id IN ({placeholders})",
                [_now(), *identifiers],
            )

    # ----------------------------------------------------------- mutation

    async def remember(
        self,
        kind: Any,
        title: Any,
        content: Any,
        tags: Any = None,
        source: Any = None,
    ) -> dict[str, Any]:
        """Store a memory, or reinforce the existing one with the same title."""
        cleaned_kind = _clean_kind(kind)
        cleaned_title = _clean_text(title, MAX_TITLE_CHARS, "title")
        cleaned_content = _clean_text(content, MAX_CONTENT_CHARS, "content")
        cleaned_tags = _clean_tags(tags)
        cleaned_source = _compact_whitespace(source)[:MAX_SOURCE_CHARS] if isinstance(source, str) else ""
        # Asked before the write, not inside it: the comparison is a request to
        # another model and the database work runs in a thread that has no
        # business waiting on one.
        by_meaning = await self._duplicate_by_meaning(
            cleaned_kind, cleaned_title, cleaned_content, cleaned_tags
        )
        return await asyncio.to_thread(
            self._remember_sync,
            cleaned_kind,
            cleaned_title,
            cleaned_content,
            cleaned_tags,
            cleaned_source,
            by_meaning,
        )

    async def _duplicate_by_meaning(
        self, kind: str, title: str, content: str, tags: str
    ) -> int | None:
        """The id of a memory this one repeats by meaning rather than by wording.

        Two passes, because the request cannot be made from inside the database
        thread: the cheap lexical scan runs first and only what it leaves
        standing is sent to the embedding model. Every way this can fail - no
        model configured, no server, a timeout, a vector of the wrong size -
        returns ``None``, and the caller then merges on the token rules alone.
        A comparison that cannot be had costs the store a duplicate memory, and
        nothing worse.
        """
        if self.embedder is None or kind not in DEDUPED_KINDS:
            return None
        text = f"{title}\n{content}\n{tags}"
        candidates = await asyncio.to_thread(self._candidates_sync, kind, text)
        if not candidates:
            return None
        vectors = await self.embedder.embed([text, *(other for _, other in candidates)])
        if vectors is None or len(vectors) != len(candidates) + 1:
            return None
        best: tuple[float, int] | None = None
        for vector, (identifier, _) in zip(vectors[1:], candidates, strict=True):
            score = cosine(vectors[0], vector)
            if score >= DUPLICATE_COSINE and (best is None or score > best[0]):
                best = (score, identifier)
        return best[1] if best is not None else None

    def _candidates_sync(self, kind: str, text: str) -> list[tuple[int, str]]:
        """Stored memories worth one embedding request to compare against."""
        own = content_tokens(text)
        if not own:
            return []
        placeholders = ",".join("?" for _ in DEDUPED_KINDS)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT id, title, content, tags FROM memories WHERE kind IN ({placeholders})",
                DEDUPED_KINDS,
            ).fetchall()
        candidates: list[tuple[int, str]] = []
        for row in rows:
            other = f"{row['title']}\n{row['content']}\n{row['tags']}"
            if content_tokens(other) & own:
                candidates.append((row["id"], other))
        return candidates

    def _remember_sync(
        self, kind: str, title: str, content: str, tags: str, source: str, by_meaning: int | None = None
    ) -> dict[str, Any]:
        now = _now()
        key = _title_key(title)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM memories WHERE kind = ? AND title_key = ?", (kind, key)
            ).fetchone()
            if existing is not None:
                confidence = min(1.0, existing["confidence"] + REINFORCE_CONFIDENCE_STEP)
                connection.execute(
                    "UPDATE memories SET content = ?, tags = ?, source = ?, "
                    "success_count = success_count + 1, uses = uses + 1, confidence = ?, "
                    "updated_at = ?, last_used_at = ? WHERE id = ?",
                    (content, tags, source, confidence, now, now, existing["id"]),
                )
                connection.execute(
                    "INSERT INTO outcomes (memory_id, outcome, note, created_at) VALUES (?, ?, ?, ?)",
                    (existing["id"], "success", "reinforced by remember", now),
                )
                return {
                    "status": "reinforced",
                    "id": existing["id"],
                    "kind": kind,
                    "title": title,
                    "confidence": confidence,
                    "success_count": existing["success_count"] + 1,
                }
            duplicate = self._find_duplicate_sync(connection, kind, title, content, tags)
            if duplicate is None and by_meaning is not None:
                # The lexical scan is the one that decides when both of them can
                # answer, so it goes first. This is what a restatement with no
                # shared word falls through to, and it is re-checked here because
                # the memory it named may have been culled between the two passes.
                row = connection.execute(
                    "SELECT * FROM memories WHERE id = ?", (by_meaning,)
                ).fetchone()
                if row is not None and row["kind"] in DEDUPED_KINDS:
                    duplicate = dict(row)
            if duplicate is not None:
                # The title is not the identity, the content is. The same lesson
                # learned twice arrives worded differently - the model invents
                # its own title each time - and storing it twice fills the hint
                # block with one answer in two voices and makes both look
                # weaker. The newer wording is kept because it is the one
                # written with everything known so far.
                confidence = min(1.0, duplicate["confidence"] + REINFORCE_CONFIDENCE_STEP)
                connection.execute(
                    "UPDATE memories SET title = ?, content = ?, tags = ?, source = ?, "
                    "success_count = success_count + 1, uses = uses + 1, confidence = ?, "
                    "updated_at = ?, last_used_at = ? WHERE id = ?",
                    (
                        title,
                        content,
                        tags,
                        source,
                        confidence,
                        now,
                        now,
                        duplicate["id"],
                    ),
                )
                connection.execute(
                    "INSERT INTO outcomes (memory_id, outcome, note, created_at) VALUES (?, ?, ?, ?)",
                    (duplicate["id"], "success", "merged a memory that said the same thing", now),
                )
                return {
                    "status": "duplicate",
                    "id": duplicate["id"],
                    "kind": duplicate["kind"],
                    "title": title,
                    "confidence": confidence,
                    "success_count": duplicate["success_count"] + 1,
                }
            cursor = connection.execute(
                "INSERT INTO memories "
                "(kind, title, title_key, content, tags, source, confidence, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (kind, title, key, content, tags, source, DEFAULT_CONFIDENCE, now, now),
            )
            self._prune(connection)
            return {
                "status": "created",
                "id": cursor.lastrowid,
                "kind": kind,
                "title": title,
                "confidence": DEFAULT_CONFIDENCE,
                "success_count": 0,
            }

    def _find_duplicate_sync(
        self, connection: sqlite3.Connection, kind: str, title: str, content: str, tags: str
    ) -> dict[str, Any] | None:
        """The memory this one repeats, if any.

        Only the knowledge kinds are compared: the log of what was done is
        deliberately allowed to hold many similar entries, because two turns
        can look alike and still be two things that happened.
        """
        if kind not in DEDUPED_KINDS:
            return None
        text = f"{title}\n{content}\n{tags}"
        if len(content_tokens(text)) < MIN_SHARED_TOKENS:
            return None
        placeholders = ",".join("?" for _ in DEDUPED_KINDS)
        rows = connection.execute(
            f"SELECT * FROM memories WHERE kind IN ({placeholders})",
            DEDUPED_KINDS,
        ).fetchall()
        best: dict[str, Any] | None = None
        best_score = 0.0
        for row in rows:
            candidate = dict(row)
            other = f"{candidate['title']}\n{candidate['content']}\n{candidate['tags']}"
            if not says_the_same_thing(text, other):
                continue
            score = overlap(text, other)
            if score >= best_score:
                best, best_score = candidate, score
        return best

    def _prune(self, connection: sqlite3.Connection) -> None:
        """Drop the weakest memories once the cap is exceeded."""
        total = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        excess = total - MAX_MEMORIES
        if excess <= 0:
            return
        connection.execute(
            "DELETE FROM memories WHERE id IN ("
            "SELECT id FROM memories ORDER BY confidence ASC, success_count ASC, "
            "COALESCE(last_used_at, updated_at) ASC LIMIT ?)",
            (excess,),
        )

    async def record_outcome(self, memory_id: Any, success: Any, note: Any = None) -> dict[str, Any]:
        """Reinforce or degrade one memory after it was reused."""
        if isinstance(memory_id, bool) or not isinstance(memory_id, int):
            raise AgentError("id must be the integer of a stored memory.")
        if not isinstance(success, bool):
            raise AgentError("success must be true or false.")
        cleaned_note = _compact_whitespace(note)[:MAX_SOURCE_CHARS] if isinstance(note, str) else ""
        return await asyncio.to_thread(self._record_outcome_sync, memory_id, success, cleaned_note)

    def _record_outcome_sync(self, memory_id: int, success: bool, note: str) -> dict[str, Any]:
        now = _now()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                raise AgentError(f"There is no memory with id {memory_id}.")
            step = SUCCESS_CONFIDENCE_STEP if success else -FAILURE_CONFIDENCE_STEP
            confidence = max(0.0, min(1.0, row["confidence"] + step))
            success_count = row["success_count"] + (1 if success else 0)
            failure_count = row["failure_count"] + (0 if success else 1)
            connection.execute(
                "UPDATE memories SET success_count = ?, failure_count = ?, confidence = ?, "
                "uses = uses + 1, updated_at = ?, last_used_at = ? WHERE id = ?",
                (success_count, failure_count, confidence, now, now, memory_id),
            )
            connection.execute(
                "INSERT INTO outcomes (memory_id, outcome, note, created_at) VALUES (?, ?, ?, ?)",
                (memory_id, "success" if success else "failure", note, now),
            )
            return {
                "id": memory_id,
                "title": row["title"],
                "success_count": success_count,
                "failure_count": failure_count,
                "confidence": confidence,
            }

    async def forget(self, memory_id: Any) -> bool:
        """Delete one memory and its outcome history."""
        if isinstance(memory_id, bool) or not isinstance(memory_id, int):
            raise AgentError("id must be the integer of a stored memory.")
        return await asyncio.to_thread(self._forget_sync, memory_id)

    def _forget_sync(self, memory_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    # -------------------------------------------------------------- reads

    async def statistics(self) -> dict[str, Any]:
        """Counts used by the ``/memory`` panel."""
        return await asyncio.to_thread(self._statistics_sync)

    def _statistics_sync(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(success_count), 0) AS successes, "
                "COALESCE(SUM(failure_count), 0) AS failures, COALESCE(SUM(uses), 0) AS uses, "
                "COALESCE(AVG(confidence), 0) AS confidence FROM memories"
            ).fetchone()
            return {
                "total": row["total"],
                "successes": row["successes"],
                "failures": row["failures"],
                "uses": row["uses"],
                "confidence": row["confidence"],
                "fts": self.fts_enabled,
            }

    async def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """The strongest memories, for display."""
        return await asyncio.to_thread(self._recent_sync, limit)

    async def reviewable(self, limit: int = 40) -> list[dict[str, Any]]:
        """Auto-captured entries still waiting to be judged, oldest first.

        Oldest first because the review is a cull of a backlog: the entries
        that have been waiting longest are the ones that survived a turn
        without anyone deciding they mattered.
        """
        return await asyncio.to_thread(self._reviewable_sync, limit)

    def _reviewable_sync(self, limit: int) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 100))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories WHERE source = ? ORDER BY created_at ASC, id ASC LIMIT ?",
                (AUTO_CAPTURE_SOURCE, bounded),
            ).fetchall()
            return [dict(row) for row in rows]

    async def apply_review(self, actions: Sequence[ReviewAction]) -> int:
        """Reinforce or delete each entry the review judged, and count the changes.

        A kept entry is reinforced *and* marked as judged in the same
        statement, because a cull that left the survivors still looking
        unreviewed would offer them again on the next pass and the log would
        grow without end instead of shrinking.
        """
        return await asyncio.to_thread(self._apply_review_sync, list(actions))

    def _apply_review_sync(self, actions: list[ReviewAction]) -> int:
        if not actions:
            return 0
        now = _now()
        changed = 0
        with self._connect() as connection:
            for action in actions:
                if action.keep:
                    note = action.note or "kept by the memory review"
                    row = connection.execute(
                        "SELECT confidence FROM memories WHERE id = ? AND source = ?",
                        (action.entry_id, AUTO_CAPTURE_SOURCE),
                    ).fetchone()
                    if row is None:
                        continue
                    connection.execute(
                        "UPDATE memories SET success_count = success_count + 1, uses = uses + 1, "
                        "confidence = ?, source = ?, updated_at = ?, last_used_at = ? WHERE id = ?",
                        (
                            min(1.0, row["confidence"] + REVIEW_REINFORCE_STEP),
                            REVIEWED_SOURCE,
                            now,
                            now,
                            action.entry_id,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO outcomes (memory_id, outcome, note, created_at) VALUES (?, ?, ?, ?)",
                        (action.entry_id, "success", note, now),
                    )
                    changed += 1
                else:
                    cursor = connection.execute(
                        "DELETE FROM memories WHERE id = ? AND source = ?",
                        (action.entry_id, AUTO_CAPTURE_SOURCE),
                    )
                    changed += cursor.rowcount if cursor.rowcount > 0 else 0
        return changed

    def _recent_sync(self, limit: int) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 50))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY confidence DESC, success_count DESC, updated_at DESC LIMIT ?",
                (bounded,),
            ).fetchall()
            return [dict(row) for row in rows]


# ---------------------------------------------------------------- formatting


def _stats_label(memory: dict[str, Any]) -> str:
    return f"{memory['success_count']}✓/{memory['failure_count']}✗"


def _excerpt(value: str, limit: int) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: limit - 1].rstrip()}…"


def format_direct_answer(memory: dict[str, Any]) -> str:
    """Render a memory reused verbatim, without a model round-trip."""
    return memory["content"]


def format_recall(memories: Sequence[dict[str, Any]]) -> str:
    """Render recall results for the tool result channel."""
    if not memories:
        return (
            "No stored memory matches that query yet. After this task works, save the "
            "procedure with remember so a later session can reuse it."
        )
    lines = ["Stored knowledge that may apply (verify before relying on it):"]
    for memory in memories:
        lines.append(
            f"#{memory['id']} [{memory['kind']}] {memory['title']} "
            f"({_stats_label(memory)}, confidence {memory['confidence']:.2f})"
        )
        lines.append(f"    {_excerpt(memory['content'], 600)}")
    return "\n".join(lines)


def format_memory_hints(memories: Sequence[dict[str, Any]], max_chars: int = MAX_HINT_CHARS) -> str:
    """Render a bounded hint block for the system prompt."""
    if not memories:
        return ""
    header = (
        "Relevant knowledge recorded in earlier sessions. It may be stale; verify "
        "before relying on it, and reinforce or correct it with record_outcome."
    )
    lines = [header]
    used = len(header)
    for memory in memories:
        summary = _excerpt(memory["content"], 220)
        line = f"- [{memory['kind']}] {memory['title']} ({_stats_label(memory)}): {summary}"
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines) if len(lines) > 1 else ""


def format_memory_stats(statistics: dict[str, Any], memories: Sequence[dict[str, Any]]) -> str:
    """Render the ``/memory`` summary."""
    lines = [
        f"Memory: {statistics['total']} entries · {statistics['successes']} successes · "
        f"{statistics['failures']} failures · {statistics['uses']} reuses · "
        f"average confidence {statistics['confidence']:.2f} · "
        f"{'FTS5 search' if statistics['fts'] else 'LIKE fallback'}"
    ]
    for memory in memories:
        lines.append(
            f"#{memory['id']} [{memory['kind']}] {memory['title']} ({_stats_label(memory)})"
        )
    if not memories:
        lines.append("Nothing learned yet. Procedures are saved as the model verifies them.")
    return "\n".join(lines)


# -------------------------------------------------------------------- tools


def create_memory_tools() -> list[dict[str, Any]]:
    """Tool definitions exposed to the model when memory is enabled."""
    return [
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": (
                    "Search Ara's persistent memory for a procedure, fact, or past experience that "
                    "already answers this task. Call it before a non-trivial task to reuse verified "
                    "knowledge instead of rediscovering it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What you are trying to do or find out"},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_RECALL_LIMIT,
                            "description": f"Maximum results; defaults to {DEFAULT_RECALL_LIMIT}",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Save what worked so a later session starts from it. Use it after a verified success: "
                    "record the concrete procedure or conclusion, not a restatement of the request."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short, reuse-friendly title"},
                        "content": {"type": "string", "description": "The procedure or knowledge, with the steps that worked"},
                        "kind": {
                            "type": "string",
                            "enum": list(ALLOWED_KINDS),
                            "description": "Defaults to procedure",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional keywords that help future recall",
                        },
                        "source": {"type": "string", "description": "Optional note about where this came from"},
                    },
                    "required": ["title", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_outcome",
                "description": (
                    "Report whether a recalled memory actually worked, so its confidence reflects reality. "
                    "A success reinforces it; a failure degrades it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "The memory id, shown as #id in recall results"},
                        "success": {"type": "boolean", "description": "true if it worked, false if it did not"},
                        "note": {"type": "string", "description": "Optional short note about the outcome"},
                    },
                    "required": ["id", "success"],
                },
            },
        },
    ]


def format_remember_result(result: dict[str, Any]) -> str:
    """Render the confirmation for ``remember``."""
    if result.get("status") == "duplicate":
        # Saying so is the point: the model should learn that the store already
        # had this, rather than believing it taught the agent something new.
        return (
            f"Memory #{result['id']} [{result['kind']}] already said this, so it was merged into that one "
            f"instead of stored twice ({result['success_count']} successes, confidence "
            f"{result['confidence']:.2f}). Recall it rather than saving it again."
        )
    if result.get("status") == "reinforced":
        return (
            f"Reinforced memory #{result['id']} [{result['kind']}] \"{result['title']}\" "
            f"({result['success_count']} successes, confidence {result['confidence']:.2f})."
        )
    return (
        f"Saved memory #{result['id']} [{result['kind']}] \"{result['title']}\" "
        f"(confidence {result['confidence']:.2f}). Reuse it with recall and report how it went with "
        "record_outcome."
    )


def format_outcome_result(result: dict[str, Any]) -> str:
    """Render the confirmation for ``record_outcome``."""
    return (
        f"Memory #{result['id']} \"{result['title']}\" now has {result['success_count']} successes and "
        f"{result['failure_count']} failures (confidence {result['confidence']:.2f})."
    )
