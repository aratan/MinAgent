"""Project file selection for ``/init``.

Picks the files most likely to describe a project's architecture and entry
points, in priority order, within a character budget. Secret-looking files are
never selected, and every excerpt is redacted before the model sees it.
"""

from __future__ import annotations

import os
import re
import stat as stat_module
from typing import Any, Awaitable, Callable

from .errors import AgentError
from .jsutil import decode_utf8, locale_key
from .secrets import redact_likely_secrets

MAX_FILES = 24
MAX_FILE_CHARS = 16_000
MAX_TOTAL_CHARS = 64_000

PRIORITY_FILES = {
    "readme.md": 0, "readme": 0,
    "package.json": 1, "pyproject.toml": 1, "requirements.txt": 1, "cargo.toml": 1,
    "go.mod": 1, "pom.xml": 1, "build.gradle": 1, "build.gradle.kts": 1,
    "composer.json": 1, "gemfile": 1, "makefile": 1, "cmakelists.txt": 1,
    "tsconfig.json": 2, "dockerfile": 2,
    "vite.config.js": 2, "vite.config.mjs": 2, "vite.config.ts": 2,
    "next.config.js": 2, "next.config.mjs": 2, "next.config.ts": 2,
}

SOURCE_EXTENSIONS = re.compile(
    r"\.(?:c|cc|cpp|cs|go|h|hpp|java|js|jsx|mjs|cjs|php|py|rb|rs|sh|sql|swift|ts|tsx|vue|svelte|html|css)$",
    re.IGNORECASE,
)
TEXT_EXTENSIONS = re.compile(
    r"\.(?:md|txt|json|toml|ya?ml|xml|gradle|properties|c|cc|cpp|cs|go|h|hpp|java|js|jsx|mjs|cjs"
    r"|php|py|rb|rs|sh|sql|swift|ts|tsx|vue|svelte|html|css)$",
    re.IGNORECASE,
)
PROJECT_DIRECTORIES = {
    "src", "app", "lib", "cmd", "server", "client", "packages", "apps",
    "pages", "api", "web", "backend", "frontend", "tests", "test", "__tests__",
}
_SECRET_NAME = re.compile(r"(?:secret|credential|private[-_.]?key)", re.IGNORECASE)
_TEST_NAME = re.compile(r"(?:^|[._-])(?:test|spec)(?:[._-]|$)", re.IGNORECASE)
_ENTRY_POINT = re.compile(r"^(?:index|main|app|server|cli|lib)\.[^.]+$", re.IGNORECASE)
_CONFIG_FILE = re.compile(
    r"^(?:vite|next|webpack|rollup|eslint|prettier|postcss)\.config\.[^.]+$", re.IGNORECASE
)
_NON_ALNUM = re.compile(r"[^a-z0-9]")


def init_candidate_score(name: str, nested_depth: int, project_name: str = "") -> int | None:
    """Score a candidate file for ``/init``, or return ``None`` to skip it.

    Lower scores win. Entry points and manifests come first, then source files,
    then tests, and finally unfamiliar files that are still better than nothing.
    """
    lower_name = name.lower()
    if lower_name == "agents.md" or lower_name.startswith(".env") or lower_name.endswith(".lock"):
        return None
    if _SECRET_NAME.search(name):
        return None
    priority = PRIORITY_FILES.get(lower_name)
    if priority is not None:
        return priority + nested_depth * 3
    if not TEXT_EXTENSIONS.search(name):
        return None
    if not SOURCE_EXTENSIONS.search(name):
        return None
    if project_name:
        stem = os.path.splitext(lower_name)[0]
        if _NON_ALNUM.sub("", stem) == project_name:
            return 4 + nested_depth * 3
    if _TEST_NAME.search(name):
        return 11 + nested_depth * 3
    if _ENTRY_POINT.search(name):
        return 5 + nested_depth * 3
    if nested_depth <= 1 and _CONFIG_FILE.search(name):
        return 8
    # An unfamiliar entry point is still more useful than an inventory alone.
    return 12 + nested_depth * 3 if SOURCE_EXTENSIONS.search(name) else None


def _scandir_sorted(path: str) -> list[os.DirEntry]:
    try:
        with os.scandir(path) as scan:
            return sorted(scan, key=lambda entry: locale_key(entry.name))
    except OSError:
        return []


async def collect_project_essentials(
    root_directory: str,
    read_workspace_raw: Callable[[str], Awaitable[bytes]],
    max_total_chars: int = MAX_TOTAL_CHARS,
) -> dict[str, Any]:
    """Select and read the project files that best describe this codebase."""
    if not callable(read_workspace_raw):
        raise AgentError("A workspace file reader is required for /init.")
    if not isinstance(max_total_chars, int) or isinstance(max_total_chars, bool):
        raise AgentError("/init requires a whole-number project file budget.")
    if max_total_chars < 512:
        raise AgentError(
            "OPENAI_CONTEXT_WINDOW leaves too little room for /init; increase it and try again."
        )
    total_limit = min(MAX_TOTAL_CHARS, max_total_chars)
    project_name = _NON_ALNUM.sub("", os.path.basename(root_directory).lower())
    candidates: dict[str, int] = {}

    def add_candidate(path: str, name: str, depth: int) -> None:
        relative_path = os.path.relpath(path, root_directory).replace(os.sep, "/")
        score = init_candidate_score(name, depth, project_name)
        if score is None:
            return
        candidates[relative_path] = min(candidates.get(relative_path, float("inf")), score)

    def mode_of(entry: os.DirEntry) -> int:
        try:
            return entry.stat(follow_symlinks=False).st_mode
        except OSError:
            return -1

    for entry in _scandir_sorted(root_directory):
        mode = mode_of(entry)
        if mode == -1 or stat_module.S_ISLNK(mode):
            continue
        path = os.path.join(root_directory, entry.name)
        if stat_module.S_ISREG(mode):
            add_candidate(path, entry.name, 0)
        if not stat_module.S_ISDIR(mode) or entry.name.lower() not in PROJECT_DIRECTORIES:
            continue
        for child in _scandir_sorted(path):
            child_mode = mode_of(child)
            if child_mode == -1 or stat_module.S_ISLNK(child_mode):
                continue
            child_path = os.path.join(path, child.name)
            if stat_module.S_ISREG(child_mode):
                add_candidate(child_path, child.name, 1)
            if stat_module.S_ISDIR(child_mode) and entry.name.lower() in ("packages", "apps", "cmd"):
                for nested in _scandir_sorted(child_path):
                    nested_mode = mode_of(nested)
                    if nested_mode == -1 or stat_module.S_ISLNK(nested_mode):
                        continue
                    if stat_module.S_ISREG(nested_mode):
                        add_candidate(os.path.join(child_path, nested.name), nested.name, 2)

    selected = sorted(candidates.items(), key=lambda item: (item[1], locale_key(item[0])))[:MAX_FILES]
    file_excerpt_limit = min(MAX_FILE_CHARS, max(512, total_limit // max(1, min(len(selected), 8))))
    files: list[dict[str, Any]] = []
    total_chars = 0
    for relative_path, _score in selected:
        if total_chars >= total_limit:
            break
        try:
            data = await read_workspace_raw(relative_path)
            full_content = decode_utf8(data)
            if "\0" in full_content:
                continue
            safe_content = redact_likely_secrets(full_content)
            remaining = total_limit - total_chars
            excerpt = safe_content[: min(file_excerpt_limit, remaining)]
            files.append({"path": relative_path, "content": excerpt, "truncated": len(excerpt) < len(safe_content)})
            total_chars += len(excerpt)
        except Exception:
            # Unreadable, missing, or unsafe project files are skipped.
            continue
    return {"files": files, "candidate_count": len(candidates)}
