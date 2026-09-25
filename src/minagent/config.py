"""Configuration loading and validation.

Reads ``.env`` from the MinAgent project root, then validates every setting and
normalises the OpenAI-compatible endpoint to its ``/chat/completions`` URL.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import MutableMapping
from urllib.parse import urlsplit, urlunsplit

from .errors import AgentError, find_application_root

DEFAULT_CONTEXT_WINDOW = 262144
DEFAULT_ENDPOINT_TIMEOUT_SECONDS = 7 * 60
DEFAULT_EXTENSION_TIMEOUT_SECONDS = 7 * 60
DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = 2 * 60
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_POSITIVE_INTEGER = re.compile(r"^\d+$")
_DIRECTORY_ENTRY_LIMIT = re.compile(r"^-?\d+$")


def load_env_file(file_path: str, target: MutableMapping[str, str] | None = None) -> None:
    """Load ``NAME=value`` pairs into ``target`` without overriding existing keys.

    A missing file is ignored; a malformed line is a hard error.
    """
    env = os.environ if target is None else target
    try:
        with open(file_path, encoding="utf-8") as handle:
            contents = handle.read()
    except FileNotFoundError:
        return
    for index, line in enumerate(re.split(r"\r?\n", contents)):
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        match = _ASSIGNMENT.match(trimmed)
        if not match:
            raise AgentError(f"Invalid .env entry on line {index + 1}. Expected NAME=value.")
        name = match.group(1)
        if name in env:
            continue
        value = match.group(2).strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        else:
            value = re.sub(r"\s+#.*$", "", value).strip()
        env[name] = value


def _cleaned(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip()


def parse_positive_integer(value: str | None, name: str, fallback: int) -> int:
    """Parse a strictly positive integer setting."""
    cleaned = _cleaned(value)
    if not cleaned:
        return fallback
    if not _POSITIVE_INTEGER.match(cleaned):
        raise AgentError(f"{name} must be a positive integer.")
    parsed = int(cleaned)
    if parsed < 1:
        raise AgentError(f"{name} must be a positive integer.")
    return parsed


def parse_directory_entry_limit(value: str | None) -> int:
    """Parse ``WORKSPACE_LIST_LIMIT``: ``0``, ``-1``, or a positive integer."""
    cleaned = _cleaned(value)
    if not cleaned:
        return 0
    if not _DIRECTORY_ENTRY_LIMIT.match(cleaned):
        raise AgentError("WORKSPACE_LIST_LIMIT must be 0, -1, or a positive integer.")
    parsed = int(cleaned)
    if parsed < -1:
        raise AgentError("WORKSPACE_LIST_LIMIT must be 0, -1, or a positive integer.")
    return parsed


def parse_terminal_mode(value: str | None) -> str:
    """Parse ``TERMINAL_MODE``, which only accepts lowercase auto/ask/off."""
    normalized = (value or "").strip()
    if normalized in ("auto", "ask", "off"):
        return normalized
    raise AgentError("TERMINAL_MODE must be lowercase: auto, ask, or off.")


def parse_boolean_setting(value: str | None, name: str, fallback: bool) -> bool:
    """Parse an ``on``/``off`` boolean setting."""
    cleaned = _cleaned(value)
    if not cleaned:
        return fallback
    normalized = cleaned.lower()
    if normalized == "on":
        return True
    if normalized == "off":
        return False
    raise AgentError(f"{name} must be on or off.")


def parse_input_modalities(value: str | None) -> list[str]:
    """Parse ``OPENAI_INPUT``; it must include ``text`` and may include ``image``."""
    source = value if (_cleaned(value) or "") else "text,image"
    items = [item.lower() for item in re.split(r"[\s,]+", source) if item]
    unique = list(dict.fromkeys(items))
    if "text" not in unique or any(item not in ("text", "image") for item in unique):
        raise AgentError("OPENAI_INPUT must include text and only supports the values text,image.")
    return unique


def assert_supported_python_version(version: str | None = None) -> None:
    """Fail fast when the interpreter predates the features the agent relies on."""
    running = version or ".".join(str(part) for part in sys.version_info[:3])
    parts = running.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        major, minor = 0, 0
    if major < 3 or (major == 3 and minor < 12):
        raise AgentError(f"MinAgent requires Python 3.12 or later. Installed version: {running}.")


def _trim_single_trailing_slash(path: str) -> str:
    return path[:-1] if path.endswith("/") else path


def make_endpoint(base_url: str) -> str:
    """Normalise a base URL to its ``/chat/completions`` endpoint."""
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise AgentError("OPENAI_BASE_URL must be a valid HTTP or HTTPS URL.")
    trimmed = _trim_single_trailing_slash(parsed.path or "")
    if not trimmed.endswith("/chat/completions"):
        trimmed = f"{trimmed}/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, trimmed, parsed.query, ""))


@dataclass(frozen=True, slots=True)
class Config:
    """Validated MinAgent configuration.

    Immutable and explicitly typed, so a mistyped setting name fails at
    import/type-check time instead of surfacing as a ``KeyError`` at runtime.
    """

    application_root: str
    root_directory: str
    workspace_name: str
    endpoint: str
    api_key: str | None
    model: str
    context_window: int
    endpoint_timeout_ms: int
    input_modalities: list[str]
    show_reasoning: bool
    compaction_reserve_tokens: int
    compaction_keep_recent_tokens: int
    workspace_list_limit: int
    terminal_mode: str
    terminal_command_shell: str
    terminal_timeout_seconds: int
    mcp_timeout_ms: int
    skills_enabled: bool
    mcp_enabled: bool
    memory_enabled: bool
    memory_db_path: str
    memory_direct_answer: bool
    web_search_enabled: bool
    ollama_api_key: str | None
    web_search_base_url: str
    web_search_timeout_seconds: int


def load_configuration(
    application_root: str | None = None,
    cwd: str | None = None,
    env: MutableMapping[str, str] | None = None,
) -> Config:
    """Load and validate the full MinAgent configuration."""
    assert_supported_python_version()
    root = application_root or find_application_root()
    environment = os.environ if env is None else env
    load_env_file(os.path.join(root, ".env"), environment)

    working_directory = cwd or os.getcwd()
    root_directory = os.path.realpath(working_directory)
    model = (environment.get("OPENAI_MODEL") or "").strip()
    if not model:
        raise AgentError("Set OPENAI_MODEL to the model identifier available on your endpoint.")

    context_window = parse_positive_integer(
        environment.get("OPENAI_CONTEXT_WINDOW"), "OPENAI_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW
    )
    endpoint_timeout_seconds = parse_positive_integer(
        environment.get("OPENAI_TIMEOUT_SECONDS"),
        "OPENAI_TIMEOUT_SECONDS",
        DEFAULT_ENDPOINT_TIMEOUT_SECONDS,
    )
    mcp_timeout_seconds = parse_positive_integer(
        environment.get("MCP_TIMEOUT_SECONDS"),
        "MCP_TIMEOUT_SECONDS",
        DEFAULT_EXTENSION_TIMEOUT_SECONDS,
    )
    terminal_timeout_seconds = parse_positive_integer(
        environment.get("TERMINAL_TIMEOUT_SECONDS"),
        "TERMINAL_TIMEOUT_SECONDS",
        DEFAULT_EXTENSION_TIMEOUT_SECONDS,
    )
    web_search_timeout_seconds = parse_positive_integer(
        environment.get("WEB_SEARCH_TIMEOUT_SECONDS"),
        "WEB_SEARCH_TIMEOUT_SECONDS",
        DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
    )
    workspace_name = os.path.basename(root_directory) or "workspace"
    if os.name == "nt":
        terminal_shell = (environment.get("ComSpec") or "").strip() or "cmd.exe"
    else:
        terminal_shell = "/bin/sh"

    return Config(
        application_root=root,
        root_directory=root_directory,
        workspace_name=workspace_name,
        endpoint=make_endpoint(environment.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"),
        api_key=(environment.get("OPENAI_API_KEY") or "").strip() or None,
        model=model,
        context_window=context_window,
        endpoint_timeout_ms=endpoint_timeout_seconds * 1000,
        input_modalities=parse_input_modalities(environment.get("OPENAI_INPUT")),
        compaction_reserve_tokens=min(16384, context_window // 8),
        compaction_keep_recent_tokens=min(20000, context_window // 8),
        workspace_list_limit=parse_directory_entry_limit(environment.get("WORKSPACE_LIST_LIMIT")),
        terminal_mode=parse_terminal_mode(environment.get("TERMINAL_MODE") or "ask"),
        terminal_command_shell=terminal_shell,
        terminal_timeout_seconds=terminal_timeout_seconds,
        mcp_timeout_ms=mcp_timeout_seconds * 1000,
        skills_enabled=parse_boolean_setting(environment.get("SKILLS_ENABLED"), "SKILLS_ENABLED", False),
        show_reasoning=parse_boolean_setting(
            environment.get("OPENAI_SHOW_REASONING"), "OPENAI_SHOW_REASONING", False
        ),
        mcp_enabled=parse_boolean_setting(environment.get("MCP_ENABLED"), "MCP_ENABLED", False),
        memory_enabled=parse_boolean_setting(environment.get("MEMORY_ENABLED"), "MEMORY_ENABLED", False),
        memory_db_path=(environment.get("MEMORY_DB_PATH") or "").strip()
        or os.path.join(root, ".agents", "memory", "memoria.db"),
        memory_direct_answer=parse_boolean_setting(
            environment.get("MEMORY_DIRECT_ANSWER"), "MEMORY_DIRECT_ANSWER", True
        ),
        web_search_enabled=parse_boolean_setting(
            environment.get("WEB_SEARCH_ENABLED"), "WEB_SEARCH_ENABLED", False
        ),
        ollama_api_key=(environment.get("OLLAMA_API_KEY") or "").strip() or None,
        web_search_base_url=(environment.get("WEB_SEARCH_BASE_URL") or "https://ollama.com/api")
        .strip()
        .rstrip("/"),
        web_search_timeout_seconds=web_search_timeout_seconds,
    )
