"""Configuration loading and validation.

Reads ``.env`` from the MinAgent project root, then validates every setting and
normalises the OpenAI-compatible endpoint to its ``/chat/completions`` URL.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from .capabilities import DEFAULT_CAPABILITY_IDLE_TURNS
from .compute import (
    DEFAULT_JOB_TIMEOUT_SECONDS,
    DEFAULT_QUEUE_LIMIT,
    DEFAULT_VRAM_TOTAL_MIB,
    VOICE_TIMEOUT_SECONDS,
    parse_ollama_mode,
)
from .context_budget import ContextPolicy
from .errors import AgentError, find_application_root

DEFAULT_CONTEXT_WINDOW = 262144
DEFAULT_MAX_TOOL_ROUNDS = 64
# Inline characters kept for an oversized tool result. The rest stays in the
# archive and the model reads it back with recall_tool_output.
DEFAULT_TOOL_PREVIEW_CHARS = 12000
# Recent tool results kept verbatim in the transcript. Older ones are replaced by
# a retrievable stub before compaction, which frees context without summarising.
DEFAULT_TOOL_RESULT_KEEP = 3
# Whether consecutive read-only tools in one response may run at the same time.
# Writing tools, approval-gated tools and MCP tools are never batched.
DEFAULT_PARALLEL_TOOLS = True
DEFAULT_CONTEXT_HIGH_WATERMARK = ContextPolicy().high_watermark
DEFAULT_CONTEXT_LOW_WATERMARK = ContextPolicy().low_watermark
DEFAULT_ENDPOINT_TIMEOUT_SECONDS = 7 * 60
DEFAULT_EXTENSION_TIMEOUT_SECONDS = 7 * 60
DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = 2 * 60
DEFAULT_VISION_TIMEOUT_SECONDS = 5 * 60
"""A vision model on a laptop GPU takes minutes; the default has to clear that."""
_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_POSITIVE_INTEGER = re.compile(r"^\d+$")
_DIRECTORY_ENTRY_LIMIT = re.compile(r"^-?\d+$")
_RATIO = re.compile(r"^\d*\.?\d+$")


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


def parse_non_negative_integer(value: str | None, name: str, fallback: int) -> int:
    """Parse a setting where ``0`` is a meaningful value, not a missing one."""
    cleaned = _cleaned(value)
    if not cleaned:
        return fallback
    if not _POSITIVE_INTEGER.match(cleaned):
        raise AgentError(f"{name} must be zero or a positive integer.")
    return int(cleaned)


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


def parse_approval_mode(value: str | None, name: str, fallback: str) -> str:
    """Parse an approval mode setting, which only accepts lowercase auto/ask/off.

    ``ask`` is the safe default: a tool that was never confirmed should not run
    just because the user was not looking at the prompt.
    """
    normalized = (value or "").strip()
    if not normalized:
        return fallback
    if normalized in ("auto", "ask", "off"):
        return normalized
    raise AgentError(f"{name} must be lowercase: auto, ask, or off.")


def parse_ratio_setting(value: str | None, name: str, fallback: float) -> float:
    """Parse a window fraction written as ``0.75``, ``75%``, or ``75``."""
    cleaned = _cleaned(value)
    if not cleaned:
        return fallback
    normalized = cleaned[:-1].strip() if cleaned.endswith("%") else cleaned
    if not _RATIO.match(normalized):
        raise AgentError(f"{name} must be a percentage like 75% or a fraction like 0.75.")
    parsed = float(normalized)
    ratio = parsed / 100 if cleaned.endswith("%") or parsed > 1 else parsed
    if not 0 < ratio <= 1:
        raise AgentError(f"{name} must be greater than 0 and at most 1 (or 100%).")
    return ratio


def parse_ollama_unload_mode(value: str | None) -> str:
    """Parse ``COMPUTE_UNLOAD_OLLAMA``, which decides who may evict a model.

    The default is ``off`` because unloading is not free for the user: the
    model reloads on the next request. It is a cache Ollama keeps on purpose, so
    dropping it to start a video is a trade the person at the keyboard should
    agree to, not one the agent should make silently.
    """
    return parse_ollama_mode(value)


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


def parse_on_demand_images(value: str | None) -> bool:
    """Parse ``IMAGE_INPUT_MODE``.

    ``on_demand`` is the default: an image path stays a path until the model asks
    for the pixels. ``eager`` attaches them straight away, which is the old
    behaviour and the only reason to want it is a model that refuses to call a
    tool before it will look at anything.
    """
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"on_demand", "on-demand", "ondemand"}:
        return True
    if normalized == "eager":
        return False
    raise AgentError(
        f"IMAGE_INPUT_MODE must be on_demand or eager, got {value!r}."
    )


def parse_input_modalities(value: str | None) -> list[str]:
    """Parse ``OPENAI_INPUT``; it must include ``text`` and may include ``image``."""
    source = _cleaned(value) or "text,image"
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
    max_tool_rounds: int
    tool_preview_chars: int
    tool_result_keep: int
    parallel_tools: bool
    capability_idle_turns: int
    context_high_watermark: float
    context_low_watermark: float
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
    mcp_approval_mode: str
    memory_enabled: bool
    memory_db_path: str
    memory_direct_answer: bool
    memory_eureka: bool
    memory_reflection_interval: int
    web_search_enabled: bool
    ollama_api_key: str | None
    web_search_base_url: str
    web_search_timeout_seconds: int
    vision_enabled: bool
    vision_model: str
    vision_base_url: str
    vision_timeout_seconds: int
    on_demand_images: bool
    compute_enabled: bool
    compute_vram_total_mib: int
    compute_job_timeout_seconds: int
    compute_voice_timeout_seconds: int
    compute_unload_ollama: str
    compute_queue_limit: int


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
    vision_timeout_seconds = parse_positive_integer(
        environment.get("VISION_TIMEOUT_SECONDS"),
        "VISION_TIMEOUT_SECONDS",
        DEFAULT_VISION_TIMEOUT_SECONDS,
    )
    workspace_name = os.path.basename(root_directory) or "workspace"
    terminal_shell = (
        (environment.get("ComSpec") or "").strip() or "cmd.exe" if os.name == "nt" else "/bin/sh"
    )

    return Config(
        application_root=root,
        root_directory=root_directory,
        workspace_name=workspace_name,
        endpoint=make_endpoint(environment.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"),
        api_key=(environment.get("OPENAI_API_KEY") or "").strip() or None,
        model=model,
        context_window=context_window,
        endpoint_timeout_ms=endpoint_timeout_seconds * 1000,
        max_tool_rounds=parse_positive_integer(
            environment.get("MAX_TOOL_ROUNDS"), "MAX_TOOL_ROUNDS", DEFAULT_MAX_TOOL_ROUNDS
        ),
        tool_preview_chars=parse_positive_integer(
            environment.get("TOOL_PREVIEW_CHARS"), "TOOL_PREVIEW_CHARS", DEFAULT_TOOL_PREVIEW_CHARS
        ),
        tool_result_keep=parse_positive_integer(
            environment.get("TOOL_RESULT_KEEP"), "TOOL_RESULT_KEEP", DEFAULT_TOOL_RESULT_KEEP
        ),
        parallel_tools=parse_boolean_setting(
            environment.get("PARALLEL_TOOLS"), "PARALLEL_TOOLS", DEFAULT_PARALLEL_TOOLS
        ),
        capability_idle_turns=parse_non_negative_integer(
            environment.get("CAPABILITY_IDLE_TURNS"),
            "CAPABILITY_IDLE_TURNS",
            DEFAULT_CAPABILITY_IDLE_TURNS,
        ),
        context_high_watermark=parse_ratio_setting(
            environment.get("CONTEXT_HIGH_WATERMARK"),
            "CONTEXT_HIGH_WATERMARK",
            DEFAULT_CONTEXT_HIGH_WATERMARK,
        ),
        context_low_watermark=parse_ratio_setting(
            environment.get("CONTEXT_LOW_WATERMARK"),
            "CONTEXT_LOW_WATERMARK",
            DEFAULT_CONTEXT_LOW_WATERMARK,
        ),
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
        mcp_approval_mode=parse_approval_mode(
            environment.get("MCP_APPROVAL_MODE"), "MCP_APPROVAL_MODE", "ask"
        ),
        memory_enabled=parse_boolean_setting(environment.get("MEMORY_ENABLED"), "MEMORY_ENABLED", False),
        memory_db_path=(environment.get("MEMORY_DB_PATH") or "").strip()
        or os.path.join(root, ".agents", "memory", "memoria.db"),
        memory_eureka=parse_boolean_setting(environment.get("MEMORY_EUREKA"), "MEMORY_EUREKA", True),
        memory_reflection_interval=parse_positive_integer(
            environment.get("MEMORY_REFLECTION_INTERVAL"), "MEMORY_REFLECTION_INTERVAL", 10
        ),
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
        vision_enabled=parse_boolean_setting(environment.get("VISION_ENABLED"), "VISION_ENABLED", False),
        vision_model=(environment.get("VISION_MODEL") or "qwen3.5:9b-q4_K_M").strip(),
        vision_base_url=(environment.get("VISION_BASE_URL") or "http://localhost:11434")
        .strip()
        .rstrip("/"),
        vision_timeout_seconds=vision_timeout_seconds,
        on_demand_images=parse_on_demand_images(environment.get("IMAGE_INPUT_MODE")),
        compute_enabled=parse_boolean_setting(
            environment.get("COMPUTE_ENABLED"), "COMPUTE_ENABLED", False
        ),
        compute_vram_total_mib=parse_positive_integer(
            environment.get("COMPUTE_VRAM_TOTAL_MIB"),
            "COMPUTE_VRAM_TOTAL_MIB",
            DEFAULT_VRAM_TOTAL_MIB,
        ),
        compute_job_timeout_seconds=parse_positive_integer(
            environment.get("COMPUTE_JOB_TIMEOUT_SECONDS"),
            "COMPUTE_JOB_TIMEOUT_SECONDS",
            DEFAULT_JOB_TIMEOUT_SECONDS,
        ),
        compute_voice_timeout_seconds=parse_positive_integer(
            environment.get("COMPUTE_VOICE_TIMEOUT_SECONDS"),
            "COMPUTE_VOICE_TIMEOUT_SECONDS",
            VOICE_TIMEOUT_SECONDS,
        ),
        compute_unload_ollama=parse_ollama_unload_mode(
            environment.get("COMPUTE_UNLOAD_OLLAMA")
        ),
        compute_queue_limit=parse_positive_integer(
            environment.get("COMPUTE_QUEUE_LIMIT"),
            "COMPUTE_QUEUE_LIMIT",
            DEFAULT_QUEUE_LIMIT,
        ),
    )
