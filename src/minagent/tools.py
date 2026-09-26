"""Tool schemas exposed to the model.

Kept apart from the conversation loop so the prompt surface can be read and
tested without loading the terminal UI.
"""

from __future__ import annotations

from typing import Any


def build_tools() -> list[dict[str, Any]]:
    """The workspace tool schemas exposed to the model."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a workspace file or a specifically user-provided file path outside it; never list outside directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 1, "description": "First line to return, starting at 1"},
                        "limit": {"type": "integer", "minimum": 1, "description": "Maximum number of lines to return"},
                        "column": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Character position within the first returned line, starting at 1; use the continuation value for long lines",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List immediate workspace entries (default: root), including hidden entries; do not follow links. Raise limit if truncated.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10000,
                            "description": "Maximum entries to return; defaults to 500",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace one exact, unique text block in an existing workspace file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string", "description": "Non-empty exact text to replace; it must occur once"},
                        "new_text": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "old_text", "new_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Create or replace one workspace file; missing parent folders are created. Use create_directory for a folder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string", "description": "Complete file contents"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_directory",
                "description": "Create a workspace folder, including missing parent folders.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete one regular file inside the workspace.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_directory",
                "description": "Recursively delete a workspace subdirectory; linked or special entries are blocked.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            },
        },
    ]


def build_terminal_tool() -> dict[str, Any]:
    """The shell tool schema, exposed only when the terminal is enabled."""
    return {
        "type": "function",
        "function": {
            "name": "run_terminal",
            "description": (
                "Run one shell command in the workspace. Use it for system facts such as the current "
                "date and time (`date`), the environment, or installed tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }


def build_tool_output_recall_tool() -> dict[str, Any]:
    """The schema for reading back an archived tool result.

    A result too large for the context window is stored in full and replaced by
    a preview that names its id, so the model can retrieve any part of it
    instead of working from a permanently lossy summary.
    """
    return {
        "type": "function",
        "function": {
            "name": "recall_tool_output",
            "description": (
                "Read back a tool result that was too large to keep in the conversation. A truncated "
                "result ends with a note naming its id; pass that id here, with an offset and limit, "
                "to read any character range of the original output. Reread before assuming what a "
                "truncated result said."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The archived output id from the truncation note"},
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "First character to return, starting at 0",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 40000,
                        "description": "Maximum number of characters to return; defaults to 8000",
                    },
                },
                "required": ["id"],
            },
        },
    }
