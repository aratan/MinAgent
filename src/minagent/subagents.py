"""Modules the agent writes for itself, and sub-agents it runs.

An agent that can only do what its author thought of will keep hitting the
same wall. A module here is the way past it: a small Python file that exposes
``create_tools()``, which the agent writes once and then loads like any
built-in capability.

That is also the risk, stated plainly: this runs code the model wrote, on this
machine, with the user's permissions. Nothing in a prompt prevents a module
from doing something the user would not have chosen, which is why the durable
ones go onto a git branch. Not as a safety mechanism that prevents harm - a
branch does not stop code from running - but as the review point: the change is
a diff, in a branch, that the user can read before it matters and revert after.
Ephemeral modules are the opposite trade: they never touch the repository, so
there is nothing to review, and they are dropped instead.

Every module is parsed and checked for the interface before it is stored, so a
syntax error or a missing ``create_tools`` is a message to the model rather
than an exception the next tool call stumbles into.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import AgentError

MODULE_SUFFIX = ".py"
DEFAULT_DIRECTORY = ".minagent/subagents"
DEFAULT_MAX_EPHEMERAL = 4

MAX_MODULE_BYTES = 64 * 1024
"""A capability module is a tool, not a program. Past this it is a project."""

REQUIRED_FUNCTION = "create_tools"
"""The one name a module must define, so the loader has a contract to rely on."""

SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{1,47}$")
"""Lowercase snake_case: these names become import names and file names."""

BRANCH_PREFIX = "agent/module-"

INTERFACE_TEMPLATE = '''"""{summary}"""

from typing import Any


def create_tools() -> list[dict[str, Any]]:
    """The tool schemas this module contributes to the agent."""
    return [
        {{
            "type": "function",
            "function": {{
                "name": "{tool_name}",
                "description": "{description}",
                "parameters": {{
                    "type": "object",
                    "properties": {{
                        "value": {{"type": "string", "description": "What to act on"}}
                    }},
                    "required": ["value"],
                }},
            }},
        }}
    ]
'''
"""The smallest module that loads, so the model has a shape to copy rather than
invent. It is deliberately a stub: a module that already does the work would be
a module the model has to read instead of write."""


@dataclass
class SubagentModule:
    """One written module, whether it is in memory or on disk."""

    name: str
    source: str
    ephemeral: bool
    tools: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """One line about what it offers, for the capability index."""
        names = [tool.get("function", {}).get("name", "?") for tool in self.tools]
        return f"{self.name} ({'ephemeral' if self.ephemeral else 'on branch'}): {', '.join(names)}"


def validate_source(name: str, source: str) -> None:
    """Check a module parses and offers the interface, before it is stored.

    A model that writes a broken module should be told now, in terms it can
    act on, rather than have the import fail inside a later tool call where the
    stack trace means nothing to it.
    """
    if not SAFE_NAME.match(name or ""):
        raise AgentError(
            f"Invalid module name {name!r}. Use lowercase letters, digits and underscores, "
            "starting with a letter, e.g. 'invoice_reader'."
        )
    if not source.strip():
        raise AgentError("The module source is empty.")
    if len(source.encode("utf-8")) > MAX_MODULE_BYTES:
        raise AgentError(
            f"The module is over {MAX_MODULE_BYTES // 1024} KB. That is a project, not a tool."
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise AgentError(
            f"The module does not parse: line {error.lineno}, {error.msg}. Fix that first."
        ) from error
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if REQUIRED_FUNCTION not in defined:
        raise AgentError(
            f"The module must define {REQUIRED_FUNCTION}(). Without it there is nothing to load."
        )


def build_tools(name: str, source: str) -> list[dict[str, Any]]:
    """Import a validated module and return the tools it contributes.

    This executes code the model wrote, on this machine, with the user's
    permissions. That is what the feature is - a module is a tool the agent
    wrote for itself - and it is why the durable ones land on a branch the
    user can read, and why the guidance tells the agent to say so out loud.

    It runs in its own module namespace rather than this process's, so a
    module cannot reach in and mutate the agent's own globals while importing.
    """
    import types

    namespace: dict[str, Any] = {"__name__": f"minagent_subagent_{name}", "__file__": f"{name}{MODULE_SUFFIX}"}
    module = types.ModuleType(namespace["__name__"])
    module.__dict__.update(namespace)
    try:
        exec(compile(source, f"{name}{MODULE_SUFFIX}", "exec"), module.__dict__)
    except Exception as error:  # noqa: BLE001 - the model wrote it, report it
        raise AgentError(
            f"The module raised while loading: {type(error).__name__}: {error}"
        ) from error
    factory = module.__dict__.get(REQUIRED_FUNCTION)
    if factory is None:
        raise AgentError(f"The module must define {REQUIRED_FUNCTION}().")
    try:
        tools = factory()
    except Exception as error:  # noqa: BLE001 - the model wrote it, report it
        raise AgentError(
            f"{REQUIRED_FUNCTION}() raised: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(tools, list) or not tools:
        raise AgentError(
            f"{REQUIRED_FUNCTION}() must return a non-empty list of tool schemas, "
            f"got {type(tools).__name__}."
        )
    for tool in tools:
        if not isinstance(tool, dict) or "function" not in tool or "name" not in tool.get("function", {}):
            raise AgentError(
                f"{REQUIRED_FUNCTION}() returned a malformed tool: every entry needs a "
                "'function' with a 'name'."
            )
    return list(tools)


def render_template(name: str, summary: str, tool_name: str, description: str) -> str:
    """A working skeleton, so the first module is a copy rather than a guess."""
    return INTERFACE_TEMPLATE.format(
        summary=(summary or f"The {name} capability.").replace('"""', "'''"),
        tool_name=tool_name,
        description=description.replace('"', "'"),
    )


class SubagentStore:
    """Holds the modules this session has written, durable and throwaway alike."""

    def __init__(
        self,
        directory: str = DEFAULT_DIRECTORY,
        max_ephemeral: int = DEFAULT_MAX_EPHEMERAL,
        runner: Any = None,
    ) -> None:
        self.directory = Path(directory)
        self.max_ephemeral = max_ephemeral
        self._runner = runner
        self._modules: dict[str, SubagentModule] = {}

    # -- listing ---------------------------------------------------------

    def names(self) -> list[str]:
        """Every module name held, durable first."""
        return sorted(self._modules)

    def get(self, name: str) -> SubagentModule:
        """One module, or an error listing what exists."""
        module = self._modules.get(name)
        if module is None:
            available = ", ".join(self.names()) or "none yet"
            raise AgentError(f"No module named {name!r}. Available: {available}.")
        return module

    def list_modules(self) -> str:
        """The catalogue, one line per module."""
        if not self._modules:
            return (
                "No modules written yet. write_module creates one; it needs a name and a "
                f"{REQUIRED_FUNCTION}() function."
            )
        return "\n".join(module.summary() for module in self._modules.values())

    # -- writing ---------------------------------------------------------

    def create(
        self,
        name: str,
        source: str,
        ephemeral: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> SubagentModule:
        """Store a module, on a branch or only in memory.

        The tools are whatever the caller managed to build from the source, so
        a module that parsed but produced nothing usable is caught before it
        reaches the catalogue.
        """
        validate_source(name, source)
        if tools is None:
            tools = build_tools(name, source)
        if not tools:
            raise AgentError(
                f"{REQUIRED_FUNCTION}() returned no tools. A module that offers nothing is dead weight."
            )
        if name in self._modules and not ephemeral:
            existing = self._modules[name]
            if not existing.ephemeral:
                raise AgentError(
                    f"{name} is already stored on a branch. Use write_module again to replace it; "
                    "the old version stays in git history."
                )
        if ephemeral:
            self._evict_old_ephemeral()
        if not ephemeral:
            self._write_to_branch(name, source)
        module = SubagentModule(
            name=name, source=source, ephemeral=ephemeral, tools=list(tools or [])
        )
        self._modules[name] = module
        return module

    def _evict_old_ephemeral(self) -> None:
        """Drop the oldest throwaway modules once there are too many.

        Bounded on purpose: a store that only grows is a store the context
        window pays for, and throwaway modules are the ones with the least
        claim on it.
        """
        ephemeral = [m for m in self._modules.values() if m.ephemeral]
        while len(ephemeral) >= self.max_ephemeral:
            oldest = ephemeral.pop(0)
            self._modules.pop(oldest.name, None)

    def delete(self, name: str) -> str:
        """Remove a module from the catalogue.

        Only removes the file for a durable one when the caller asks: the file
        is the reviewable artefact, and unlinking it quietly would undo the
        reason for putting it on a branch in the first place.
        """
        module = self.get(name)
        self._modules.pop(name, None)
        if not module.ephemeral:
            self._remove_from_branch(name)
        return name

    def _write_to_branch(self, name: str, source: str) -> None:
        """Write the file, then commit it to its own branch.

        The file goes down first: ``git add`` on a path that does not exist
        fails, and committing before writing would leave a branch pointing at
        nothing. Every module gets a branch of its own rather than sharing
        one, so a half-finished module cannot be mistaken for a finished one
        and the diff shows exactly what that module adds.
        """
        path = self.directory / f"{name}{MODULE_SUFFIX}"
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        branch = f"{BRANCH_PREFIX}{name}"
        commands = [
            # -B, not -b: rewriting a module in a later session must replace its
            # branch, and ``checkout -b`` refuses when the branch already exists.
            ["git", "checkout", "-B", branch],
            ["git", "add", str(path)],
            ["git", "commit", "-m", f"agent: add {name} module"],
        ]
        for index, command in enumerate(commands):
            try:
                self._run_git(command, f"writing {name}")
            except AgentError:
                if index > 0:
                    # The branch was created by the checkout above, so a failure
                    # at any later step leaves it pointing at nothing. Take it
                    # back off rather than leave a stray branch behind.
                    try:
                        self._run_git(["git", "branch", "-D", branch], f"rolling back {name}")
                    except AgentError:
                        pass
                raise

    def _remove_from_branch(self, name: str) -> None:
        """Commit the removal on the module's own branch."""
        path = self.directory / f"{name}{MODULE_SUFFIX}"
        commands = [
            ["git", "checkout", f"{BRANCH_PREFIX}{name}"],
            ["git", "rm", "-f", str(path)],
            ["git", "commit", "-m", f"agent: remove {name} module"],
        ]
        for command in commands:
            self._run_git(command, f"removing {name}")

    def _run_git(self, command: list[str], what: str) -> None:
        """One git command, through the test runner when there is one.

        Both paths go through here so the rollback above cannot end up on a
        branch of the code that the tests never execute.
        """
        if self._runner is not None:
            self._runner(command)
            return
        self._git(command, what)

    def _git(self, command: list[str], what: str) -> None:
        """Run one git command, turning its complaint into something readable."""
        import shutil

        binary = shutil.which(command[0])
        if binary is None:
            raise AgentError(f"{command[0]} is not installed, so modules cannot be stored.")
        result = subprocess.run(
            [binary, *command[1:]],
            cwd=str(self.directory.parent) if self.directory.parent.exists() else None,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            tail = detail[-1] if detail else "no output"
            raise AgentError(f"git failed while {what}: {tail}")


def create_subagent_tools() -> list[dict[str, Any]]:
    """The tool schemas for writing and managing modules."""
    return [
        {
            "type": "function",
            "function": {
                "name": "write_module",
                "description": (
                    "Write a new capability module: a Python file defining create_tools(), the same "
                    "shape the built-in capabilities use. A durable module is committed to its own "
                    "git branch, but only after the user approves it in the terminal, because it is "
                    "Python you wrote and it will run on their machine. Use ephemeral=true for "
                    "something you will only need this turn: it needs no approval and never reaches "
                    "the repository."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "lowercase_snake_case, e.g. 'invoice_reader'",
                        },
                        "source": {"type": "string", "description": "The full module source"},
                        "ephemeral": {
                            "type": "boolean",
                            "default": False,
                            "description": "true keeps it in memory only and drops it later",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One line on what it is for, for the catalogue",
                        },
                    },
                    "required": ["name", "source"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_modules",
                "description": "List the modules written this session, and the tools each contributes.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_module",
                "description": "Remove a module from the catalogue. A durable one is also removed on its branch.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "module_template",
                "description": (
                    "Return a minimal working module to copy, so a new one is a copy rather than a guess."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Module name to template"},
                        "summary": {"type": "string", "description": "One line on what it is for"},
                    },
                    "required": ["name"],
                },
            },
        },
    ]


SUBAGENTS_GUIDANCE = (
    "A module is Python you wrote, loaded as a capability: it must define create_tools() returning "
    "tool schemas. That means it runs on this machine with the user's permissions. A durable module is "
    "therefore confirmed in the terminal before anything is written, and its branch is the place the "
    "user can read what it does. If they decline, do not ask again and do not try to reach the same "
    "result another way: offer ephemeral=true instead, and say what the difference is. Prefer "
    "ephemeral=true for a one-turn helper anyway: it needs no approval, never reaches the repository, "
    "and is dropped when the session stops using it. A durable module is for something that will be "
    "wanted again, and when you write one, say plainly in your answer that it is code you wrote."
)
"""Sits with the tools, because the honesty requirement is part of the feature."""
