"""Subagent tests: what is accepted, what is refused, and what reaches git."""

import pytest

from minagent.errors import AgentError
from minagent.subagents import (
    BRANCH_PREFIX,
    SubagentStore,
    render_template,
    validate_source,
)

GOOD = '''from typing import Any


def create_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": "read_invoice", "parameters": {"type": "object", "properties": {}}},
        }
    ]
'''


@pytest.fixture
def git_calls() -> list[list[str]]:
    return []


@pytest.fixture
def store(git_calls) -> SubagentStore:
    git_calls.clear()
    return SubagentStore(runner=git_calls.append)


def test_the_template_parses_and_carries_the_interface():
    """The first module should be a copy, so the template has to be a real one."""
    source = render_template("facturas", "Lee facturas en PDF.", "read_invoice", "Lee una factura")
    validate_source("facturas", source)
    assert "def create_tools" in source
    assert "read_invoice" in source


def test_a_syntax_error_is_reported_with_a_line_number():
    with pytest.raises(AgentError, match="does not parse"):
        validate_source("roto", "def create_tools(:\n    pass")


def test_a_module_without_the_interface_is_refused():
    with pytest.raises(AgentError, match="must define create_tools"):
        validate_source("vacio", "x = 1\n")


@pytest.mark.parametrize("name", ["Facturas", "9vidas", "con guion", "a", "x" * 60])
def test_a_name_that_would_not_be_an_import_name_is_refused(name):
    with pytest.raises(AgentError, match="Invalid module name"):
        validate_source(name, GOOD)


def test_an_oversized_module_is_refused_as_a_project():
    with pytest.raises(AgentError, match="project, not a tool"):
        validate_source("enorme", GOOD + "#" + "x" * (64 * 1024))


def test_a_durable_module_is_committed_to_its_own_branch(store, git_calls):
    store.create("facturas", GOOD)
    assert git_calls[0] == ["git", "checkout", "-B", f"{BRANCH_PREFIX}facturas"]
    assert git_calls[1] == ["git", "add", ".minagent/subagents/facturas.py"]
    assert git_calls[2][0] == "git" and git_calls[2][1] == "commit"


def test_the_module_is_on_disk_before_git_is_asked_to_add_it(tmp_path, git_calls):
    """``git add`` on a path that does not exist fails, and the branch would be
    left pointing at nothing."""
    store = SubagentStore(str(tmp_path / "subagents"), runner=git_calls.append)
    store.create("facturas", GOOD)
    written = tmp_path / "subagents" / "facturas.py"
    assert written.exists()
    assert written.read_text() == GOOD


def test_a_failed_commit_does_not_leave_a_stray_branch(tmp_path, git_calls):
    store = SubagentStore(str(tmp_path / "subagents"), runner=git_calls.append)

    def runner(command: list[str]) -> None:
        git_calls.append(command)
        if command[1] == "commit":
            # What _git raises when the command fails.
            raise AgentError("git failed while writing facturas: simulated")

    store._runner = runner
    with pytest.raises(AgentError, match="simulated"):
        store.create("facturas", GOOD)
    # The rollback has to be attempted, and it has to name the branch.
    assert ["git", "branch", "-D", f"{BRANCH_PREFIX}facturas"] in git_calls


def test_an_ephemeral_module_never_reaches_git(store, git_calls):
    store.create("temporal", GOOD, ephemeral=True)
    assert git_calls == []


def test_a_module_that_offers_no_tools_is_refused(store, git_calls):
    empty = GOOD.replace('"name": "read_invoice"', '"name": ""').replace("[", "[] if True else [")
    with pytest.raises(AgentError):
        store.create("vacio", empty)
    assert git_calls == []


def test_a_module_that_raises_while_loading_is_reported_not_swallowed():
    """The model wrote it, so the traceback is its problem to fix and the
    message has to name the error rather than fail on the next tool call."""
    broken = GOOD.replace("    return [", "    raise ValueError('nope')\n    return [")
    with pytest.raises(AgentError, match="create_tools\\(\\) raised"):
        SubagentStore(runner=lambda c: None).create("falla", broken, ephemeral=True)


def test_a_module_that_returns_the_wrong_shape_is_refused():
    wrong = "def create_tools():\n    return 'no soy una lista'\n"
    with pytest.raises(AgentError, match="non-empty list"):
        SubagentStore(runner=lambda c: None).create("raro", wrong, ephemeral=True)


def test_a_module_raising_at_import_time_names_the_error():
    at_import = "raise RuntimeError('fallo al importar')\n\n\ndef create_tools():\n    return []\n"
    with pytest.raises(AgentError, match="raised while loading"):
        SubagentStore(runner=lambda c: None).create("importa", at_import, ephemeral=True)


def test_ephemeral_modules_are_capped_so_the_catalogue_cannot_grow_forever(git_calls):
    store = SubagentStore(max_ephemeral=3, runner=git_calls.append)
    for index in range(5):
        store.create(f"temp{index}", GOOD, ephemeral=True)
    assert len(store.names()) == 3
    assert "temp0" not in store.names()


def test_replacing_a_durable_module_says_where_the_old_one_stayed(store):
    store.create("facturas", GOOD)
    with pytest.raises(AgentError, match="already stored on a branch"):
        store.create("facturas", GOOD)


def test_deleting_something_that_was_never_written_lists_what_exists(store):
    with pytest.raises(AgentError, match="Available:"):
        store.delete("nunca")


def test_deleting_a_durable_module_removes_it_on_its_own_branch(store, git_calls):
    store.create("facturas", GOOD)
    git_calls.clear()
    store.delete("facturas")
    assert git_calls[0] == ["git", "checkout", f"{BRANCH_PREFIX}facturas"]
    assert git_calls[1] == ["git", "rm", "-f", ".minagent/subagents/facturas.py"]


def test_an_empty_catalogue_says_what_a_module_needs(store):
    assert "create_tools" in store.list_modules()


def test_the_catalogue_names_the_tools_each_module_contributes(store):
    store.create("facturas", GOOD)
    assert "read_invoice" in store.list_modules()
    assert "on branch" in store.list_modules()
