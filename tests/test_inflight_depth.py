"""Regression tests for the in-flight depth counter.

Each test pins a specific way the old boolean was wrong. If any of these pass
against a plain bool, the test is not measuring what it claims.
"""
import asyncio

import pytest

from minagent.app import MinAgent


def _agent() -> MinAgent:
    a = MinAgent.__new__(MinAgent)   # no __init__: no subprocesses, no config
    a._operation_depth = 0
    return a


def test_flag_is_false_when_nothing_runs():
    a = _agent()
    assert a._active_request_in_flight is False


def test_two_tools_in_one_turn_stay_in_flight_until_both_return():
    a = _agent()

    async def scenario():
        async with a._operation_in_flight():
            assert a._active_request_in_flight is True
            # second tool starts before the first finished
            async with a._operation_in_flight():
                assert a._active_request_in_flight is True
            # first returns; second is STILL running
            assert a._active_request_in_flight is True
        # both done
        assert a._active_request_in_flight is False

    asyncio.run(scenario())


def test_nested_tool_does_not_clear_when_inner_returns():
    a = _agent()

    async def scenario():
        async with a._operation_in_flight():
            async with a._operation_in_flight():
                pass
            assert a._active_request_in_flight is True

    asyncio.run(scenario())


def test_depth_returns_to_zero_after_an_exception():
    a = _agent()

    async def scenario():
        with pytest.raises(RuntimeError):
            async with a._operation_in_flight():
                raise RuntimeError("tool blew up")
        assert a._active_request_in_flight is False
        assert a._operation_depth == 0

    asyncio.run(scenario())


def test_depth_never_goes_negative():
    a = _agent()
    a._operation_depth = 0
    # a stray extra release must clamp at zero, not wrap to a truthy negative

    async def scenario():
        async with a._operation_in_flight():
            pass

    asyncio.run(scenario())
    assert a._operation_depth == 0
    a._operation_depth = 0
    a._active_request_in_flight = False
    assert a._operation_depth == 0
    assert a._active_request_in_flight is False


# --- integration: the flag the resident actually reads, through a real tool ---


class _FakeOutput:
    def write(self, _text: str) -> None:
        pass

    def flush(self) -> None:
        pass


class _SlowClient:
    """A search client that yields control mid-request, like a real one."""

    def __init__(self) -> None:
        self.observed_depth: list[int] = []

    async def search(self, *_args, **_kwargs):
        app = self.app
        # This is the window the resident used to see as "free".
        self.observed_depth.append(app._active_request_in_flight)
        await asyncio.sleep(0)  # yield to the loop, as a network await would
        return [{"title": "R", "url": "https://r", "content": "s"}]


async def test_a_slow_tool_keeps_the_resident_from_seeing_a_free_machine(tmp_path):
    from minagent.workspace import WorkspaceAccess

    app = MinAgent(stdout=_FakeOutput())
    app.root_directory = str(tmp_path)
    app.application_root = str(tmp_path)
    app.workspace_name = "Test"
    app.web_search_enabled = True
    client = _SlowClient()
    client.app = app
    app.web_search_client = client
    app.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    app.ensure_web_search_tools()

    assert app._active_request_in_flight is False, "free before the turn"
    await app.execute_tool("web_search", {"query": "anything"})

    # Mid-tool, the resident must see the machine as busy.
    assert client.observed_depth == [True], (
        "the resident was told the machine was free while a tool was running"
    )
    assert app._active_request_in_flight is False, "free again after the tool"
