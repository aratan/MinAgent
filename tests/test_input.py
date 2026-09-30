"""Input control tests: the ydotool command lines, and what is refused.

The command lines are asserted exactly rather than loosely, because ydotool's
contract with this module is the whole point of it: it does not fail loudly on
a key name it cannot read, it emits a delay and carries on, so a wrong code
here would look like a working click that never landed.
"""

import re
from pathlib import Path

import pytest

from minagent.errors import AgentError
from minagent.input import (
    BUTTON_CLICK_CODES,
    KEYCODES,
    InputClient,
    create_input_tools,
    keycode,
)


@pytest.fixture
def calls() -> list[list[str]]:
    return []


@pytest.fixture
def client(calls: list[list[str]]) -> InputClient:
    calls.clear()
    return InputClient(runner=lambda arguments: calls.append(arguments) or "")


async def test_a_chord_holds_the_modifier_and_releases_it_in_reverse(client, calls):
    assert await client.key("ctrl+shift+t") == "Pressed ctrl+shift+t."
    # KEY_LEFTCTRL 29, KEY_LEFTSHIFT 42, KEY_T 20: down in order, up reversed.
    assert calls[0] == ["key", "--", "29:1", "42:1", "20:1", "20:0", "42:0", "29:0"]


async def test_a_single_key_needs_no_sequence_padding(client, calls):
    await client.key("enter")
    assert calls[0] == ["key", "--", "28:1", "28:0"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("q", 16), ("t", 20), ("p", 25), ("a", 30), ("l", 38), ("z", 44), ("m", 50),
        ("0", 11), ("9", 10), ("f5", 63),
    ],
)
def test_letters_and_digits_use_the_kernel_numbering(name, expected):
    """The three letter rows are 16, 30 and 44, not one run from KEY_A."""
    assert keycode(name) == expected


def test_letters_are_not_one_run_from_a():
    """The arithmetic that looks right is wrong, and q is the witness.

    Filling the table as KEY_A plus the letter's offset is the obvious way to
    write it and it types the wrong key: q is 16, not 46. Found against a real
    desktop, where every other letter happened to land somewhere plausible.
    """
    assert keycode("q") == 16
    assert keycode("q") != 30 + (ord("q") - ord("a"))


def test_the_whole_table_agrees_with_the_kernel_it_runs_on():
    """Every letter and digit, checked against the header the kernel itself uses.

    The spot checks above cover the rows this project has typed by hand before.
    The rest of the alphabet had never been checked against anything, and a
    wrong code is silent: ydotool accepts a number and presses whatever is at
    that number. The kernel header is the authority, so it is the oracle.
    """
    header = Path("/usr/include/linux/input-event-codes.h")
    if not header.is_file():
        pytest.skip("the kernel input header is not installed")
    definitions = {
        name: int(value)
        for name, value in re.findall(r"^#define\s+(KEY_[A-Z0-9]+)\s+(\d+)", header.read_text(), re.M)
    }
    characters = [key for key in KEYCODES if len(key) == 1 and key.isalnum()]
    assert len(characters) == 36, "the table should carry all 26 letters and 10 digits"
    wrong = {
        key: (KEYCODES[key], definitions.get(f"KEY_{key.upper()}"))
        for key in characters
        if KEYCODES[key] != definitions.get(f"KEY_{key.upper()}")
    }
    assert wrong == {}


async def test_an_unknown_key_is_refused_rather_than_silently_delayed(client, calls):
    """ydotool accepts a name it cannot read and only delays, so this must not reach it."""
    with pytest.raises(AgentError, match="Unknown key"):
        await client.key("nope+z")
    assert calls == []


async def test_moving_absolutely_passes_the_flag_and_needs_no_read_back(client, calls):
    assert await client.move_mouse(100, 200) == "Moved the pointer to (100, 200)."
    assert calls[0] == ["mousemove", "--absolute", "--", "100", "200"]


async def test_moving_relatively_omits_the_flag(client, calls):
    assert await client.move_mouse(-10, 5, relative=True) == "Moved the pointer by (-10, 5)."
    assert calls[0] == ["mousemove", "--", "-10", "5"]


async def test_a_coordinate_far_off_the_screen_is_a_mistake_not_a_move(client, calls):
    with pytest.raises(AgentError, match="within"):
        await client.move_mouse(99999, 0)
    assert calls == []


async def test_one_coordinate_alone_is_refused(client):
    with pytest.raises(AgentError, match="both x and y"):
        await client.move_mouse(100, None)  # type: ignore[arg-type]
    with pytest.raises(AgentError, match="both x and y"):
        await client.move_mouse(None, 100)  # type: ignore[arg-type]


@pytest.mark.parametrize("button,code", sorted(BUTTON_CLICK_CODES.items()))
async def test_a_click_uses_the_ydotool_button_codes(client, calls, button, code):
    await client.click(button)
    assert hex(code) in calls[0]


async def test_a_double_click_repeats_the_code_inside_one_call(client, calls):
    assert await client.click("left", 2) == "Double-clicked the left button."
    # One call, the code twice: two separate calls would not be a double click.
    assert calls[0].count("0xc0") == 2


async def test_a_click_count_of_three_is_not_a_thing(client):
    with pytest.raises(AgentError, match="1 or 2"):
        await client.click("left", 3)


async def test_holding_and_releasing_a_button_are_separate_calls(client, calls):
    await client.mouse_button_down("right")
    await client.mouse_button_up("right")
    assert calls[0] == ["click", "--", "0x41"]
    assert calls[1] == ["click", "--", "0x81"]


async def test_an_unknown_button_is_refused(client, calls):
    with pytest.raises(AgentError, match="Unknown button"):
        await client.click("scroll")
    assert calls == []


async def test_scrolling_down_sends_a_negative_count_behind_a_separator(client, calls):
    """Without the ``--`` a negative count is read as an unknown flag and scrolls nothing."""
    assert await client.scroll("down", 2) == "Scrolled down by 2 wheel click(s)."
    assert calls[0] == ["mousemove", "--wheel", "--", "0", "-2"]


async def test_scrolling_up_sends_a_positive_count(client, calls):
    await client.scroll("up", 3)
    assert calls[0] == ["mousemove", "--wheel", "--", "0", "3"]


async def test_an_amount_beyond_a_reasonable_scroll_is_refused(client, calls):
    with pytest.raises(AgentError, match="at most 500"):
        await client.scroll("down", 5000)
    assert calls == []


async def test_scrolling_rejects_a_direction_it_cannot_map(client):
    with pytest.raises(AgentError, match="Unknown scroll direction"):
        await client.scroll("sideways")


async def test_typing_uses_the_flag_this_ydotool_actually_has(client, calls):
    """``type`` takes --key-delay; the shorter --delay does not exist."""
    await client.type_text("hola")
    assert calls[0][:2] == ["type", "--key-delay"]


def test_typing_is_bounded_so_a_model_cannot_paste_a_whole_file(client):
    import asyncio

    with pytest.raises(AgentError, match="at most 4000"):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
            client.type_text("x" * 4001)
        )


def test_the_tools_name_real_keyboard_and_mouse_operations():
    names = {tool["function"]["name"] for tool in create_input_tools()}
    assert names == {
        "press_keys",
        "type_text",
        "move_mouse",
        "click_mouse",
        "scroll_screen",
        "mouse_button_down",
        "mouse_button_up",
    }
