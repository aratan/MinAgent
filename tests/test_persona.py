"""Ara's persona: loaded from files, capped, and honest about both."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minagent.persona import (  # noqa: E402
    PERSONA_MAX_BYTES,
    describe_persona,
    load_persona_file,
    persona_overflows,
    persona_sections,
)


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_missing_persona_is_not_an_error(tmp_path: Path) -> None:
    """No file yet means no section, not a broken prompt."""
    assert persona_sections(str(tmp_path)) == []
    assert load_persona_file(str(tmp_path / "nope.md")) is None


def test_a_written_persona_is_loaded_in_order(tmp_path: Path) -> None:
    _write(tmp_path, "agente/IDENTITY.md", "I am Ara.")
    _write(tmp_path, "agente/USER.md", "Answer in Spanish.")
    sections = persona_sections(str(tmp_path))
    assert [section["name"] for section in sections] == ["Identity", "User"]
    assert sections[0]["content"] == "I am Ara."


def test_the_real_persona_is_small() -> None:
    """The persona that ships has to be worth paying for on every request.

    The file this replaced was a personality of 413,207 bytes. Nothing stops an
    edited file from growing again, so the shipped one is asserted to stay far
    below the cap rather than merely assumed to.
    """
    root = Path(__file__).resolve().parent.parent
    assert not persona_overflows(str(root)), f"shipped persona is oversized: {persona_overflows(str(root))}"


def test_an_oversized_persona_is_cut_and_says_it_was_cut(tmp_path: Path) -> None:
    """A context bomb has to announce itself, or it is just context.

    Truncating silently is the failure this exists to prevent: the model would
    read a persona with rules missing from the end and behave as though those
    rules had never been written.
    """
    big = tmp_path / "BIG.md"
    big.write_text("HEAD-MARKER\n" + ("x" * (PERSONA_MAX_BYTES * 2)), encoding="utf-8")
    loaded = load_persona_file(str(big), limit=512)

    assert loaded is not None
    assert "HEAD-MARKER" in loaded
    assert "[truncated:" in loaded
    assert "512" in loaded
    assert len(loaded.encode("utf-8")) < 1024


def test_truncation_never_splits_a_character(tmp_path: Path) -> None:
    """The cap is in bytes, so the cut point is usually mid-character."""
    path = tmp_path / "MULTI.md"
    path.write_text("ñ" * 2000, encoding="utf-8")
    loaded = load_persona_file(str(path), limit=101)
    assert loaded is not None
    assert "\udcff" not in loaded  # lone surrogate from a bad decode
    assert "ñ" in loaded  # the bytes around it survived intact


def test_a_persona_at_the_cap_is_left_alone(tmp_path: Path) -> None:
    """The cap is a ceiling, not a target: an exactly-sized file is whole."""
    path = tmp_path / "EXACT.md"
    path.write_text("a" * 512, encoding="utf-8")
    loaded = load_persona_file(str(path), limit=512)
    assert loaded == "a" * 512
    assert "[truncated:" not in loaded


def test_an_unreadable_persona_is_treated_as_absent(tmp_path: Path) -> None:
    """A corrupt file must not take the prompt down with it."""
    path = tmp_path / "BROKEN.md"
    path.write_bytes(b"\xff\xfe\x00invalid utf-8 \xc3\x28")
    assert load_persona_file(str(path)) is None


def test_an_empty_persona_adds_no_section(tmp_path: Path) -> None:
    """Whitespace is not an identity, and an empty section is paid for anyway."""
    _write(tmp_path, "agente/IDENTITY.md", "   \n\n  ")
    assert persona_sections(str(tmp_path)) == []


def test_overflow_is_reported_for_the_human_not_just_the_model(tmp_path: Path) -> None:
    """The model is told in-band; the human is told here, by name."""
    _write(tmp_path, "agente/IDENTITY.md", "x" * 900)
    _write(tmp_path, "agente/USER.md", "short")
    reported = persona_overflows(str(tmp_path), limit=512)
    assert len(reported) == 1
    assert reported[0].startswith("Identity")
    assert "900" in reported[0]


def test_describe_is_readable_when_there_is_nothing_to_describe() -> None:
    assert describe_persona([]) == "sin personalidad escrita"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_the_persona_lands_after_the_rules_never_before_them() -> None:
    """A description that overwrote the rules could talk the agent out of them.

    Ordering is the whole safety property: Core carries what must hold
    whatever else is loaded, so nothing an agent writes about itself can sit
    in front of it and be read first.
    """
    from minagent.app import MinAgent

    class _Silent:
        def write(self, text: str) -> None:
            pass

    app = MinAgent(stdout=_Silent())
    app.application_root = str(Path(__file__).resolve().parent.parent)
    app.workspace_name = "Test"
    sections = app.build_base_system_prompt()
    names = [section["name"] for section in sections]

    assert names[0] == "Core"
    assert "Identity" in names and "User" in names
    assert names.index("Core") < names.index("Identity")
