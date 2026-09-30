"""Compute tests: the VRAM budget, job serialization, and the agent wiring.

The orchestrator is the only place that decides whether a GPU job is allowed to
start, so these tests are mostly about that decision. Everything that would
otherwise need a card is injected: ``_vram_probe`` supplies the reading, and
``runner`` stands in for the subprocess. That is what makes the interesting
cases - a refused job, two jobs racing, an offload choice - testable at all.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

import minagent.compute
from minagent.app import MinAgent
from minagent.capabilities import build_builtin_capabilities
from minagent.compute import (
    MUSIC_TOOL_NAME,
    QUEUE_TOOL_NAME,
    RESULT_TOOL_NAME,
    SPEAK_TOOL_NAME,
    STATUS_TOOL_NAME,
    TRANSCRIBE_TOOL_NAME,
    VIDEO_TOOL_NAME,
    VOICE_RESERVE_MIB,
    VRAM_HEADROOM_MIB,
    ComputeOrchestrator,
    VramReading,
    VramRelease,
    create_compute_tools,
    estimate_video_vram,
    parse_result_line,
)
from minagent.config import load_configuration
from minagent.errors import AgentError
from minagent.workspace import WorkspaceAccess

FULL_CARD = 8188


def _orchestrator(tmp_path: Path, **overrides: Any) -> ComputeOrchestrator:
    """An orchestrator over a fake card, with a runner that records its calls.

    The backends are stubbed into the temporary project so the real script
    resolution runs: a test that skipped that path would not notice a rename.

    The VRAM reading is fixed here, and that is load-bearing rather than
    incidental. A test that measures the real card passes or fails depending on
    whether Ollama happens to have a model resident, which is not something a
    test about job ordering should care about. One test did, and failed for a
    reason that had nothing to do with what it was checking.
    """
    scripts = tmp_path / "scripts" / "compute"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ("voz.py", "video_ltx.py", "musica.py"):
        (scripts / name).write_text('print("RESULT {\\"ok\\": true, \\"path\\": \\"salida/stub\\"}")\n')
    orchestrator = ComputeOrchestrator(root_directory=str(tmp_path), **overrides)
    orchestrator._vram_probe = lambda: VramReading(
        total_mib=FULL_CARD, free_mib=FULL_CARD - 100, used_by_others_mib=100
    )
    return orchestrator


def _recorder(result: str = "RESULT {\"ok\": true, \"path\": \"salida/x.mp4\"}") -> Any:
    calls: list[list[str]] = []

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        calls.append(list(argv))
        return result

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


# ----------------------------------------------------------------- the budget
def test_a_job_that_fits_is_allowed(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    reading = orchestrator.require_headroom("Video", 4000)
    # A clean card leaves the full budget minus the reserve and the headroom.
    assert reading.available_for_heavy_mib == FULL_CARD - 100 - VOICE_RESERVE_MIB - VRAM_HEADROOM_MIB
    assert orchestrator.read_vram().free_mib == FULL_CARD - 100


def test_a_job_that_does_not_fit_is_refused_with_the_numbers(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator._vram_probe = lambda: VramReading(
        total_mib=FULL_CARD, free_mib=900, used_by_others_mib=6300, processes=("llama-server (6300 MiB)",)
    )
    with pytest.raises(AgentError) as failure:
        orchestrator.require_headroom("Video", 5200)
    message = str(failure.value)
    assert "5200" in message
    assert "900" in message
    # The voice reserve is why the card is short, and the holder is named: the
    # useful answer is "llama-server is using 6.2 GB", not "out of memory".
    assert str(VOICE_RESERVE_MIB) in message
    assert "llama-server" in message
    assert "frames" in message


def test_a_music_refusal_does_not_suggest_changing_video_settings(tmp_path: Path) -> None:
    """Advice that does not match the job sends the model off to fix nothing."""
    orchestrator = _orchestrator(tmp_path)
    orchestrator._vram_probe = lambda: VramReading(total_mib=FULL_CARD, free_mib=300)
    with pytest.raises(AgentError) as failure:
        orchestrator.require_headroom("Music generation", 3200)
    message = str(failure.value)
    # "shorter duration" and not merely "music": advice has to be something the
    # caller can actually act on. Naming a model that is not on offer would send
    # them to a choice that does not exist.
    assert "shorter duration" in message
    assert "frames" not in message


def test_the_voice_reserve_is_never_lent_to_a_heavy_job(tmp_path: Path) -> None:
    """The reserve is the reason voice stays responsive, so it is not negotiable."""
    orchestrator = _orchestrator(tmp_path)
    orchestrator._vram_probe = lambda: VramReading(
        total_mib=FULL_CARD, free_mib=VOICE_RESERVE_MIB + VRAM_HEADROOM_MIB + 500
    )
    with pytest.raises(AgentError):
        orchestrator.require_headroom("Music", 2800)


def test_a_card_with_no_nvidia_gpu_still_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host with no card must not fail every call; the budget maths still works."""
    from minagent.compute import probe_vram

    monkeypatch.setenv("PATH", "/nonexistent")
    reading = probe_vram(FULL_CARD)
    assert reading.total_mib == FULL_CARD
    assert reading.free_mib == FULL_CARD
    assert reading.used_by_others_mib == 0


def test_the_probe_names_the_process_holding_the_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The useful answer is "llama-server has 6.2 GB", not "out of memory"."""
    from minagent.compute import probe_vram

    _fake_nvidia_smi(monkeypatch, tmp_path, "10589, /usr/local/lib/ollama/llama-server, 6390 MiB\n")
    reading = probe_vram(FULL_CARD)
    assert reading.total_mib == 8188
    assert reading.free_mib == 7346
    assert reading.used_by_others_mib == 6390
    assert "llama-server" in reading.processes[0]


def test_the_orchestrator_does_not_ask_the_user_to_kill_its_own_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A running backend is real VRAM, but it is not a third party to go and free.

    Otherwise a status call during a render would tell the user to kill the
    video they just asked for.
    """
    from minagent.compute import probe_vram

    _fake_nvidia_smi(monkeypatch, tmp_path, "4242, python3, 5000 MiB\n10589, llama-server, 6390 MiB\n")
    both = probe_vram(FULL_CARD)
    assert both.used_by_others_mib == 11_390
    assert len(both.processes) == 2

    own_only = probe_vram(FULL_CARD, ignore_pids={4242})
    assert own_only.used_by_others_mib == 6390
    assert all("python3" not in name for name in own_only.processes)

    # And the free figure is untouched either way: the memory is still spent.
    assert own_only.free_mib == both.free_mib


# ------------------------------------------------------------- serialization
async def test_two_heavy_jobs_never_overlap(tmp_path: Path) -> None:
    """The whole scheduling policy: one at a time, in submission order."""
    orchestrator = _orchestrator(tmp_path)
    order: list[str] = []
    active = 0
    peak = 0

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        order.append(argv[1])
        await asyncio.sleep(0.01)
        active -= 1
        return 'RESULT {"ok": true, "path": "salida/x.mp4"}'

    await asyncio.gather(
        orchestrator.generate_video("one", frames=17, runner=runner),
        orchestrator.generate_music("two", runner=runner),
    )
    assert peak == 1
    assert order == ["one", "two"]


async def test_voice_does_not_wait_behind_a_render(tmp_path: Path) -> None:
    """The point of resident voice: it answers while a video is still running."""
    orchestrator = _orchestrator(tmp_path)
    rendering = asyncio.Event()

    async def heavy_runner(script: Path, argv: list[str], timeout: int) -> str:
        rendering.set()
        await asyncio.sleep(0.05)
        return 'RESULT {"ok": true, "path": "salida/x.mp4"}'

    async def voice_runner(script: Path, argv: list[str], timeout: int) -> str:
        # If speak_text took the heavy lock, this would deadlock behind the
        # render rather than returning while the render is still in flight.
        return 'RESULT {"ok": true, "path": "salida/habla.wav"}'

    video = asyncio.create_task(orchestrator.generate_video("x", frames=17, runner=heavy_runner))
    await rendering.wait()
    spoken = await orchestrator.speak("hola", runner=voice_runner)
    assert not video.done()
    assert spoken.endswith("habla.wav")
    await video


async def test_a_refused_job_never_takes_the_lock(tmp_path: Path) -> None:
    """A job that cannot fit must not block the ones that can."""
    orchestrator = _orchestrator(tmp_path)
    orchestrator._vram_probe = lambda: VramReading(total_mib=FULL_CARD, free_mib=400)
    with pytest.raises(AgentError):
        await orchestrator.generate_video("x", frames=97, runner=_recorder())
    assert not orchestrator.busy


# ------------------------------------------------------------------ estimates
def test_frames_drive_video_memory_and_offload_chooses_the_mode() -> None:
    """The guidance tells the model frames matter and steps do not."""
    small = estimate_video_vram(17, "group")
    large = estimate_video_vram(97, "group")
    assert large > small
    assert estimate_video_vram(49, "sequential") < estimate_video_vram(49, "group")


def test_only_the_lightest_music_model_fits_alongside_voice(tmp_path: Path) -> None:
    """The default has to be the one that runs, not the one that sounds best."""
    from minagent.compute import MUSIC_VRAM_ESTIMATE_MIB

    assert min(MUSIC_VRAM_ESTIMATE_MIB, key=lambda name: MUSIC_VRAM_ESTIMATE_MIB[name]) == "small"
    assert estimate_video_vram(49, "sequential") < estimate_video_vram(49, "group")


async def test_an_unknown_offload_mode_is_refused(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    with pytest.raises(AgentError) as failure:
        await orchestrator.generate_video("x", offload="telepathy", runner=_recorder())
    assert "offload" in str(failure.value)


async def _run(coroutine: Any) -> Any:
    return await coroutine


# ------------------------------------------------------------------- outputs
def test_the_result_line_is_parsed_and_failures_become_errors() -> None:
    assert parse_result_line('RESULT {"ok": true, "path": "salida/a.mp4"}') == "salida/a.mp4"
    with pytest.raises(AgentError) as failure:
        parse_result_line('RESULT {"ok": false, "error": "se acabó la VRAM"}')
    assert "VRAM" in str(failure.value)


def test_the_poster_is_reported_alongside_the_video() -> None:
    line = json.dumps({"ok": True, "path": "salida/a.mp4", "poster": "salida/a.png"})
    assert "salida/a.png" in parse_result_line(f"RESULT {line}")


def test_output_paths_cannot_escape_the_output_directory(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    escaped = orchestrator.output_path("../../etc/passwd", "mp4", "ltx")
    assert escaped.parent == tmp_path / "salida"
    assert escaped.name == "passwd.mp4"
    # A bare name falls back to the stem, with the right extension.
    assert orchestrator.output_path("", "mp4", "ltx").name == "ltx.mp4"
    assert orchestrator.output_path("clip", "mp4", "ltx").name == "clip.mp4"


async def test_transcription_refuses_a_path_outside_the_workspace(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    outside = tmp_path.parent / "elsewhere.mp3"
    outside.write_bytes(b"not really audio")
    with pytest.raises(AgentError) as failure:
        await orchestrator.transcribe(str(outside))
    assert "workspace" in str(failure.value)


# ------------------------------------------------------------------ the agent
class _FakeOutput:
    def write(self, value: str) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _agent(tmp_path: Path, env: dict[str, str]) -> MinAgent:
    agent = MinAgent(stdout=_FakeOutput())
    agent.root_directory = str(tmp_path)
    agent.application_root = str(tmp_path)
    agent.workspace_name = "Test"
    agent.workspace_access = WorkspaceAccess(str(tmp_path), "Test", 0)
    config = load_configuration(
        str(tmp_path), str(tmp_path), {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "m", **env}
    )
    agent.compute_enabled = config.compute_enabled
    if agent.compute_enabled:
        agent.orchestrator = ComputeOrchestrator(
            root_directory=str(tmp_path),
            vram_total_mib=config.compute_vram_total_mib,
            job_timeout_seconds=config.compute_job_timeout_seconds,
            voice_timeout_seconds=config.compute_voice_timeout_seconds,
        )
    agent.ensure_compute_tools()
    agent.rebuild_capabilities()
    return agent


def test_compute_is_off_unless_it_is_asked_for(tmp_path: Path) -> None:
    agent = _agent(tmp_path, {})
    assert not agent.compute_enabled
    assert agent.orchestrator is None
    with pytest.raises(AgentError) as failure:
        agent.run_compute_status({})
    assert "COMPUTE_ENABLED" in str(failure.value)


def test_the_tools_are_registered_but_not_callable_until_loaded(tmp_path: Path) -> None:
    """On-demand loading is what keeps five schemas out of every request."""
    agent = _agent(tmp_path, {"COMPUTE_ENABLED": "on"})
    assert agent.compute_enabled
    assert agent.capabilities is not None
    assert agent.capabilities.get("compute") is not None

    # The schema is registered with the agent, but not published to the model:
    # an unloaded capability must cost nothing per request.
    assert SPEAK_TOOL_NAME in agent._tool_schemas
    assert SPEAK_TOOL_NAME not in {tool["function"]["name"] for tool in agent.tools}

    agent.capabilities.load(["compute"])
    agent.publish_loaded_tools()
    published = {tool["function"]["name"] for tool in agent.tools}
    assert SPEAK_TOOL_NAME in published
    assert VIDEO_TOOL_NAME in published
    assert MUSIC_TOOL_NAME in published
    assert TRANSCRIBE_TOOL_NAME in published
    assert STATUS_TOOL_NAME in published


def test_the_guidance_states_the_vram_policy(tmp_path: Path) -> None:
    """The model cannot sequence a card it was never told about."""
    entries = build_builtin_capabilities(terminal_mode="off", compute_enabled=True)
    guidance = next(entry for entry in entries if entry.name == "compute").guidance
    assert "8 GB" in guidance
    assert "serialized" in guidance
    # The two facts that stop the expensive mistakes.
    assert "frames" in guidance and "steps do not" in guidance
    assert "compute_status" in guidance


def test_no_compute_capability_when_it_is_disabled() -> None:
    entries = build_builtin_capabilities(terminal_mode="off", compute_enabled=False)
    assert [entry.name for entry in entries].count("compute") == 0


def test_every_tool_has_a_schema_and_a_label(tmp_path: Path) -> None:
    schemas = {schema["function"]["name"] for schema in create_compute_tools()}
    assert schemas == {
        SPEAK_TOOL_NAME,
        TRANSCRIBE_TOOL_NAME,
        VIDEO_TOOL_NAME,
        MUSIC_TOOL_NAME,
        QUEUE_TOOL_NAME,
        RESULT_TOOL_NAME,
        STATUS_TOOL_NAME,
    }
    from minagent.app import FILE_TOOL_LABELS

    for name in schemas:
        assert name in FILE_TOOL_LABELS


def _fake_nvidia_smi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, table: str) -> None:
    """Put a stub nvidia-smi on PATH so the probe reads a known card.

    The real binary is shadowed rather than mocked, because the probe resolves
    it with ``shutil.which`` and shells out; patching either would skip the
    part that actually breaks when the output format changes.
    """
    binary = tmp_path / "bin" / "nvidia-smi"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        '  *memory.total*) echo "8188 MiB, 7346 MiB" ;;\n'
        f'  *) printf %s "{table}" ;;\n'
        "esac\n"
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary.parent))


async def test_the_probe_counts_the_processes_holding_the_card(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The unit-suffixed form is the one this box used to fail on.

    The queries ask for ``nounits``, so the table arrives as bare numbers. A
    pattern that required the "MiB" suffix dropped every row, and the reading
    said nothing was using the card while llama-server sat on 6.4 GB of it - so
    a job that could not fit was refused instead of unloading anything, and the
    refusal blamed free memory that was there.
    """
    from minagent.compute import probe_vram

    _fake_nvidia_smi(monkeypatch, tmp_path, "103536, /usr/local/lib/ollama/llama-server, 6392\\n")

    reading = probe_vram(8188)

    assert reading.used_by_others_mib == 6392
    assert reading.processes == ("llama-server (6392 MiB)",)


async def test_the_probe_still_reads_a_table_that_keeps_the_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from minagent.compute import probe_vram

    _fake_nvidia_smi(monkeypatch, tmp_path, "103536, /usr/local/lib/ollama/llama-server, 6392 MiB\\n")

    assert probe_vram(8188).used_by_others_mib == 6392


async def test_status_reports_the_measurement_not_a_guess(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    orchestrator._vram_probe = lambda: VramReading(
        total_mib=FULL_CARD, free_mib=900, used_by_others_mib=6300, processes=("llama-server (6300 MiB)",)
    )
    report = orchestrator.status_text()
    assert "900 MiB free of 8188" in report
    assert "llama-server" in report
    assert "idle" in report


def test_config_reads_the_compute_settings(tmp_path: Path) -> None:
    config = load_configuration(
        str(tmp_path),
        str(tmp_path),
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_MODEL": "m",
            "COMPUTE_ENABLED": "on",
            "COMPUTE_VRAM_TOTAL_MIB": "16384",
            "COMPUTE_JOB_TIMEOUT_SECONDS": "600",
        },
    )
    assert config.compute_enabled is True
    assert config.compute_vram_total_mib == 16384
    assert config.compute_job_timeout_seconds == 600


def test_a_bad_compute_setting_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(AgentError) as failure:
        load_configuration(
            str(tmp_path),
            str(tmp_path),
            {"OPENAI_API_KEY": "test", "OPENAI_MODEL": "m", "COMPUTE_VRAM_TOTAL_MIB": "mucha"},
        )
    assert "COMPUTE_VRAM_TOTAL_MIB" in str(failure.value)


# --------------------------------------------------------------- the backends
def test_the_backend_scripts_exist_and_report_json() -> None:
    """A backend that cannot emit RESULT {...} is invisible to the orchestrator."""
    root = Path(__file__).resolve().parent.parent
    for name in ("voz.py", "video_ltx.py", "musica.py"):
        script = root / "scripts" / "compute" / name
        assert script.is_file(), f"{name} is missing"
        assert "def emit(" in script.read_text(encoding="utf-8")
        # The allocator setting has to be set before CUDA initialises.
        assert "PYTORCH_CUDA_ALLOC_CONF" in script.read_text(encoding="utf-8")


def test_a_missing_backend_says_where_it_was_looked_for(tmp_path: Path) -> None:
    orchestrator = ComputeOrchestrator(root_directory=str(tmp_path))
    with pytest.raises(AgentError) as failure:
        orchestrator.resolve_script("video_ltx.py")
    message = str(failure.value)
    assert "scripts" in message
    assert "COMPUTE_SCRIPTS_DIR" in message


def test_a_missing_whisper_says_how_to_install_it() -> None:
    """The one diagnosis the user cannot work out alone.

    whisper.cpp is a separate install from a separate project, and the failure
    mode that actually happens is a symlink pointing at a build directory that
    a reboot or a /tmp cleanup removed. That reads as a missing binary, so the
    message has to carry the install command and the name the binary is built
    under, or the user is left with a model on disk and no words.
    """
    script = Path(__file__).resolve().parent.parent / "scripts" / "compute" / "voz.py"
    environment = {**os.environ, "PATH": "", "WHISPER_MODEL_DIR": ""}
    completed = subprocess.run(
        [sys.executable, str(script), "--audio", "lo-que-sea.wav"],
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    payload = completed.stdout.split("RESULT ", 1)[-1]
    answer = json.loads(payload)
    assert answer["ok"] is False
    assert "whisper.cpp" in answer["error"]
    assert "pacman -S whisper.cpp" in answer["error"]
    assert "whisper-cli" in answer["error"]


def test_compute_scripts_dir_overrides_where_backends_are_looked_for(tmp_path: Path) -> None:
    """The escape hatch for a checkout whose scripts live outside the project."""
    elsewhere = tmp_path / "otro-sitio"
    elsewhere.mkdir()
    (elsewhere / "video_ltx.py").write_text("# stub\n")
    orchestrator = ComputeOrchestrator(root_directory=str(tmp_path))
    os.environ["COMPUTE_SCRIPTS_DIR"] = str(elsewhere)
    try:
        assert orchestrator.resolve_script("video_ltx.py") == elsewhere / "video_ltx.py"
    finally:
        del os.environ["COMPUTE_SCRIPTS_DIR"]


def test_a_generated_path_is_reported_relative_to_the_workspace(tmp_path: Path) -> None:
    agent = _agent(tmp_path, {"COMPUTE_ENABLED": "on"})
    assert agent.orchestrator is not None
    absolute = str(tmp_path / "salida" / "a.mp4")
    assert agent.relative_to_workspace(absolute) == os.path.join("salida", "a.mp4")


# ---------------------------------------------------------------- the queue
async def test_a_queued_job_returns_an_id_without_waiting(tmp_path: Path) -> None:
    """A render is minutes; asking for one must not spend minutes of turn time."""
    orchestrator = _orchestrator(tmp_path)
    job_id = orchestrator.submit_video("un gato", frames=17)
    assert job_id.startswith("job-")
    assert "waiting" in orchestrator.result_text(job_id).lower()
    await asyncio.sleep(0.05)


async def test_several_jobs_queue_in_order_and_run_one_at_a_time(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    active = 0
    peak = 0
    order: list[str] = []

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        order.append(argv[1])
        await asyncio.sleep(0.02)
        active -= 1
        return 'RESULT {"ok": true, "path": "salida/x.mp4"}'

    orchestrator._spawn = runner  # type: ignore[method-assign]
    ids = [orchestrator.submit_video(f"clip {index}", frames=17) for index in range(3)]
    for job_id in ids:
        for _ in range(100):
            if "finished" in orchestrator.result_text(job_id):
                break
            await asyncio.sleep(0.01)
    assert peak == 1
    assert order == ["clip 0", "clip 1", "clip 2"]


async def test_a_queued_job_reports_waiting_then_done(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    release = asyncio.Event()
    order: list[str] = []

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        order.append(argv[1])
        if argv[1] == "blocker":
            await release.wait()
        return 'RESULT {"ok": true, "path": "salida/queued.mp4"}'

    orchestrator._spawn = runner  # type: ignore[method-assign]
    # The blocker goes first and holds the card, so the second is observably
    # waiting rather than running.
    orchestrator.submit_video("blocker", frames=17)
    job_id = orchestrator.submit_video("x", frames=17)
    await asyncio.sleep(0.03)
    assert "position 1" in orchestrator.result_text(job_id)
    release.set()
    for _ in range(100):
        if "finished" in orchestrator.result_text(job_id):
            break
        await asyncio.sleep(0.01)
    assert order == ["blocker", "x"]
    assert "salida/queued.mp4" in orchestrator.result_text(job_id)


async def test_a_late_poll_still_gets_the_result(tmp_path: Path) -> None:
    """Finishing must not make the id stop answering, or a slow poll loses the work."""
    orchestrator = _orchestrator(tmp_path)

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        return 'RESULT {"ok": true, "path": "salida/late.mp4"}'

    orchestrator._spawn = runner  # type: ignore[method-assign]
    job_id = orchestrator.submit_video("x", frames=17)
    await asyncio.sleep(0.05)
    assert "salida/late.mp4" in orchestrator.result_text(job_id)
    assert orchestrator.queued == 0


async def test_a_refused_job_keeps_answering_its_id(tmp_path: Path) -> None:
    """A job that cannot run must still answer, saying why.

    It failed because the card was full, which is the whole point of the VRAM
    check. The bug this pins down is that the background task swallowed the
    refusal, so the id left the queue without an outcome and every later poll
    reported "No job with id 'job-1'" - a job that was refused reading as one
    that was never created.
    """
    orchestrator = _orchestrator(tmp_path)
    orchestrator._vram_probe = lambda: VramReading(total_mib=FULL_CARD, free_mib=200)
    job_id = orchestrator.submit_music("jazz", seconds=5)
    answer = ""
    for _ in range(100):
        answer = orchestrator.result_text(job_id)
        if "failed" in answer:
            break
        await asyncio.sleep(0.01)
    assert "failed" in answer
    assert "VRAM" in answer, "the refusal has to name why, not just that it failed"
    assert orchestrator.queued == 0


async def test_a_job_that_ends_without_an_outcome_is_still_reachable(
    tmp_path: Path,
) -> None:
    """The id must survive even the bookkeeping path that has no reason."""
    orchestrator = _orchestrator(tmp_path)
    entry = orchestrator._register("Music generation", "musica.py", ["--x"], 100)
    orchestrator._retire(entry)
    answer = orchestrator.result_text("job-1")
    assert "failed" in answer
    assert "without recording" in answer


def test_an_unknown_job_id_is_a_clear_error(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    with pytest.raises(AgentError) as failure:
        orchestrator.result_text("job-999")
    assert "compute_status" in str(failure.value)


async def test_the_queue_is_bounded(tmp_path: Path) -> None:
    """Minutes of GPU time per entry means the backlog needs a ceiling."""
    orchestrator = _orchestrator(tmp_path, queue_limit=2)

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        await asyncio.sleep(0.05)
        return 'RESULT {"ok": true}'

    orchestrator._spawn = runner  # type: ignore[method-assign]
    orchestrator.submit_video("a", frames=17)
    orchestrator.submit_video("b", frames=17)
    with pytest.raises(AgentError) as failure:
        orchestrator.submit_video("c", frames=17)
    assert "COMPUTE_QUEUE_LIMIT" in str(failure.value)


async def test_status_lists_the_queue_with_positions(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        await asyncio.sleep(0.05)
        return 'RESULT {"ok": true}'

    orchestrator._spawn = runner  # type: ignore[method-assign]
    orchestrator.submit_video("a", frames=17)
    orchestrator.submit_video("b", frames=17)
    report = orchestrator.status_text()
    assert "1. Video generation" in report
    assert "2. Video generation" in report


async def test_a_queued_job_and_a_blocking_one_share_the_card(tmp_path: Path) -> None:
    """The single slot is the policy; two entry points must not each take one."""
    orchestrator = _orchestrator(tmp_path)
    active = 0
    peak = 0

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.02)
        active -= 1
        return 'RESULT {"ok": true, "path": "salida/x.mp4"}'

    orchestrator._spawn = runner  # type: ignore[method-assign]
    orchestrator.submit_video("queued", frames=17)
    await orchestrator.generate_video("blocking", frames=17, runner=runner)
    assert peak == 1


# ------------------------------------------------------------- the VRAM hold
async def test_the_ollama_mode_means_the_same_thing_however_it_is_spelled(
    tmp_path: Path,
) -> None:
    """``on`` is what the .env file says, and it has to behave like ``auto``.

    The field was compared against the string ``auto`` directly, so a caller
    who wrote ``on`` - the spelling in the documentation and in .env - silently
    got the behaviour of ``off`` and no model was ever unloaded.

    The unload itself is stubbed. This is about the setting, and asking the
    real one would make the test pass or fail depending on whether this machine
    happens to have a model resident - which is what it just did.
    """
    attempts: list[str] = []

    async def stub_unload(root_directory: str) -> VramRelease:
        attempts.append(root_directory)
        return VramRelease(models=["qwen3.5:9b"], message="Unloaded qwen3.5:9b.")

    monkeypatched = minagent.compute.unload_ollama
    minagent.compute.unload_ollama = stub_unload
    try:
        for spelling in ("on", "auto", "ON", " on "):
            orchestrator = _orchestrator(tmp_path, ollama_mode=spelling)
            orchestrator._vram_probe = lambda: VramReading(
                total_mib=FULL_CARD, free_mib=200, used_by_others_mib=7000,
                processes=("llama-server (7000 MiB)",),
            )
            assert (await orchestrator._release_vram()).models, f"{spelling!r} did not unload"
        for spelling in ("off", "ask", ""):
            orchestrator = _orchestrator(tmp_path, ollama_mode=spelling)
            orchestrator._vram_probe = lambda: VramReading(
                total_mib=FULL_CARD, free_mib=200, used_by_others_mib=7000,
            )
            assert (await orchestrator._release_vram()).models == [], f"{spelling!r} unloaded anyway"
    finally:
        minagent.compute.unload_ollama = monkeypatched
    assert len(attempts) == 4


async def test_ollama_is_never_unloaded_unless_asked(tmp_path: Path) -> None:
    """Evicting someone's warm model is their decision, not the agent's."""
    orchestrator = _orchestrator(tmp_path, ollama_mode="off")
    assert (await orchestrator._release_vram()).models == []


async def test_a_job_that_cannot_fit_still_names_the_holder(tmp_path: Path) -> None:
    """Off means "do not touch it", not "fail vaguely"."""
    orchestrator = _orchestrator(tmp_path, ollama_mode="off")
    orchestrator._vram_probe = lambda: VramReading(
        total_mib=FULL_CARD, free_mib=200, used_by_others_mib=7000, processes=("llama-server (7000 MiB)",)
    )
    with pytest.raises(AgentError) as failure:
        await orchestrator.generate_video("x", frames=49)
    assert "llama-server" in str(failure.value)
    assert "COMPUTE_UNLOAD_OLLAMA" not in str(failure.value)


def test_the_unload_mode_parsing_is_explicit() -> None:
    from minagent.compute import parse_ollama_mode

    assert parse_ollama_mode(None) == "off"
    assert parse_ollama_mode("") == "off"
    assert parse_ollama_mode("off") == "off"
    assert parse_ollama_mode("on") == "auto"
    assert parse_ollama_mode("auto") == "auto"
    # "ask" is a real mode the orchestrator honours as "do not", so the decision
    # stays with the person watching.
    assert parse_ollama_mode("ask") == "ask"
    with pytest.raises(AgentError):
        parse_ollama_mode("maybe")


def test_asking_does_not_unload_either(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path, ollama_mode="ask")
    assert orchestrator.ollama_mode != "auto"


async def test_unload_reports_when_ollama_has_nothing_loaded(tmp_path: Path) -> None:
    """The common case must be a one-line answer, not an error."""
    from minagent.compute import unload_ollama

    fake = tmp_path / "ollama"
    fake.write_text('#!/bin/sh\necho "NAME    ID    SIZE"\n')
    fake.chmod(0o755)
    original = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{original}"
    try:
        release = await unload_ollama(str(tmp_path))
    finally:
        os.environ["PATH"] = original
    assert "no model loaded" in release.message
    # Nothing was taken, so there is nothing to put back either.
    assert release.models == []


async def test_unload_names_the_model_it_stopped(tmp_path: Path) -> None:
    from minagent.compute import unload_ollama

    fake = tmp_path / "ollama"
    # `ps` reports a loaded model; `stop` succeeds; there is no nvidia-smi here
    # so the free-memory wait ends on its own timeout.
    fake.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  ps) printf "NAME\\tID\\tSIZE\\tPROCESSOR\\tCONTEXT\\tUNTIL\\nllama3:8b\\tabc\\t6.2GB\\t100%GPU\\t"\n'
        '     printf "4 minutes from now\\n" ;;\n'
        '  stop) exit 0 ;;\n'
        'esac\n'
    )
    fake.chmod(0o755)
    original = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{tmp_path}:{original}"
    try:
        release = await unload_ollama(str(tmp_path), seconds=1.0)
    finally:
        os.environ["PATH"] = original
    assert "llama3:8b" in release.message
    assert "goes back after the job" in release.message
    # The name is what makes the model recoverable: without it the job could
    # only free the card, not put the user's session back the way it found it.
    assert release.models == ["llama3:8b"]


# ------------------------------------------------------------- putting it back


class _Cycle:
    """Records the whole evict-render-restore cycle in the order it happened.

    The card is a small state machine rather than a fixed reading, because what
    is being tested is the cycle: too tight to start, free once the model is
    out, and occupied again once it is back. A fixed reading cannot express the
    last state, which is the one the user is left looking at.
    """

    def __init__(self) -> None:
        self.events: list[str] = []
        self.model_loaded = True

    def probe(self) -> VramReading:
        used = 5600 if self.model_loaded else 900
        return VramReading(
            total_mib=FULL_CARD,
            free_mib=FULL_CARD - used,
            used_by_others_mib=used,
            processes=("llama-server (5600 MiB)",) if self.model_loaded else (),
        )

    async def unload(self, root_directory: str) -> VramRelease:
        self.model_loaded = False
        self.events.append("unload")
        return VramRelease(models=["qwen3.5:9b"], message="Unloaded qwen3.5:9b; 4700 MiB came back.")

    async def restore(self, release: VramRelease) -> str:
        if not release.models:
            return ""
        self.model_loaded = True
        self.events.append("restore")
        return f"Reloaded {', '.join(release.models)} into VRAM for the next request."


def _cycling_orchestrator(tmp_path: Path) -> tuple[ComputeOrchestrator, _Cycle]:

    cycle = _Cycle()
    orchestrator = _orchestrator(tmp_path, ollama_mode="on")
    orchestrator._vram_probe = cycle.probe

    async def release_vram() -> VramRelease:
        return await cycle.unload(orchestrator.root_directory)

    orchestrator._release_vram = release_vram  # type: ignore[method-assign]
    orchestrator._restore_vram = cycle.restore  # type: ignore[method-assign]
    return orchestrator, cycle


async def test_the_model_a_job_evicted_goes_back_when_the_job_is_done(tmp_path: Path) -> None:
    """The render must not leave the session cold; that cost lands on the next message."""
    orchestrator, cycle = _cycling_orchestrator(tmp_path)

    record = await orchestrator.generate_video(
        "a boat", frames=17, steps=12, runner=_recorder('RESULT {"ok": true, "path": "salida/x.mp4"}')
    )

    assert record.outcome == "done"
    # The order is the point: the card is freed, the job runs, and the model is
    # put back afterwards rather than left cold until someone writes a message.
    assert cycle.events == ["unload", "restore"]
    assert cycle.model_loaded


async def test_a_failed_job_still_puts_the_model_back(tmp_path: Path) -> None:
    """Leaving it unloaded would make a failure cost the user twice."""
    orchestrator, cycle = _cycling_orchestrator(tmp_path)

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        raise AgentError("the render blew up")

    with pytest.raises(AgentError, match="blew up"):
        await orchestrator.generate_video("a boat", frames=17, steps=12, runner=runner)

    assert cycle.events == ["unload", "restore"]
    assert cycle.model_loaded


async def test_a_job_that_fits_does_not_disturb_a_warm_model(tmp_path: Path) -> None:
    """Nothing is taken, so nothing is put back: the card was never a problem."""
    orchestrator, cycle = _cycling_orchestrator(tmp_path)
    orchestrator._vram_probe = lambda: VramReading(total_mib=FULL_CARD, free_mib=FULL_CARD - 200, used_by_others_mib=200)

    await orchestrator.generate_video(
        "a boat", frames=17, steps=12, runner=_recorder('RESULT {"ok": true, "path": "salida/x.mp4"}')
    )

    assert cycle.events == []


async def test_a_reload_that_ollama_refuses_is_reported_not_raised() -> None:
    from minagent.compute import reload_ollama

    # A refused reload is the same situation as a server that is down: a model
    # that could not be put back makes the next message slower, and must not
    # throw away a render that already worked. The server is stood in for, so
    # the test cannot reach the real one - on a machine that has the model, a
    # live call would load it into VRAM to prove a point.
    message = await reload_ollama(
        ["qwen3.5:9b"], seconds=1.0, transport=httpx.MockTransport(lambda request: httpx.Response(500))
    )
    assert "qwen3.5:9b" in message
    assert "on the next request" in message or "on demand" in message


async def test_an_embedding_model_is_left_cold_instead_of_reported_as_a_failure() -> None:
    from minagent.compute import reload_ollama

    # The memory de-duplication loads nomic-embed-text for a few hundred
    # milliseconds and Ollama keeps it for minutes. Putting it "back" is not a
    # thing: Ollama answers 400 to a generate request for a model that cannot
    # generate, and the restore would report a failure nobody can act on.
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["embedding"]})
        return httpx.Response(400, json={"error": "does not support generate"})

    message = await reload_ollama(
        ["nomic-embed-text:latest"], seconds=1.0, transport=httpx.MockTransport(handler)
    )

    assert message == ""
    assert asked == ["/api/show"], "it asked Ollama to generate with a model that cannot generate"


async def test_a_model_that_generates_is_still_reloaded_after_being_asked_what_it_can_do() -> None:
    from minagent.compute import reload_ollama

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"]})
        return httpx.Response(200, json={"response": ""})

    message = await reload_ollama(
        ["qwen3.5:9b-q4_K_M"], seconds=1.0, transport=httpx.MockTransport(handler)
    )

    assert "qwen3.5:9b-q4_K_M" in message


async def test_which_models_ollama_is_holding() -> None:
    from minagent.compute import ollama_resident_models

    listing = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"models": [{"name": "gemma4:12b-q3km"}, {"name": "x"}]})
    )
    assert await ollama_resident_models(transport=listing) == ["gemma4:12b-q3km", "x"]

    # A server that is not there and a server holding nothing are the same thing
    # to the caller: nobody is in the way.
    for broken in (
        httpx.MockTransport(lambda request: httpx.Response(500)),
        httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("down"))),
        httpx.MockTransport(lambda request: httpx.Response(200, content=b"not json")),
    ):
        assert await ollama_resident_models(seconds=1.0, transport=broken) == []


async def test_reloading_nothing_says_nothing() -> None:
    from minagent.compute import reload_ollama

    assert await reload_ollama([]) == ""


def test_the_config_exposes_the_new_settings(tmp_path: Path) -> None:
    config = load_configuration(
        str(tmp_path),
        str(tmp_path),
        {
            "OPENAI_API_KEY": "test",
            "OPENAI_MODEL": "m",
            "COMPUTE_ENABLED": "on",
            "COMPUTE_UNLOAD_OLLAMA": "on",
            "COMPUTE_QUEUE_LIMIT": "3",
        },
    )
    assert config.compute_unload_ollama == "auto"
    assert config.compute_queue_limit == 3


def test_the_two_new_tools_are_in_the_capability(tmp_path: Path) -> None:
    entries = build_builtin_capabilities(terminal_mode="off", compute_enabled=True)
    compute = next(entry for entry in entries if entry.name == "compute")
    assert QUEUE_TOOL_NAME in compute.tool_names
    assert RESULT_TOOL_NAME in compute.tool_names
    # The guidance has to tell the model to queue rather than block, or the
    # queue exists and nothing ever uses it.
    assert "queue_job" in compute.guidance
    assert "compute_result" in compute.guidance


async def test_every_advertised_tool_is_implemented_by_the_mcp_server() -> None:
    """A tool in ``tools/list`` that raises "Unknown tool" is worse than absent.

    The server builds its list from ``create_compute_tools`` while dispatching
    through its own ``if`` chain, so adding a tool to one and forgetting the
    other advertises something that cannot be called. That is exactly what
    happened: ``queue_job`` and ``compute_result`` were listed and refused.
    """
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("compute_mcp", root / ".agents" / "mcp" / "compute" / "index.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    server = module.ComputeServer(str(root))
    advertised = {schema["function"]["name"] for schema in server.tools()}

    # Every advertised tool must be dispatchable. A bad argument is fine and
    # expected; "Unknown tool" is not.
    for name in advertised:
        try:
            await server.call(name, {})
        except ValueError as error:
            assert "Unknown tool" not in str(error), f"{name} is advertised but not implemented"
        except Exception:
            pass  # Rejected on its arguments, which is the dispatch working.


async def test_the_mcp_server_queues_and_collects() -> None:
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("compute_mcp_q", root / ".agents" / "mcp" / "compute" / "index.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    server = module.ComputeServer(str(root))

    async def runner(script: Path, argv: list[str], timeout: int) -> str:
        return 'RESULT {"ok": true, "path": "salida/q.wav"}'

    # The card is not under the test's control - an Ollama model may be
    # resident on it - so the reading is fixed rather than measured. Otherwise
    # this test fails for a reason that has nothing to do with the queue.
    server.orchestrator._vram_probe = lambda: VramReading(
        total_mib=FULL_CARD, free_mib=FULL_CARD
    )
    server.orchestrator._spawn = runner  # type: ignore[method-assign]
    answer = await server.call(QUEUE_TOOL_NAME, {"kind": "music", "prompt": "jazz", "seconds": 5})
    assert "job-1" in answer
    for _ in range(100):
        if "finished" in await server.call(RESULT_TOOL_NAME, {"job_id": "job-1"}):
            break
        await asyncio.sleep(0.01)
    assert "salida/q.wav" in await server.call(RESULT_TOOL_NAME, {"job_id": "job-1"})


def test_the_video_backend_rejects_incompatible_transformers() -> None:
    """The guard has to return a reason, not print one and carry on.

    It was written as ``return fail(...)`` and called as a bare statement, so
    the message was emitted and then ignored: the run still reached the
    tokenizer and failed 80 seconds later with the traceback the check exists
    to prevent. The versions that break it are 5.x and 4.55+, both found by
    running the real pipeline.
    """
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("video_ltx_mod", root / "scripts" / "compute" / "video_ltx.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import transformers

    version = transformers.__version__
    reason = module._check_transformers_compatibility()
    if version.startswith("5."):
        assert reason, f"transformers {version} is incompatible and must be refused"
        assert "transformers==4.49.0" in reason
    else:
        # Whatever this environment runs, the check must not block a version
        # that works: a guard that always fires would refuse every render.
        assert not reason or "necesita" in reason


def test_the_video_backend_names_the_version_that_works() -> None:
    """Whatever is refused, the message has to say what to install instead."""
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location("video_ltx_mod2", root / "scripts" / "compute" / "video_ltx.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = inspect.getsource(module._check_transformers_compatibility)
    assert "4.49.0" in source, "the message must name the version that works"
    assert "sys.executable" in source, "the install line must use the running interpreter"


async def test_the_backend_runs_under_this_interpreter_not_bare_python3(tmp_path: Path) -> None:
    """The venv has torch and the system python3 often does not.

    Spawning a bare ``python3`` would report "No module named torch" for a
    dependency that was installed all along, so the backend inherits the
    interpreter running MinAgent.
    """
    import sys

    marker = tmp_path / "interpreter.txt"
    scripts = tmp_path / "scripts" / "compute"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "musica.py").write_text(
        "import sys, pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text(sys.executable)\n"
        'print("RESULT {\\"ok\\": true}")\n'
    )
    orchestrator = ComputeOrchestrator(root_directory=str(tmp_path))
    orchestrator._vram_probe = lambda: VramReading(total_mib=FULL_CARD, free_mib=FULL_CARD)
    await orchestrator._spawn(scripts / "musica.py", [], 60)
    assert marker.read_text().strip() == sys.executable

    # An explicit override still wins, for a backend that needs another env.
    os.environ["COMPUTE_PYTHON"] = sys.executable
    try:
        marker.unlink()
        await orchestrator._spawn(scripts / "musica.py", [], 60)
        assert marker.read_text().strip() == sys.executable
    finally:
        del os.environ["COMPUTE_PYTHON"]


async def test_a_job_that_times_out_is_stopped_not_left_running(tmp_path: Path) -> None:
    """A render that overruns its budget must be killed, not left holding the card.

    This runs a real subprocess rather than an injected runner, because the
    timeout and the kill are the thing under test: a fake runner would return
    instantly and prove nothing about either.
    """
    scripts = tmp_path / "scripts" / "compute"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "musica.py").write_text("import time\ntime.sleep(120)\n")
    orchestrator = ComputeOrchestrator(root_directory=str(tmp_path), job_timeout_seconds=1)
    orchestrator._vram_probe = lambda: VramReading(total_mib=FULL_CARD, free_mib=FULL_CARD)

    started = time.time()
    with pytest.raises(AgentError) as failure:
        await orchestrator._spawn(scripts / "musica.py", [], 1)
    elapsed = time.time() - started
    assert "did not finish" in str(failure.value)
    assert elapsed < 15, f"the process was not stopped promptly ({elapsed:.0f}s)"
    assert not orchestrator.busy
