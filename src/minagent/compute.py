"""Scheduling GPU work on a card that cannot hold two heavy models at once.

An RTX 4060 Laptop has 8188 MiB of VRAM. Voice interaction - STT plus TTS -
fits in well under 2.5 GB, so it can stay resident and answer instantly. A video
or music model needs several GB on its own, and two of them at once is a
physical impossibility, not a slow configuration. So the policy is not "try it
and see":

* The voice engines are **resident**. They are loaded once and kept.
* Every heavy job is **serialized**. One at a time, in submission order, behind
  an ``asyncio.Lock``, so two diffusers pipelines never hold VRAM together.
* A heavy job runs in a **subprocess** and gets the system's 30 GB of RAM as
  backing store, with layer offloading, instead of holding weights in VRAM.
* Free VRAM is **measured, not assumed**. Another program - an Ollama server, a
  browser, a compositor - may be holding most of the card. On this machine
  ``llama-server`` alone was holding 6390 of 8188 MiB, which is the difference
  between a job that runs and one that dies with OOM. Every heavy job is
  checked against real free VRAM first, and refused with the numbers in the
  error rather than being attempted and crashing.

The subprocess boundary is what makes the last point true. An OOM inside a
library raises a traceback the model cannot act on; refusing up front, with the
measured free MiB and what the job needs, is something it can report and retry
with smaller settings. The same boundary is why the resident voice engines live
in a long-lived worker process and not in this one: the agent's own process
never imports torch, so it can never be the reason a job cannot get memory.

The heavy backends are separate scripts under ``scripts/compute/``, invoked as
subprocesses. Keeping them out of the package means a missing diffusers install
is a missing optional dependency of a script, not an import error that stops
the agent from starting at all.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import AgentError

SPEAK_TOOL_NAME = "speak_text"
TRANSCRIBE_TOOL_NAME = "transcribe_audio"
VIDEO_TOOL_NAME = "generate_video"
MUSIC_TOOL_NAME = "generate_music"
STATUS_TOOL_NAME = "compute_status"

COMPUTE_CAPABILITY_NAME = "compute"

DEFAULT_VRAM_TOTAL_MIB = 8188
"""A 4060 Laptop, which is the card this was written against."""

VOICE_RESERVE_MIB = 2560
"""Held back for the resident STT + TTS pair, and never lent to a heavy job.

Kokoro in fp16 is a few hundred MiB and whisper.cpp small/base is under a
gigabyte; 2.5 GB is the figure from the brief and leaves headroom for the
CUDA context each engine opens on its own.
"""

VRAM_HEADROOM_MIB = 512
"""Never promise the last megabytes.

An 8 GB card that reports zero free VRAM is normal: the allocator keeps a
reserved pool, and a request that fits on paper still fails at 96 MiB. Asking
for a few hundred MiB less than is there is what turns "usually works" into
"works", and it is the same lesson as ``PYTORCH_CUDA_ALLOC_CONF`` in the
generation scripts.
"""

DEFAULT_JOB_TIMEOUT_SECONDS = 30 * 60
"""A video clip on this card is minutes, not seconds. The default clears it."""

VOICE_TIMEOUT_SECONDS = 120
"""STT and TTS are interactive. Past two minutes something is wrong, not slow."""

MAX_TEXT_CHARS = 5_000
"""Kokoro's context is a sentence or two; longer input is chunked by the worker."""

MAX_PROMPT_CHARS = 2_000
MAX_AUDIO_BYTES = 100 * 1024 * 1024

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".mp4", ".webm", ".aac", ".aiff"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".gif"}

OFFLOAD_MODES = ("group", "sequential", "model")
"""How a heavy pipeline hands layers between VRAM and system RAM.

``group`` is the default and the one that fits an 8 GB card: it keeps a few
blocks resident and streams the rest. ``sequential`` is slower and asks for
less. ``model`` leaves a whole submodel on the GPU and is what does not fit -
it is offered because some pipelines are shaped for it, not because it is
recommended here.
"""

VIDEO_VRAM_ESTIMATE_MIB = {
    "group": 5200,
    "sequential": 3800,
    "model": 7200,
}
"""VRAM a video job needs per offload mode, in MiB.

Pessimistic on purpose: an estimate that under-counts does not fail the check,
it fails later as an OOM in the middle of a multi-minute render, which costs
much more than a refused call.
"""

MUSIC_VRAM_ESTIMATE_MIB = {"small": 2800, "medium": 6500, "large": 9500}
"""VRAM a MusicGen job needs per model size.

Only ``small`` fits alongside the resident voice engines. The other two are
here so the refusal can say by how much the card is short instead of just "no".
"""

DEFAULT_VIDEO_FRAMES = 49
DEFAULT_VIDEO_STEPS = 40

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def estimate_video_vram(frames: int, offload: str) -> int:
    """VRAM a video render needs, scaling with frames.

    Frames drive the attention and therefore the memory; steps do not, they
    just recompute over the same sequence. The scale factor is measured against
    49 frames, which is what the numbers above describe.
    """
    base = VIDEO_VRAM_ESTIMATE_MIB.get(offload, VIDEO_VRAM_ESTIMATE_MIB["group"])
    scale = 0.55 + 0.45 * (max(frames, 9) / DEFAULT_VIDEO_FRAMES)
    return int(base * min(scale, 1.9))


def parse_result_line(output: str) -> str:
    """Pull the ``RESULT {json}`` line out of a backend's stdout.

    The backends emit one machine-readable line and the rest is human
    progress, because library logging would otherwise be indistinguishable
    from the answer. Falling back to the raw output keeps a failure legible
    when the script died before it got to emit anything.
    """
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT "):
            try:
                payload = json.loads(line[len("RESULT ") :])
            except json.JSONDecodeError:
                return output
            if isinstance(payload, dict):
                if not payload.get("ok"):
                    raise AgentError(str(payload.get("error", "The job failed.")))
                produced = payload.get("path") or ""
                if payload.get("poster"):
                    produced += f"\nPoster: {payload['poster']}"
                return produced or json.dumps(payload, ensure_ascii=False)
    return output


def parse_result_object(output: str) -> dict[str, Any]:
    """The parsed ``RESULT`` payload, or an empty dict when there is none."""
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT "):
            try:
                payload = json.loads(line[len("RESULT ") :])
            except json.JSONDecodeError:
                return {}
            if isinstance(payload, dict):
                return payload
    return {}

# nvidia-smi prints "8188 MiB" and, on some builds, "8188MiB". One pattern for both.
_MIB = re.compile(r"(\d+(?:\.\d+)?)\s*MiB", re.IGNORECASE)


def _sanitize_name(value: str, fallback: str) -> str:
    """Reduce a requested file name to something that cannot climb out of a directory."""
    base = os.path.basename((value or "").replace("\\", "/")).strip()
    base = _SAFE_NAME.sub("-", base).strip("-.")
    if not base or base in {".", ".."}:
        return fallback
    return base[:120]


def _require_text(value: Any, tool: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentError(f"{tool} needs non-empty text.")
    text = value.strip()
    if len(text) > limit:
        raise AgentError(f"{tool} text is over the {limit} character limit.")
    return text


@dataclass(frozen=True)
class VramReading:
    """What the card actually looks like right now, in MiB."""

    total_mib: int
    free_mib: int
    used_by_others_mib: int = 0
    processes: tuple[str, ...] = ()

    @property
    def available_for_heavy_mib(self) -> int:
        """Free VRAM a heavy job may count on, after the voice reserve and headroom.

        This is the number that decides whether a job is started, so it is
        computed in one place: a check that disagrees with the planner is how
        an OOM gets through.
        """
        return max(0, self.free_mib - VOICE_RESERVE_MIB - VRAM_HEADROOM_MIB)


@dataclass
class JobRecord:
    """One submitted heavy job, for the status tool to report on."""

    kind: str
    detail: str
    started_at: float = 0.0
    finished_at: float = 0.0
    outcome: str = "queued"
    output: str = ""

    @property
    def seconds(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.finished_at or 0.0) - self.started_at


@dataclass
class ComputeOrchestrator:
    """Serializes heavy GPU jobs and keeps the voice engines resident.

    ``runner`` is the seam that makes this testable: it is the only thing that
    starts a process, so the scheduling, budgeting and refusal logic can be
    exercised without a GPU, a model, or a 30-minute wait.
    """

    root_directory: str
    vram_total_mib: int = DEFAULT_VRAM_TOTAL_MIB
    job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT_SECONDS
    voice_timeout_seconds: int = VOICE_TIMEOUT_SECONDS
    output_dirname: str = "salida"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _history: list[JobRecord] = field(default_factory=list, repr=False)
    _vram_probe: Any = None
    """``Callable[[], VramReading]``; replaced in tests. ``None`` means nvidia-smi."""
    _own_pids: set[int] = field(default_factory=set, repr=False)
    """PIDs of the backends this orchestrator started.

    They show up in nvidia-smi while they run, and a status report that told
    the user to go and kill the video they just asked for would be worse than
    useless. Their memory is real, so it still counts against the free figure;
    it is only left out of the list of other people's processes.
    """

    # ------------------------------------------------------------------ paths
    @property
    def output_directory(self) -> Path:
        """Where generated media lands: the same ``salida/`` downloads use."""
        return Path(self.root_directory) / self.output_dirname

    def _script(self, name: str) -> Path:
        """Resolve a backend script, with an error that says how to get it."""
        path = Path(self.root_directory) / "scripts" / "compute" / name
        if not path.is_file():
            raise AgentError(
                f"The {name} backend is not installed (expected {path}). It ships with the project; "
                "run it from the repository root, or set COMPUTE_SCRIPTS_DIR if it lives elsewhere."
            )
        return path

    def _scripts_dir_override(self) -> Path | None:
        raw = (os.environ.get("COMPUTE_SCRIPTS_DIR") or "").strip()
        return Path(raw) if raw else None

    def resolve_script(self, name: str) -> Path:
        """The backend script, honouring ``COMPUTE_SCRIPTS_DIR`` before the default."""
        override = self._scripts_dir_override()
        if override is not None:
            candidate = override / name
            if candidate.is_file():
                return candidate
        return self._script(name)

    def output_path(self, requested: str, suffix: str, stem: str) -> Path:
        """A public alias: the MCP server resolves output names through here too."""
        return self._output_path(requested, suffix, stem)

    # ------------------------------------------------------------------ video
    async def generate_video(
        self,
        prompt: str,
        *,
        frames: int = 49,
        steps: int = 40,
        offload: str = "group",
        name: str = "",
        runner: Any = None,
    ) -> JobRecord:
        """Render a clip with LTX-Video, alone, after checking it fits.

        Both the agent tool and the MCP server come through here, so the VRAM
        estimate the refusal uses is the same one the script reports - two
        copies of that number would drift, and the copy that drifted would be
        the one letting an OOM through.
        """
        text = _require_text(prompt, VIDEO_TOOL_NAME, MAX_PROMPT_CHARS)
        if offload not in OFFLOAD_MODES:
            raise AgentError(f"offload must be one of: {', '.join(OFFLOAD_MODES)}.")
        if frames < 9:
            raise AgentError("A clip needs at least 9 frames.")
        output = self._output_path(name, "mp4", "ltx")
        argv = [
            "--prompt", text,
            "--frames", str(frames),
            "--steps", str(steps),
            "--offload", offload,
            "--salida", str(output),
        ]
        return await self.run_heavy(
            "Video generation",
            "video_ltx.py",
            argv,
            needed_mib=estimate_video_vram(frames, offload),
            runner=runner,
        )

    # ----------------------------------------------------------------- music
    async def generate_music(
        self,
        prompt: str,
        *,
        seconds: int = 15,
        model: str = "small",
        name: str = "",
        runner: Any = None,
    ) -> JobRecord:
        """Generate music with MusicGen, alone, after checking it fits.

        ``small`` is the only size that coexists with the resident voice
        engines on an 8 GB card, so it is both the default and the reason the
        larger models are refused with a number rather than attempted.
        """
        text = _require_text(prompt, MUSIC_TOOL_NAME, MAX_PROMPT_CHARS)
        if model not in MUSIC_VRAM_ESTIMATE_MIB:
            raise AgentError(f"model must be one of: {', '.join(sorted(MUSIC_VRAM_ESTIMATE_MIB))}.")
        if seconds < 1 or seconds > 120:
            raise AgentError("Music must be between 1 and 120 seconds.")
        output = self._output_path(name, "wav", "musica")
        argv = [
            "--prompt", text,
            "--segundos", str(seconds),
            "--modelo", model,
            "--salida", str(output),
        ]
        return await self.run_heavy(
            "Music generation",
            "musica.py",
            argv,
            needed_mib=MUSIC_VRAM_ESTIMATE_MIB[model],
            runner=runner,
        )

    def _output_path(self, requested: str, suffix: str, stem: str) -> Path:
        """A writable path inside the output directory, never outside it."""
        name = _sanitize_name(requested, "")
        if not name:
            name = f"{stem}.{suffix.lstrip('.')}"
        elif not os.path.splitext(name)[1]:
            name = f"{name}.{suffix.lstrip('.')}"
        directory = self.output_directory
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name

    # ------------------------------------------------------------------- vram
    def read_vram(self) -> VramReading:
        """Measure the card. Falls back to the configured total when nvidia-smi is absent.

        A host with no NVIDIA GPU still gets a usable answer - the budget maths
        is the same - rather than a hard failure on every call.
        """
        if self._vram_probe is not None:
            return self._vram_probe()
        return probe_vram(self.vram_total_mib, ignore_pids=self._own_pids)

    def require_headroom(self, kind: str, needed_mib: int) -> VramReading:
        """Refuse a heavy job that cannot fit, naming the numbers.

        The message is the whole point. An OOM deep in diffusers is a stack
        trace the model can only report as failure; this says what is free,
        what the job wanted, and what to give up, so the retry is a decision
        rather than another identical attempt.
        """
        reading = self.read_vram()
        if needed_mib <= reading.available_for_heavy_mib:
            return reading
        detail = (
            f"{kind} needs about {needed_mib} MiB of VRAM, but only {reading.free_mib} MiB is free and "
            f"{VOICE_RESERVE_MIB} MiB is held for the resident voice engines, leaving "
            f"{reading.available_for_heavy_mib} MiB usable."
        )
        if reading.used_by_others_mib > 0:
            holders = ", ".join(reading.processes[:3]) or "another process"
            detail += f" {holders} is holding {reading.used_by_others_mib} MiB."
        # The advice has to match the job: telling someone who asked for music
        # to reduce the frame count sends them off to fix the wrong thing.
        if kind.lower().startswith("music"):
            detail += (
                " Free it before retrying, or use the small model and a shorter duration; the larger "
                "MusicGen models need more VRAM than this card has free with voice resident."
            )
        else:
            detail += (
                " Free it before retrying, or shrink the job: fewer frames (not fewer steps, which only "
                "cost time), or --offload sequential for the least VRAM."
            )
        raise AgentError(detail)

    # ------------------------------------------------------------------- jobs
    def _record(self, record: JobRecord) -> JobRecord:
        self._history.append(record)
        del self._history[:-20]
        return record

    @property
    def busy(self) -> bool:
        """Whether a heavy job is running, so the model does not pile work up."""
        return self._lock.locked()

    async def run_heavy(
        self,
        kind: str,
        script_name: str,
        argv: list[str],
        *,
        needed_mib: int,
        timeout_seconds: int | None = None,
        runner: Any = None,
    ) -> JobRecord:
        """Run one heavy job alone, checking free VRAM before starting it.

        The lock is the whole scheduling policy: voice stays responsive during
        a twenty-minute video because voice never asks for it, and two heavy
        jobs never overlap because this is the only path that takes it.
        """
        self.require_headroom(kind, needed_mib)
        script = self.resolve_script(script_name)
        async with self._lock:
            record = self._record(JobRecord(kind=kind, detail=" ".join(argv[:1])))
            loop = asyncio.get_running_loop()
            record.started_at = loop.time()
            record.outcome = "running"
            try:
                result = await (runner or self._spawn)(
                    script,
                    argv,
                    timeout_seconds if timeout_seconds is not None else self.job_timeout_seconds,
                )
            except asyncio.CancelledError:
                record.outcome = "cancelled"
                record.finished_at = loop.time()
                raise
            except AgentError:
                record.outcome = "failed"
                record.finished_at = loop.time()
                raise
            record.finished_at = loop.time()
            record.outcome = "done"
            record.output = parse_result_line(result)
            return record

    async def _spawn(self, script: Path, argv: list[str], timeout_seconds: int) -> str:
        """Start a backend script and collect its output.

        The child gets the same allocator setting as the hand-written generation
        scripts: on an 8 GB card fragmentation, not capacity, is what turns a
        request that should fit into an OOM, and it has to be set before CUDA
        initialises.
        """
        environment = dict(os.environ)
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # The backend runs under the interpreter that is running MinAgent, not
        # whatever `python3` happens to be on PATH. Those are frequently
        # different environments - on this machine the venv has torch and the
        # system interpreter does not - and a bare `python3` would fail with
        # "No module named torch" while the dependency was installed all along.
        interpreter = os.environ.get("COMPUTE_PYTHON", "").strip() or sys.executable or "python3"
        process = await asyncio.create_subprocess_exec(
            interpreter,
            str(script),
            *argv,
            cwd=self.root_directory,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        self._own_pids.add(process.pid)
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except TimeoutError as error:
            _terminate(process)
            raise AgentError(
                f"The job did not finish within {timeout_seconds}s and was stopped. Video on this card "
                "takes minutes; raise COMPUTE_JOB_TIMEOUT_SECONDS or shrink the job."
            ) from error
        finally:
            self._own_pids.discard(process.pid)
        text = stdout.decode("utf-8", "replace") if stdout else ""
        if process.returncode != 0:
            tail = "\n".join(text.strip().splitlines()[-12:])
            raise AgentError(
                f"The generation job failed (exit {process.returncode}).\n{tail or 'No output was produced.'}"
            )
        return text.strip()

    # ------------------------------------------------------------------ voice
    async def speak(self, text: str, voice: str = "", speed: float = 1.0, runner: Any = None) -> str:
        """Speak text with the resident Kokoro engine and return the wav path.

        This path deliberately does not take the heavy lock. That is the entire
        point of keeping TTS resident: a reply can be spoken while a video is
        still rendering, because the video lives in another process and Kokoro
        owns its own small slice of VRAM that the heavy planner refuses to
        touch.
        """
        spoken = _require_text(text, SPEAK_TOOL_NAME, MAX_TEXT_CHARS)
        if speed <= 0:
            raise AgentError("speed must be greater than 0.")
        output = self._output_path("", "wav", "hablaje")
        argv = ["--texto", spoken, "--salida", str(output)]
        if voice:
            argv += ["--voz", voice]
        if speed != 1.0:
            argv += ["--velocidad", f"{speed:g}"]
        return parse_result_line(await self._voice("speak", argv, runner))

    async def transcribe(self, path: str, language: str = "", runner: Any = None) -> str:
        """Transcribe an audio or video file with the resident whisper.cpp engine."""
        resolved = self._resolve_media(path, AUDIO_SUFFIXES | VIDEO_SUFFIXES, TRANSCRIBE_TOOL_NAME)
        argv = ["--audio", resolved]
        if language:
            argv += ["--idioma", language]
        raw = await self._voice("transcribe", argv, runner)
        # parse_result_line raises on ok:false, so the text is only read from a
        # result that already succeeded.
        parse_result_line(raw)
        return str(parse_result_object(raw).get("text", raw))

    async def _voice(self, operation: str, argv: list[str], runner: Any) -> str:
        """Run a voice operation in its own process.

        It goes through the same spawn as a heavy job but not through the heavy
        lock, and that is the whole distinction: the voice engines stay
        resident and answer while a render is in flight, and a CUDA failure in
        TTS cannot take the agent's own process down with it.

        ``operation`` is only for the error message, so a failure says which of
        the two engines was missing rather than naming a script.
        """
        script = self.resolve_script("voz.py")
        try:
            return await (runner or self._spawn)(script, argv, self.voice_timeout_seconds)
        except AgentError as error:
            raise AgentError(
                f"{error} The voice engines are optional dependencies; see "
                ".agents/skills/compute-gpu/SKILL.md for the install commands."
            ) from error

    def _resolve_media(self, value: Any, allowed: set[str], tool: str) -> str:
        """Resolve a media path inside the workspace, refusing anything else."""
        if not isinstance(value, str) or not value.strip():
            raise AgentError(f"{tool} needs a path to a media file.")
        candidate = value.strip()
        root = os.path.realpath(self.root_directory)
        resolved = os.path.realpath(candidate if os.path.isabs(candidate) else os.path.join(root, candidate))
        if resolved != root and not resolved.startswith(root + os.sep):
            raise AgentError(f"{tool} only reads files inside the workspace.")
        if not os.path.isfile(resolved):
            raise AgentError(f"No file at {candidate}. List the directory to see what is there.")
        suffix = os.path.splitext(resolved)[1].lower()
        if suffix not in allowed:
            raise AgentError(
                f"{suffix or candidate} is not a supported media type. Supported: "
                f"{', '.join(sorted(allowed))}."
            )
        size = os.path.getsize(resolved)
        if size > MAX_AUDIO_BYTES:
            raise AgentError(f"{candidate} is {size // (1024 * 1024)} MB, over the limit.")
        return resolved

    # ----------------------------------------------------------------- report
    def status_text(self) -> str:
        """A report of the card and the queue, for the model to reason with."""
        reading = self.read_vram()
        lines = [
            "GPU compute status.",
            f"VRAM: {reading.free_mib} MiB free of {reading.total_mib} MiB total.",
            f"Reserved for resident voice engines: {VOICE_RESERVE_MIB} MiB "
            f"(+{VRAM_HEADROOM_MIB} MiB headroom).",
            f"Usable by a heavy job right now: {reading.available_for_heavy_mib} MiB.",
        ]
        if reading.used_by_others_mib > 0:
            holders = ", ".join(reading.processes[:5]) or "an unnamed process"
            lines.append(f"Holding VRAM outside this agent: {holders} ({reading.used_by_others_mib} MiB).")
        else:
            lines.append("Nothing outside this agent is holding VRAM.")
        lines.append(f"Queue: {'busy' if self.busy else 'idle'}; heavy jobs run one at a time.")
        if self._history:
            lines.append("Recent heavy jobs:")
            for record in self._history[-5:]:
                output = f" -> {os.path.basename(record.output)}" if _looks_like_path(record.output) else ""
                lines.append(
                    f"  - {record.kind}: {record.outcome} in {record.seconds:.0f}s{output}"
                )
        return "\n".join(lines)


def _looks_like_path(value: str) -> bool:
    return bool(value) and ("/" in value or "\\" in value) and "\n" not in value


def _terminate(process: Any) -> None:
    """Stop a subprocess and its children, best effort."""
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        return


def probe_vram(default_total_mib: int, ignore_pids: set[int] | None = None) -> VramReading:
    """Ask ``nvidia-smi`` what the card holds, and who is holding it.

    Reading the per-process table matters as much as the free figure: on this
    machine an Ollama server was sitting on 6390 of 8188 MiB, and the useful
    thing to tell the model is not "not enough memory" but "llama-server is
    using 6.2 GB, free it or use a smaller job".

    ``ignore_pids`` is this orchestrator's own backends. Their memory is real
    and already reflected in the free figure, so it is only left out of the
    "someone else is holding VRAM" list, which is a list of things to go and
    fix.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return VramReading(total_mib=default_total_mib, free_mib=default_total_mib)

    def run(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                [binary, *arguments], capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout if completed.returncode == 0 else ""

    summary = run("--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits")
    numbers = re.findall(r"\d+", summary)
    if len(numbers) < 2:
        return VramReading(total_mib=default_total_mib, free_mib=default_total_mib)
    total_mib, free_mib = int(numbers[0]), int(numbers[1])

    skip = ignore_pids or set()
    used_by_others = 0
    names: list[str] = []
    table = run("--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits")
    for line in table.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        pid, name = parts[0], parts[1]
        amount = _MIB.search(parts[2])
        if not name or amount is None:
            continue
        if pid.isdigit() and int(pid) in skip:
            continue
        names.append(f"{os.path.basename(name)} ({int(float(amount.group(1)))} MiB)")
        used_by_others += int(float(amount.group(1)))
    return VramReading(
        total_mib=total_mib,
        free_mib=free_mib,
        used_by_others_mib=used_by_others,
        processes=tuple(names),
    )


# --------------------------------------------------------------------- tools
def create_compute_tools() -> list[dict[str, Any]]:
    """Tool schemas for the ``compute`` capability."""
    return [
        {
            "type": "function",
            "function": {
                "name": SPEAK_TOOL_NAME,
                "description": (
                    "Speak text out loud with the local Kokoro voice and save a wav in salida/. The engine "
                    "stays resident, so this is fast and works even while a video or music job is "
                    "rendering. Use it to say something the user should hear. Returns the wav path, not "
                    "the audio itself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "What to say, in the language to speak."},
                        "voice": {
                            "type": "string",
                            "description": "Optional Kokoro voice name, e.g. 'af_heart' or 'ef_dora'.",
                        },
                        "speed": {
                            "type": "number",
                            "description": "Speaking rate around 1.0; 1.1 is faster, 0.9 slower.",
                        },
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TRANSCRIBE_TOOL_NAME,
                "description": (
                    "Transcribe a speech or video file in the workspace with the local whisper.cpp engine "
                    "and return the text. The engine stays resident, so this is fast. Put the media file "
                    "in the workspace first if it is not there yet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Workspace-relative path to the audio or video file.",
                        },
                        "language": {
                            "type": "string",
                            "description": "Optional language code such as 'es' or 'en'; omit to autodetect.",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": VIDEO_TOOL_NAME,
                "description": (
                    "Generate a short video clip from a text prompt with LTX-Video and save an mp4 in "
                    "salida/. This is heavy: on this card it takes minutes, it runs alone, and the voice "
                    "engines stay resident so speech still works while it runs. If free VRAM is too low "
                    "the call is refused with the numbers rather than crashing - free the memory, or ask "
                    "for fewer frames."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "What the video should show."},
                        "frames": {
                            "type": "integer",
                            "description": "Frame count; more frames means more VRAM and longer renders. 25 is fast, 49 default.",
                        },
                        "steps": {
                            "type": "integer",
                            "description": "Diffusion steps. More is better quality but slower, not smaller in VRAM.",
                        },
                        "offload": {
                            "type": "string",
                            "enum": list(OFFLOAD_MODES),
                            "description": (
                                "How layers move between VRAM and RAM. 'group' fits an 8 GB card and is "
                                "the default; 'sequential' needs the least VRAM and is slowest."
                            ),
                        },
                        "name": {"type": "string", "description": "Optional output file name."},
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": MUSIC_TOOL_NAME,
                "description": (
                    "Generate music or ambience from a text prompt with MusicGen and save a wav in "
                    "salida/. Heavy, like video: it runs alone, the voice engines stay resident, and it "
                    "is refused up front when free VRAM is too low."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "What to hear, e.g. 'lo-fi hip hop, warmRhodes, relaxed'.",
                        },
                        "seconds": {
                            "type": "integer",
                            "description": "Duration; 10 is quick, 30 default. Longer needs more time, not much more VRAM.",
                        },
                        "name": {"type": "string", "description": "Optional output file name."},
                    },
                    "required": ["prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": STATUS_TOOL_NAME,
                "description": (
                    "Report free VRAM, what is holding it, the voice reserve, and recent heavy jobs. Call "
                    "this before a heavy job if a previous one was refused for lack of memory - it names "
                    "the process to free."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def format_speak_result(path: str) -> str:
    """Report the wav that Kokoro wrote, in the workspace-relative form the model uses."""
    return f"Spoke the text to {path}. The audio is in that file; it is not in the conversation."


def format_transcribe_result(path: str, text: str) -> str:
    """The transcript, with the source named so the model can cite it."""
    return f"Transcript of {path}:\n\n{text.strip()}"


def format_heavy_result(record: JobRecord, produced: str) -> str:
    """Report a finished heavy job, including what it left in ``salida/``."""
    return (
        f"{record.kind} finished in {record.seconds:.0f}s.\n{produced}".strip()
    )
