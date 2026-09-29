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
QUEUE_TOOL_NAME = "queue_job"
RESULT_TOOL_NAME = "compute_result"

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

OLLAMA_UNLOAD_MODES = ("off", "ask", "auto")
"""How far the orchestrator may go to evict a resident Ollama model.

``off`` never touches it. ``ask`` is not implemented here - the decision
belongs to whoever is watching - and the orchestrator uses ``off`` for it, so
the mode exists to be read rather than acted on. ``auto`` unloads without
asking, which is why it is not the default: unloading a model drops the
server's KV cache and the next request pays to load it again.
"""

OFFLOAD_MODES = ("sequential", "group", "model")
"""How a heavy pipeline hands layers between VRAM and system RAM.

``sequential`` is the default because it is the one that actually runs on this
machine, and that was measured rather than assumed. The card is not free: the
Ollama server holds resident whatever model is answering the conversation, which
on this machine is 5.5 of the 8 GB. Against that, ``group`` OOMs during setup
and ``sequential`` peaked at 696 MiB over a 17-frame render. ``group`` is
faster when the card is empty, which is why it is offered rather than removed.
``model`` leaves a whole submodel resident and does not fit here.
"""

VIDEO_VRAM_ESTIMATE_MIB = {
    "sequential": 1200,
    "group": 5200,
    "model": 7200,
}
"""VRAM a video job needs per offload mode, in MiB.

``sequential`` is measured: 696 MiB peaked on a 17-frame render, and the
estimate is rounded up from there. The others are reasoned from how much each
one leaves resident. Pessimistic on purpose, because an estimate that
under-counts does not fail the check - it fails later, as an OOM in the middle
of a multi-minute render, which costs far more than a refused call.
"""

MUSIC_VRAM_ESTIMATE_MIB = {"music": 3200, "base": 4600, "full": 6200}
"""VRAM an AudioLDM2 job needs per model size, in MiB.

``music`` is the default because it is the one that coexists with the resident
voice engines; the others carry a Miscellaneous vocoder and are here so the
refusal can say by how much the card is short rather than just "no".
"""

DEFAULT_VIDEO_FRAMES = 49
DEFAULT_VIDEO_STEPS = 40
DEFAULT_OFFLOAD = "sequential"

MAX_MUSIC_SECONDS = 30
"""The measured ceiling on one music job.

The duration is the input, not a decode budget: AudioLDM2 denoises the whole
latent at once, so a 120 s request is not 4x the work of a 30 s one, it is a
latent that does not fit. Chain clips instead.
"""

DEFAULT_QUEUE_LIMIT = 8
"""How many heavy jobs may wait before a new one is refused.

A bound rather than an open-ended list: the queue holds full render requests,
each of which is minutes of GPU time, and an unbounded one is how a model that
misreads "queue it" ends up with a backlog nobody is waiting for.
"""

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def estimate_video_vram(frames: int, offload: str) -> int:
    """VRAM a video render needs, scaling with frames.

    Frames drive the attention and therefore the memory; steps do not, they
    just recompute over the same sequence. The scale factor is measured against
    49 frames, which is what the numbers above describe.
    """
    base = VIDEO_VRAM_ESTIMATE_MIB.get(offload, VIDEO_VRAM_ESTIMATE_MIB[DEFAULT_OFFLOAD])
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
    sequence: int = 0
    """Submission order, so the queue can say who is ahead of whom."""

    @property
    def seconds(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.finished_at or 0.0) - self.started_at

    @property
    def waiting(self) -> bool:
        """Queued but not started: what a caller is actually waiting on."""
        return self.outcome == "queued"


@dataclass
class QueueEntry:
    """A job waiting for the card, with the position the user would be told."""

    record: JobRecord
    event: asyncio.Event
    job: asyncio.Task[Any] | None = None
    needed_mib: int = 0
    kind: str = ""
    released: str = ""
    """What the VRAM release step did, or an empty string when it did nothing."""


def parse_ollama_mode(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return "off"
    if normalized in ("on", "auto"):
        return "auto"
    if normalized in ("off", "ask"):
        return normalized
    raise AgentError("COMPUTE_UNLOAD_OLLAMA must be on, off, or ask.")


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
    ollama_mode: str = "off"
    """Whether a heavy job may unload a resident Ollama model to free VRAM."""
    queue_limit: int = DEFAULT_QUEUE_LIMIT
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _queue: list[QueueEntry] = field(default_factory=list, repr=False)
    """Jobs waiting for the card, in submission order."""
    _sequence: int = 0
    _history: list[JobRecord] = field(default_factory=list, repr=False)
    _retired: dict[str, QueueEntry] = field(default_factory=dict, repr=False)
    """Finished jobs kept by id, so a late poll still gets its result."""
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
        offload: str = DEFAULT_OFFLOAD,
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
        seconds: int = 10,
        model: str = "music",
        name: str = "",
        runner: Any = None,
    ) -> JobRecord:
        """Generate music with AudioLDM2, alone, after checking it fits.

        ``music`` is the only size that coexists with the resident voice
        engines on an 8 GB card, so it is both the default and the reason the
        larger models are refused with a number rather than attempted.
        """
        text = _require_text(prompt, MUSIC_TOOL_NAME, MAX_PROMPT_CHARS)
        if model not in MUSIC_VRAM_ESTIMATE_MIB:
            raise AgentError(f"model must be one of: {', '.join(sorted(MUSIC_VRAM_ESTIMATE_MIB))}.")
        if seconds < 1 or seconds > MAX_MUSIC_SECONDS:
            raise AgentError(f"Music must be between 1 and {MAX_MUSIC_SECONDS} seconds.")
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
                " Free it before retrying, or use the 'music' model and a shorter duration; the other "
                "sizes need more VRAM than this card has free with voice resident."
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
        """Run one heavy job alone, waiting its turn behind any other.

        A job that does not fit right now is queued rather than refused, because
        the thing usually holding the card is another of these jobs: refusing
        the second video of a two-video request would be a bug dressed as a
        policy. Only a job that still will not fit once it reaches the front of
        the queue is refused, and by then the queue is empty and the reading is
        the freshest one there is.
        """
        script = self.resolve_script(script_name)
        entry = self._register(kind, script_name, argv, needed_mib)
        try:
            return await self._run_queued(
                entry,
                script_name,
                argv,
                script=script,
                timeout_seconds=timeout_seconds,
                runner=runner,
            )
        finally:
            self._retire(entry)

    def submit_video(
        self,
        prompt: str,
        *,
        frames: int = DEFAULT_VIDEO_FRAMES,
        steps: int = DEFAULT_VIDEO_STEPS,
        offload: str = DEFAULT_OFFLOAD,
        name: str = "",
    ) -> str:
        """Queue a video render and return its job id without waiting."""
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
        return self.submit(
            "Video generation", "video_ltx.py", argv, needed_mib=estimate_video_vram(frames, offload)
        )

    def submit_music(
        self,
        prompt: str,
        *,
        seconds: int = 10,
        model: str = "music",
        name: str = "",
    ) -> str:
        """Queue a music generation and return its job id without waiting."""
        text = _require_text(prompt, MUSIC_TOOL_NAME, MAX_PROMPT_CHARS)
        if model not in MUSIC_VRAM_ESTIMATE_MIB:
            raise AgentError(f"model must be one of: {', '.join(sorted(MUSIC_VRAM_ESTIMATE_MIB))}.")
        if seconds < 1 or seconds > MAX_MUSIC_SECONDS:
            raise AgentError(f"Music must be between 1 and {MAX_MUSIC_SECONDS} seconds.")
        output = self._output_path(name, "wav", "musica")
        argv = [
            "--prompt", text,
            "--segundos", str(seconds),
            "--modelo", model,
            "--salida", str(output),
        ]
        return self.submit(
            "Music generation", "musica.py", argv, needed_mib=MUSIC_VRAM_ESTIMATE_MIB[model]
        )

    async def _run_queued(
        self,
        entry: QueueEntry,
        script_name: str,
        argv: list[str],
        *,
        script: Path | None = None,
        timeout_seconds: int | None = None,
        runner: Any = None,
    ) -> JobRecord:
        """Wait for the card, then run the job, recording how it went.

        Both entry points go through here so a queued job and a blocking one
        cannot differ in what they check: the VRAM decision is made at the front
        of the queue, where the card is actually free, rather than at the moment
        the request happened to arrive.
        """
        script = script or self.resolve_script(script_name)
        record = entry.record
        loop = asyncio.get_running_loop()
        try:
            async with self._lock:
                if not self._can_fit(entry.needed_mib):
                    entry.released = await self._release_vram()
                if not self._can_fit(entry.needed_mib):
                    self.require_headroom(entry.kind, entry.needed_mib)
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
        finally:
            if entry in self._queue:
                self._queue.remove(entry)
            entry.event.set()

    def _retire(self, entry: QueueEntry) -> None:
        """Move a finished job out of the queue, keeping it answerable by id."""
        if entry in self._queue:
            self._queue.remove(entry)
        if entry.record.outcome in {"done", "failed", "cancelled"}:
            self._retired[f"job-{entry.record.sequence}"] = entry
            # Bounded, like the history: a poll cannot resurrect every job ever
            # run, only the recent ones.
            for stale in list(self._retired)[:-self.queue_limit]:
                del self._retired[stale]

    def _can_fit(self, needed_mib: int) -> bool:
        return self.read_vram().available_for_heavy_mib >= needed_mib

    async def _release_vram(self) -> str:
        """Unload a resident Ollama model, if that is what is in the way.

        Opt-in via ``COMPUTE_UNLOAD_OLLAMA=on``, because it is the user's
        server and the model reloads on the next request. Off by default for
        that reason: silently evicting someone's warm model to start a video is
        not a decision the agent should make on its own.
        """
        if self.ollama_mode != "auto":
            return ""
        reading = self.read_vram()
        if reading.used_by_others_mib <= 0:
            return ""
        return await unload_ollama(self.root_directory)

    # ------------------------------------------------------- the async queue
    def submit(
        self,
        kind: str,
        script_name: str,
        argv: list[str],
        *,
        needed_mib: int,
    ) -> str:
        """Start a heavy job in the background and return its id straight away.

        The point is that a turn can queue several requests without spending
        the whole turn waiting: a 20-minute render blocks the loop's caller, so
        three videos asked for at once would take an hour of turn time and no
        feedback in between. Here they take a second to accept, and the caller
        polls ``compute_result``.
        """
        entry = self._register(kind, script_name, argv, needed_mib)
        job_id = f"job-{entry.record.sequence}"
        loop = asyncio.get_running_loop()
        entry.job = loop.create_task(self._execute(entry, script_name, argv))
        return job_id


    def _register(self, kind: str, script_name: str, argv: list[str], needed_mib: int) -> QueueEntry:
        """Enqueue a job without starting it, used by both the blocking and async paths."""
        if len(self._queue) >= self.queue_limit:
            raise AgentError(
                f"{self.queue_limit} heavy jobs are already waiting and one is running. A render is "
                "minutes of GPU time, so queueing more just builds a backlog nobody is waiting for. "
                "Wait for compute_status to report fewer waiting jobs, or raise COMPUTE_QUEUE_LIMIT."
            )
        self._sequence += 1
        record = JobRecord(kind=kind, detail=script_name, sequence=self._sequence)
        entry = QueueEntry(record=record, event=asyncio.Event(), needed_mib=needed_mib, kind=kind)
        self._queue.append(entry)
        self._record(record)
        return entry

    async def _execute(self, entry: QueueEntry, script_name: str, argv: list[str]) -> JobRecord:
        """The background body of a queued job.

        It takes the same lock as a blocking call, so a queued job and a
        directly requested one share the single slot on the card rather than
        each assuming it has the GPU to itself.
        """
        try:
            return await self._run_queued(entry, script_name, argv)
        except Exception:  # noqa: BLE001 - the outcome is read back, not raised
            return entry.record
        finally:
            self._retire(entry)

    def result_text(self, job_id: str) -> str:
        """What a queued job has to say right now, or that the id is unknown."""
        entry = self._job(job_id)
        record = entry.record
        if record.waiting:
            position = self._position(entry)
            return f"Job {job_id} is waiting: position {position} of {self.queued} in the queue."
        if record.outcome == "running":
            return f"Job {job_id} is rendering now, on its own. It takes minutes; call again later."
        if record.outcome == "done":
            note = f"\n{entry.released}" if entry.released else ""
            return f"Job {job_id} finished in {record.seconds:.0f}s.{note}\n{record.output}"
        return f"Job {job_id} {record.outcome}."

    def _job(self, job_id: str) -> QueueEntry:
        wanted = (job_id or "").strip()
        for entry in self._queue:
            if f"job-{entry.record.sequence}" == wanted:
                return entry
        # A finished job leaves the queue, but its id must keep answering: a
        # caller that polls a moment late should be told the result, not that
        # the job never existed.
        retired = self._retired.get(wanted)
        if retired is not None:
            return retired
        raise AgentError(
            f"No job with id '{wanted}'. Call compute_status for the queue and recent jobs."
        )

    def _position(self, entry: QueueEntry) -> int:
        waiting = [item for item in self._queue if item.record.waiting]
        return waiting.index(entry) + 1 if entry in waiting else len(waiting)

    @property
    def queued(self) -> int:
        """How many jobs are waiting for the card, not counting the running one."""
        return sum(1 for entry in self._queue if entry.record.waiting)

    def queue_text(self) -> str:
        """The queue as the model and the user read it.

        Positions are included because "you are third" is the difference between
        waiting and wondering whether a request was received at all.
        """
        waiting = [entry for entry in self._queue if entry.record.waiting]
        if not waiting:
            return "Queue: empty."
        lines = [f"Queue: {len(waiting)} waiting, one job at a time."]
        for position, entry in enumerate(waiting, start=1):
            note = f" ({entry.released})" if entry.released else ""
            lines.append(f"  {position}. {entry.kind}{note}")
        return "\n".join(lines)

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
        waiting = self.queue_text()
        if waiting != "Queue: empty.":
            lines.append(waiting)
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


async def unload_ollama(root_directory: str, seconds: float = 20.0) -> str:
    """Unload whatever Ollama has resident, so the card is actually free.

    On this machine an Ollama model was sitting on 6390 of 8188 MiB, which is
    most of the reason a video job gets refused. Ollama keeps a model resident
    for a keep-alive window after the last request, and that is deliberate -
    it is a cache, not a leak - so the way to get the memory back is to ask it
    to unload, not to kill the process. Killing ``ollama serve`` would also
    take down anything else pointed at it, which for a server the user may be
    using for their own requests is a much bigger thing to do than evicting a
    cache entry.

    Returns a line saying what happened, for the caller to fold into its own
    report. Failure is not an error: the caller is about to re-measure the
    card anyway, and the answer is in that measurement.
    """
    binary = shutil.which("ollama")
    if binary is None:
        return "Ollama is not installed; nothing to unload."

    async def ollama(*arguments: str) -> subprocess.CompletedProcess[str]:
        process = await asyncio.create_subprocess_exec(
            binary,
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        return subprocess.CompletedProcess(arguments, process.returncode or 0, stdout.decode("utf-8", "replace"), "")

    # `ollama ps` is the loaded models and their VRAM. It is the right source
    # because it is Ollama's own view of its cache, not a guess from nvidia-smi
    # about which of its processes is which.
    try:
        listing = await asyncio.wait_for(ollama("ps"), timeout=15)
    except (TimeoutError, OSError) as error:
        return f"Could not ask Ollama what it has loaded: {error}"

    if listing.returncode != 0:
        return "Ollama did not answer, so nothing was unloaded."
    rows = [line for line in listing.stdout.splitlines()[1:] if line.strip()]
    if not rows:
        return "Ollama has no model loaded; the memory is not its."

    unloaded: list[str] = []
    for row in rows:
        name = row.split()[0] if row.split() else ""
        if not name:
            continue
        # `ollama stop` expires the keep-alive, which is the supported way to
        # drop a resident model. The alternative, killing the process, would
        # take the whole server with it.
        stopped = await ollama("stop", name)
        if stopped.returncode == 0:
            unloaded.append(name)
        else:
            return (
                f"Tried to unload {name} and Ollama refused. Free it yourself with "
                f"`ollama stop {name}`, then retry."
            )

    if not unloaded:
        return "Ollama had a model loaded but none could be named; free it and retry."

    # The stop is asynchronous: the process has to exit and the driver has to
    # release the memory. Reporting success before that would hand the next
    # check a reading that has not caught up yet, which is the same stale
    # measurement the whole feature exists to avoid.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    before = free_vram_mib()
    while loop.time() < deadline:
        await asyncio.sleep(0.5)
        if free_vram_mib() - before > 512:
            break
    reclaimed = max(0, free_vram_mib() - before)
    return (
        f"Unloaded {', '.join(unloaded)} from Ollama, which had been holding VRAM; "
        f"{reclaimed} MiB came back. The next request for that model reloads it, "
        "which takes a few seconds."
    )


def free_vram_mib() -> int:
    """Free VRAM right now, or 0 when nvidia-smi cannot answer."""
    return probe_vram(DEFAULT_VRAM_TOTAL_MIB).free_mib


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
                                "How layers move between VRAM and RAM. 'sequential' is the default: it "
                                "is the one that fits while another model holds the card, and it is "
                                "the slowest. 'group' is faster but needs a nearly empty GPU."
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
                    "Generate music or ambience from a text prompt with AudioLDM2 and save a wav in "
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
                            "description": f"Duration, 1 to {MAX_MUSIC_SECONDS}. 10 is quick. Longer costs VRAM, not just time.",
                        },
                        "name": {"type": "string", "description": "Optional output file name."},
                    },
                    "required": ["prompt"],
                },
            },
        },            {
                "type": "function",
                "function": {
                    "name": STATUS_TOOL_NAME,
                    "description": (
                        "Report free VRAM, what is holding it, the voice reserve, the heavy job queue with "
                        "each waiting job's position, and recent jobs. Call this when a heavy job was "
                        "refused for lack of memory - it names the process to free - and to see how far "
                        "back a queue is."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": QUEUE_TOOL_NAME,
                    "description": (
                        "Submit a heavy video or music job without waiting for it and return a job id "
                        "immediately, so several can be queued in one turn instead of blocking on the "
                        "first. Jobs run one at a time in the order they were queued. Then call "
                        "compute_status to see positions, and read each finished result back with "
                        "compute_result once it reports done."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["video", "music"],
                                "description": "Which heavy job to queue.",
                            },
                            "prompt": {"type": "string", "description": "For video: what it shows. For music: what it sounds like."},
                            "frames": {"type": "integer", "description": "Video only. 17 is quick, 49 default."},
                            "steps": {"type": "integer", "description": "Video only. More steps is slower, not bigger in VRAM."},
                            "offload": {
                                "type": "string",
                                "enum": list(OFFLOAD_MODES),
                                "description": "Video only. 'sequential' is the default and fits alongside another model on the card.",
                            },
                            "seconds": {"type": "integer", "description": f"Music only. Duration, 1 to {MAX_MUSIC_SECONDS}."},
                            "name": {"type": "string", "description": "Optional output file name."},
                        },
                        "required": ["kind", "prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": RESULT_TOOL_NAME,
                    "description": (
                        "Read the result of a job submitted with queue_job: 'done' with the output path, "
                        "'waiting' with its position in the queue, or 'failed' with the reason. This is "
                        "how a queued job is collected without having blocked on it."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "job_id": {"type": "string", "description": "The id queue_job returned."}
                        },
                        "required": ["job_id"],
                    },
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
