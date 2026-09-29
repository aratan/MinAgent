"""What to give up when the window fills, and in what order.

An 8k window is not a budget you can spend evenly. The conversation grows with
the task while the fixed prompt is paid for on every request, so most of what
has to go is in the transcript, not in the setup. That decides the order: the
steps that free the most and lose the least go first, and nothing that cannot
be recovered goes at all.

1. Old tool results become archive references. This is the only step that frees
   thousands of tokens rather than hundreds, and it is lossless - the full text
   is archived and one ``recall_tool_output`` call brings back any of it.
2. Attached images are released. A single screenshot is a couple of thousand
   tokens, more than any other item here, and it is the least lossy thing to
   drop: the pixels are the only copy in the transcript, but the path is not,
   and ``view_image`` puts them straight back. The stub names the path so the
   model knows what it can reload rather than assuming it has already seen it.
3. Memory hints are dropped. They are a recall aid, and the same knowledge is
   one ``recall`` call away.
4. Capabilities the agent has not touched during the current turn are unloaded
   at once, instead of after the usual grace turns. Anything it has called
   since the turn started stays, so a task in flight never loses the tool it is
   using halfway through.
5. The capability index loses its summaries, and goes last because it is what
   the model reads to decide what to load: the names stay, so a capability can
   still be loaded, but stripping it early leaves the agent unable to tell
   what anything is for exactly when it is short on room.

What is never given up: the always-loaded tools, the core rules, the
conversation itself, and anything that would be lost rather than archived.
Compaction is not in this list because the turn loop already runs it, and it
wants the room these steps free.

Restoring is symmetric, and only happens once the context is back under the low
mark. Without that hysteresis a conversation hovering at the threshold would
shed and restore on every turn, and the prompt would change shape constantly -
which defeats provider-side prompt caching and wastes the model's attention.
"""

from __future__ import annotations

from dataclasses import dataclass

SHED_INDEX_SUMMARIES = "index summaries"
SHED_MEMORY_HINTS = "memory hints"
SHED_IDLE_CAPABILITIES = "idle capabilities"
SHED_OLD_TOOL_RESULTS = "old tool results"
SHED_ATTACHED_IMAGES = "attached images"

SHED_STEPS = (
    SHED_OLD_TOOL_RESULTS,
    SHED_ATTACHED_IMAGES,
    SHED_MEMORY_HINTS,
    SHED_IDLE_CAPABILITIES,
    SHED_INDEX_SUMMARIES,
)
"""The shedding cascade, most tokens for the least loss first."""


@dataclass(frozen=True)
class ContextPolicy:
    """When to start giving things up, and when to give them back."""

    high_watermark: float = 0.75
    low_watermark: float = 0.55

    def __post_init__(self) -> None:
        if not 0 < self.low_watermark < self.high_watermark <= 1:
            raise ValueError(
                "Context watermarks must satisfy 0 < low < high <= 1, got "
                f"low={self.low_watermark}, high={self.high_watermark}."
            )

    def pressure(self, used_tokens: int, window_tokens: int) -> float:
        """How much of the window the prompt currently occupies."""
        if window_tokens <= 0:
            return 0.0
        return used_tokens / window_tokens

    def steps_to_shed(self, used_tokens: int, window_tokens: int, already_shed: int) -> int:
        """How many steps of the cascade to apply right now.

        ``already_shed`` is how many are in force, which is what makes this
        cumulative rather than a fresh decision each turn: once the index is
        compact, staying compact is free, and the next step is only taken when
        the context is still too big.
        """
        pressure = self.pressure(used_tokens, window_tokens)
        if pressure < self.high_watermark:
            return 0
        if already_shed >= len(SHED_STEPS):
            return already_shed
        # The further past the mark, the further into the cascade to go: at the
        # mark itself one step, at the very top of the window all of them, and
        # never past the last one.
        headroom = 1 - self.high_watermark
        overshoot = min(1.0, (pressure - self.high_watermark) / headroom) if headroom else 1.0
        wanted = min(len(SHED_STEPS), 1 + int(overshoot * len(SHED_STEPS)))
        return max(already_shed, wanted)

    def should_restore(self, used_tokens: int, window_tokens: int) -> bool:
        """Whether everything that was given up should come back."""
        return self.pressure(used_tokens, window_tokens) < self.low_watermark
