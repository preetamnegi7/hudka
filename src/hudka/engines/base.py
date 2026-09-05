"""Engine protocol and the licence metadata that gates it.

Model choice here is a legal decision before it is a quality decision: output goes into
monetized video and is offered onward to third parties. Every engine therefore declares
its licence, and `require_usable()` refuses to run anything that isn't cleared for that.

The check is deliberately in code rather than in documentation, so a non-commercial model
cannot end up in a render by accident.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

SAMPLE_RATE = 44100


class LicenceError(RuntimeError):
    """Raised when an engine is not cleared for the requested use."""


@dataclass(frozen=True)
class Licence:
    """What an engine's weights permit. Verified against licence texts, September 2026."""

    name: str
    url: str
    #: Commercial use permitted at all, on any terms.
    commercial: bool
    #: Requires an explicit opt-in flag because of restrictions a user must knowingly accept.
    requires_optin: bool = False
    #: Annual revenue above which a paid licence is required (None = no cap).
    revenue_cap_usd: int | None = None
    #: Regions the licence grant does not extend to.
    territory_exclusions: tuple[str, ...] = ()
    #: Provenance of the training data — the part that matters for output cleanliness.
    training_data: str = ""
    notes: str = ""

    def summary(self) -> str:
        if not self.commercial:
            return f"{self.name} — NON-COMMERCIAL, excluded from renders"
        bits = [self.name]
        if self.revenue_cap_usd:
            bits.append(f"free commercial use under ${self.revenue_cap_usd/1_000_000:.0f}M revenue")
        if self.territory_exclusions:
            bits.append("excludes " + ", ".join(self.territory_exclusions))
        return " — ".join(bits)


@dataclass
class GenerateRequest:
    """One unit of audio to generate."""

    prompt: str
    duration: float
    seed: int
    #: Video path, for engines that condition on picture (Hunyuan foley).
    video: Path | None = None
    #: Time window within that video the audio should correspond to.
    window: tuple[float, float] | None = None
    extra: dict = field(default_factory=dict)

    def cache_key(self, engine_id: str) -> str:
        """Content hash so re-renders only regenerate what actually changed.

        Everything that reaches the model belongs here, `extra` included: steps, guidance
        and the negative prompt all change the audio, and omitting them means a user
        adjusts a setting, hears the cached stem, and concludes the control does nothing.

        Tone controls are deliberately absent - they are applied at placement, so changing
        a filter costs no generation at all.
        """
        parts = [engine_id, self.prompt, f"{self.duration:.3f}", str(self.seed)]
        if self.video is not None:
            parts.append(Path(self.video).name)
            # The window is nested here on purpose. It reaches a model only when that
            # model conditions on picture - hunyuan_foley cuts exactly this span out of
            # the video in `_extract_window`, and nothing else reads it. Hashing it for
            # every engine meant moving a cue one frame produced a new key, a cache miss,
            # and then the deletion of the old stem by generate_stems' stale-file cleanup:
            # dragging a clip threw away the sound it named.
            parts.append(f"{self.window[0]:.3f}-{self.window[1]:.3f}")
        for key in sorted(self.extra):
            parts.append(f"{key}={self.extra[key]}")
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class Engine(Protocol):
    """A generation backend. Implementations live alongside this file."""

    id: str
    licence: Licence
    #: What this engine is good for, used to validate cue/engine pairings.
    kinds: tuple[str, ...]

    def generate(self, req: GenerateRequest, out_path: Path) -> Path:
        """Render `req` to a 44.1kHz stereo WAV at `out_path` and return that path."""
        ...

    def preflight(self) -> None:
        """Raise with actionable text if this engine cannot run.

        Checked before any generation starts, so a missing dependency surfaces
        immediately rather than after the first cue has already been queued.
        """
        ...

    def unload(self) -> None:
        """Free VRAM. Called between stages — 12GB can't hold every model at once."""
        ...


def require_usable(engine: Engine, *, allow_noncommercial: bool, opted_in: set[str]) -> None:
    """Gate an engine before it runs. Raises LicenceError if it isn't cleared."""
    lic = engine.licence

    if not lic.commercial and not allow_noncommercial:
        raise LicenceError(
            f"{engine.id} is licensed {lic.name}, which forbids commercial use.\n"
            f"  Its output cannot be used in monetized video or client work.\n"
            f"  If this render is strictly personal, pass --allow-noncommercial."
        )

    if lic.requires_optin and engine.id not in opted_in:
        detail = ""
        if lic.territory_exclusions:
            detail = (
                f"\n  Its licence grant excludes {', '.join(lic.territory_exclusions)} — "
                "confirm that suits\n  where you operate and where the video will be distributed."
            )
        raise LicenceError(
            f"{engine.id} needs an explicit opt-in.\n"
            f"  Licence: {lic.summary()}{detail}\n"
            f"  Enable it with --engine {engine.id}."
        )
