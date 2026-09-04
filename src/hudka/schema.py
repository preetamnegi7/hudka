"""Pydantic models for `cues.json` — the contract between analysis and rendering.

A cue sheet is a plain, hand-editable JSON file. Every cue carries an explicit seed so
renders are reproducible and "regenerate just this one" is a seed bump rather than a
re-roll of the whole mix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# Engine ids understood by the renderer. `engines.registry` maps these to implementations
# and to the licence metadata that gates them.
EngineId = Literal[
    "stable-audio-3-medium",
    "stable-audio-3-small-sfx",
    "stable-audio-3-small-music",
    "acestep-1.5",
    "hunyuan-foley",
    "silence",  # test/CI stub: emits silence of the requested length, downloads nothing
]

Seconds = Annotated[float, Field(ge=0.0)]
Gain = Annotated[float, Field(
    ge=-60.0, le=12.0,
    description=(
        "dB relative to the normalised stem reference (-20 LUFS for beds, -12 dBFS peak "
        "for one-shots), not to the raw generation. Generated audio lands at an arbitrary "
        "level, so an offset from the raw clip produces a different balance every run."
    ),
)]


class VideoInfo(BaseModel):
    """Facts about the source video, copied from analysis.json so a cue sheet is self-contained."""

    path: str
    duration: Seconds
    fps: float = Field(gt=0)
    width: int = 0
    height: int = 0
    has_audio: bool = False
    has_dialogue: bool = False


class SfxCue(BaseModel):
    """A one-shot sound placed at a point in time — an impact, whoosh, click, footstep."""

    id: str
    at: Seconds
    duration: Annotated[float, Field(gt=0.0, le=30.0)] = 1.5
    prompt: str = Field(min_length=3)
    engine: EngineId = "stable-audio-3-small-sfx"
    gain_db: Gain = -6.0
    pan: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.0
    seed: int = 0
    shot: int | None = None
    # Generated clips often carry 50-200ms of lead-in before the actual hit. When true,
    # the renderer detects the onset and shifts placement so the transient lands on `at`
    # rather than the file merely *starting* there.
    align_transient: bool = True
    note: str = ""

    @property
    def end(self) -> float:
        return self.at + self.duration


class BedCue(BaseModel):
    """A sustained layer spanning a time range — a music bed or an ambience/room-tone bed."""

    id: str
    start: Seconds
    end: Seconds
    prompt: str = Field(min_length=3)
    engine: EngineId = "stable-audio-3-medium"
    gain_db: Gain = -18.0
    fade_in: Seconds = 0.5
    fade_out: Seconds = 1.5
    # Duck this bed under the dialogue + SFX key signal, so speech and hits always win.
    duck: bool = True
    # If the generated clip is shorter than the range, loop it with equal-power crossfades.
    loop: bool = False
    seed: int = 0
    note: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @model_validator(mode="after")
    def _check_range(self) -> BedCue:
        if self.end <= self.start:
            raise ValueError(f"bed {self.id!r}: end ({self.end}) must be after start ({self.start})")
        if self.fade_in + self.fade_out > self.duration:
            raise ValueError(
                f"bed {self.id!r}: fades ({self.fade_in}+{self.fade_out}s) exceed "
                f"its {self.duration:.2f}s length"
            )
        return self


#: Cue sheets written before stem normalisation existed carry gains that were absolute
#: attenuations of the raw generation. Rendering those under the new semantics would land
#: them roughly 15 dB too quiet, so they are rendered the old way instead.
CURRENT_VERSION = 2


class CueSheet(BaseModel):
    """The full audio design for one video."""

    #: Defaults to 1 so a sheet written before versioning keeps its original meaning.
    version: int = 1
    video: VideoInfo
    preset: str = "short-form"
    target_lufs: Annotated[float, Field(ge=-40.0, le=-5.0)] = -14.0
    true_peak_db: Annotated[float, Field(ge=-9.0, le=0.0)] = -1.0
    # Keep the source audio (dialogue/voiceover) in the mix. Turned off for silent footage.
    keep_original_audio: bool = True
    original_gain_db: Gain = 0.0
    music: list[BedCue] = Field(default_factory=list)
    ambience: list[BedCue] = Field(default_factory=list)
    sfx: list[SfxCue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> CueSheet:
        limit = self.video.duration + 0.5  # small tolerance for cues landing on the last frame
        ids: set[str] = set()

        for cue in self.all_cues():
            if cue.id in ids:
                raise ValueError(f"duplicate cue id {cue.id!r}")
            ids.add(cue.id)

        for bed in (*self.music, *self.ambience):
            if bed.start >= limit:
                raise ValueError(f"bed {bed.id!r} starts at {bed.start}s, past the {limit:.2f}s video")
        for cue in self.sfx:
            if cue.at >= limit:
                raise ValueError(f"sfx {cue.id!r} fires at {cue.at}s, past the {limit:.2f}s video")

        # Overlapping music beds fight each other and wreck the loudness target. Ambience
        # beds are allowed to overlap - layering room tones is normal and intentional.
        for a, b in _pairs(sorted(self.music, key=lambda m: m.start)):
            if b.start < a.end:
                raise ValueError(
                    f"music beds {a.id!r} and {b.id!r} overlap "
                    f"({b.start:.2f}s < {a.end:.2f}s); crossfade them into one bed instead"
                )
        return self

    def all_cues(self) -> list[SfxCue | BedCue]:
        return [*self.music, *self.ambience, *self.sfx]

    @classmethod
    def load(cls, path: Path) -> CueSheet:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @property
    def normalizes_stems(self) -> bool:
        """Whether gains in this sheet are offsets from a normalised reference."""
        return self.version >= 2

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )


def _pairs(items: list[BedCue]):
    return zip(items, items[1:])
