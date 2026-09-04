"""Measuring whether the added layers are actually audible.

The failure this exists to catch is not a crash: it is a render that completes, masters
perfectly to target, and contains music 27dB and effects 40dB below the dialogue - which
is to say, contains neither. Loudness compliance says nothing about balance, so balance
gets measured separately.

Levels are reported *relative to the source audio*, because that is the reference a
viewer actually hears against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import read_wav

#: Below roughly this much under the dialogue, a sustained bed stops registering as music
#: and becomes an artefact you notice only when it stops.
BED_INAUDIBLE_BELOW_DB = -26.0

#: One-shots are transient, so they are judged peak-against-peak. Comparing an effect's
#: peak to the dialogue's *RMS* flatters it by roughly the dialogue's crest factor and
#: makes a far-too-quiet effect look acceptable.
SFX_INAUDIBLE_BELOW_DB = -15.0


def _db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-9))


def rms_db(path: Path) -> float:
    samples, _ = read_wav(path)
    mono = samples.mean(axis=1)
    return _db(np.sqrt((mono**2).mean()))


def peak_db(path: Path) -> float:
    samples, _ = read_wav(path)
    return _db(np.abs(samples).max())


@dataclass
class Balance:
    """Bus levels relative to the source audio, in dB."""

    reference_rms_db: float
    reference_peak_db: float
    music_rms_db: float | None
    sfx_peak_db: float | None
    ambience_rms_db: float | None
    #: How many one-shots actually fire, and over what span - four effects across a
    #: minute is a different problem from four quiet ones.
    sfx_events: int = 0
    duration: float = 0.0

    @property
    def music_offset_db(self) -> float | None:
        if self.music_rms_db is None:
            return None
        return self.music_rms_db - self.reference_rms_db

    @property
    def sfx_offset_db(self) -> float | None:
        """Effect peak against source *peak* - like for like."""
        if self.sfx_peak_db is None:
            return None
        return self.sfx_peak_db - self.reference_peak_db

    @property
    def sfx_per_minute(self) -> float:
        if self.duration <= 0:
            return 0.0
        return self.sfx_events / (self.duration / 60.0)

    def problems(self) -> list[str]:
        """Human-readable complaints, empty when the balance is defensible."""
        issues: list[str] = []
        music = self.music_offset_db
        sfx = self.sfx_offset_db

        if music is not None and music < BED_INAUDIBLE_BELOW_DB:
            issues.append(
                f"music sits {abs(music):.1f} dB under the source audio - inaudible "
                f"(want no more than {abs(BED_INAUDIBLE_BELOW_DB):.0f} dB down)"
            )
        if sfx is not None and sfx < SFX_INAUDIBLE_BELOW_DB:
            issues.append(
                f"effects peak {abs(sfx):.1f} dB under the source peaks - inaudible "
                f"(want no more than {abs(SFX_INAUDIBLE_BELOW_DB):.0f} dB down)"
            )
        if self.duration > 0 and self.sfx_events and self.sfx_per_minute < 3:
            issues.append(
                f"only {self.sfx_events} effect(s) across {self.duration:.0f}s "
                f"({self.sfx_per_minute:.1f}/min) - too sparse to register as sound design"
            )
        return issues


def count_events(path: Path, min_gap: float = 0.25) -> int:
    """How many distinct one-shots fire on a bus.

    A single decaying hit crosses any fixed threshold many times as it oscillates, so
    crossings are debounced: onsets closer together than `min_gap` are one event.
    """
    samples, sr = read_wav(path)
    env = np.abs(samples.mean(axis=1))
    if env.max() <= 1e-6:
        return 0

    # Smooth to an amplitude envelope before thresholding, so the waveform's own
    # oscillation cannot register as separate events.
    window = max(1, sr // 100)  # 10ms
    smoothed = np.convolve(env, np.ones(window) / window, mode="same")
    loud = smoothed > smoothed.max() * 0.08

    onsets = np.flatnonzero(np.diff(loud.astype(np.int8)) == 1)
    if onsets.size == 0:
        return int(loud.any())

    gap = int(min_gap * sr)
    events, last = 1, onsets[0]
    for onset in onsets[1:]:
        if onset - last >= gap:
            events += 1
            last = onset
    return events


def measure(project: Path) -> Balance | None:
    """Read the bus stems written during a render and report their balance.

    Returns None when there is no source audio to reference against - a silent source
    has no meaningful balance, only absolute levels.
    """
    buses = Path(project) / "buses"
    original = buses / "original.wav"
    if not original.exists():
        return None

    music = buses / "music.wav"
    sfx = buses / "sfx.wav"
    ambience = buses / "ambience.wav"

    samples, sr = read_wav(original)
    duration = len(samples) / sr

    events = count_events(sfx) if sfx.exists() else 0

    return Balance(
        reference_rms_db=rms_db(original),
        reference_peak_db=peak_db(original),
        sfx_events=events,
        duration=duration,
        music_rms_db=rms_db(music) if music.exists() else None,
        # Peak for one-shots: their RMS over a whole timeline is meaningless, since they
        # are silent almost all of it.
        sfx_peak_db=peak_db(sfx) if sfx.exists() else None,
        ambience_rms_db=rms_db(ambience) if ambience.exists() else None,
    )
