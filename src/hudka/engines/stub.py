"""Placeholder engine - deterministic synthetic tones that cannot pass for real output.

Used by preview mode and by the offline test suite, so the whole pipeline can run with no
weights and no GPU.

It used to emit a decaying burst of white noise. That was a mistake with a real cost: a
user rendered in preview mode, heard hiss where the music should be, and concluded the
product was broken. Filtered noise sounds like a failed generator; a metronome blip
sounds like what it is. The placeholder now has to be *unmistakably* a placeholder.

What the tests rely on is preserved: deterministic per seed, a clear onset for
`find_onset`, non-zero energy, stereo, exactly the requested length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import SAMPLE_RATE, write_wav
from .base import Engine, GenerateRequest, Licence
from .licences import PUBLIC_DOMAIN

#: One blip per second on anything long enough to be a bed - nobody mistakes a metronome
#: for music, which is the whole point.
BLIP_PERIOD = 1.0
BLIP_LENGTH = 0.35
BLIP_AMPLITUDE = 0.25
#: Sustained tone under a placeholder bed, well below the blips.
DRONE_HZ = 110.0
DRONE_AMPLITUDE = 0.08


def _blip(seed: int, n: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A single sine blip with a fast decay, pitched from the seed in 440-880 Hz."""
    freq = 440.0 * 2.0 ** ((seed % 12) / 12.0)
    t = np.arange(n, dtype=np.float32) / sr
    return (np.sin(2.0 * np.pi * freq * t) * np.exp(-6.0 * t / max(t[-1], 1e-6))
            * BLIP_AMPLITUDE).astype(np.float32)


class SilenceEngine(Engine):
    #: The id stays `silence` so existing cue sheets, licence tables and tests keep working.
    id = "silence"
    licence: Licence = PUBLIC_DOMAIN
    kinds = ("sfx", "music", "ambience")

    def generate(self, req: GenerateRequest, out_path: Path) -> Path:
        n = max(1, int(round(req.duration * SAMPLE_RATE)))
        out = np.zeros(n, dtype=np.float32)

        blip_n = min(n, int(BLIP_LENGTH * SAMPLE_RATE))
        blip = _blip(req.seed or 0, blip_n)

        if req.duration <= 1.5:
            out[:blip_n] = blip
        else:
            # A bed: metronome blips over a quiet sustained drone. The drone matters for
            # more than texture - blips alone are almost all silence, and a mix built on
            # them cannot be brought to the loudness target without clipping the peaks,
            # so a preview would misreport the balance of the real render it stands in for.
            t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
            out += (np.sin(2.0 * np.pi * DRONE_HZ * t) * DRONE_AMPLITUDE).astype(np.float32)
            step = int(BLIP_PERIOD * SAMPLE_RATE)
            for start in range(0, n - blip_n + 1, step):
                out[start:start + blip_n] += blip

        return write_wav(out_path, np.stack([out, out], axis=-1), SAMPLE_RATE)

    def unload(self) -> None:
        return
