"""Silence engine — CI stub so the whole pipeline can run without downloading weights.

Emits a deterministic, seed-derived low-level noise burst of exactly the requested
length. That is enough for the mix graph, timing, loudness and provenance tests to
assert real behaviour offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..audio import SAMPLE_RATE, write_wav
from .base import Engine, GenerateRequest, Licence
from .licences import PUBLIC_DOMAIN


class SilenceEngine(Engine):
    id = "silence"
    licence: Licence = PUBLIC_DOMAIN
    kinds = ("sfx", "music", "ambience")

    def generate(self, req: GenerateRequest, out_path: Path) -> Path:
        n = max(1, int(round(req.duration * SAMPLE_RATE)))
        rng = np.random.default_rng(req.seed or 0)
        # A short decaying burst, so onset detection and placement tests have something real.
        env = np.exp(-np.linspace(0, 6, n, dtype=np.float32))
        noise = rng.standard_normal(n).astype(np.float32) * 0.05 * env
        return write_wav(out_path, np.stack([noise, noise], axis=-1), SAMPLE_RATE)

    def unload(self) -> None:
        return
