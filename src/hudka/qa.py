"""Content quality gates: a render must not pass on noise, silence or distortion.

`balance.py` measures *level* - whether the layers are audible against the dialogue.
This measures *content* - whether what is audible is actually sound. The gap between
them is how a render of pure noise once mastered perfectly to -14 LUFS, passed the
balance check, and was announced "on target" in green.

Thresholds are pinned by tests against synthesised signals (a sine, white noise, a
square wave), not copied from a notebook: spectral flatness in particular differs by a
factor of ~1.5 between amplitude- and power-spectrum estimators, and the number only
means something alongside the estimator that produced it. This module uses the
power spectrum, framed and energy-gated, median across frames.

Policy: **block** on what cannot be right (silence, saturation, heavy clipping);
**warn** on what is probably wrong but sometimes legitimate (noise-like content is also
wind, rain and air). Warnings never stop a render. Blocks name the cues and say what to
press.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, peak_dbfs, read_wav, rms_dbfs, to_stereo

#: A stem this quiet is not a sound, it is a failed generation.
SILENT_BELOW_DBFS = -60.0

#: Peak-to-RMS below this is a saturated block, not audio. Real generations measured
#: 13.5-26 dB; the garbage measured 1.4-1.7 dB. Same figure the engine uses.
SATURATION_CREST_DB = 6.0

#: Fractions, never counts: a legitimate bed contains a handful of full-scale samples
#: because the library clamps its output to +-1.
CLIP_BLOCK_FRACTION = 0.01
CLIP_WARN_FRACTION = 0.001
CLIP_LEVEL = 0.999

#: Real generations sit at |dc| <= 0.001; the saturated ones drifted to 0.07.
DC_WARN = 0.005

#: Power-spectrum flatness, framed and gated (see `spectral_flatness`). Gaussian noise
#: measures ~0.56 on this estimator, real music and effects <= 0.11.
NOISE_FLATNESS = 0.35

#: A bed that stops this far short of its range leaves silence the user did not ask for.
SHORT_BED_TOLERANCE_S = 0.5


def spectral_flatness(samples: np.ndarray, sr: int = SAMPLE_RATE,
                      frame: int = 2048, gate_db: float = -50.0) -> float:
    """Median per-frame power-spectrum flatness over frames carrying energy.

    1.0 is white noise, ~0 is a pure tone. Framing and gating matter: a one-shot is
    mostly silence, and a whole-file FFT of silence-plus-a-hit says nothing useful.
    """
    mono = to_stereo(samples).mean(axis=1)
    if mono.size < frame:
        return 0.0
    window = np.hanning(frame).astype(np.float32)
    hop = frame // 2
    gate = 10.0 ** (gate_db / 20.0)
    values = []
    for start in range(0, mono.size - frame + 1, hop):
        seg = mono[start:start + frame]
        if np.sqrt((seg ** 2).mean()) < gate:
            continue
        power = np.abs(np.fft.rfft(seg * window)) ** 2 + 1e-20
        band = power[1:-1]
        values.append(float(np.exp(np.log(band).mean()) / band.mean()))
    return float(np.median(values)) if values else 0.0


def crest_db(samples: np.ndarray) -> float:
    """Peak-to-RMS in dB. Near zero means saturated, not dynamic."""
    a = to_stereo(samples)
    peak, rms = peak_dbfs(a), rms_dbfs(a)
    if peak <= -119.0 or rms <= -119.0:
        return 0.0
    return peak - rms


@dataclass
class StemQuality:
    cue_id: str
    kind: str
    peak_db: float
    crest: float
    clipped_fraction: float
    dc_offset: float
    flatness: float
    length_s: float
    #: For beds without loop: the range they were meant to cover.
    wanted_s: float | None = None

    def problems(self) -> list[str]:
        """Conditions that cannot be right. A render must not proceed on these."""
        out: list[str] = []
        if self.peak_db < SILENT_BELOW_DBFS:
            out.append(f"{self.cue_id}: silent (peak {self.peak_db:.0f} dBFS)")
            return out  # everything else is meaningless on silence
        if self.crest < SATURATION_CREST_DB:
            out.append(f"{self.cue_id}: saturated - a distorted block, not a sound "
                       f"(crest {self.crest:.1f} dB)")
        if self.clipped_fraction >= CLIP_BLOCK_FRACTION:
            out.append(f"{self.cue_id}: clipped ({self.clipped_fraction:.1%} of samples "
                       "at full scale)")
        return out

    def warnings(self) -> list[str]:
        """Probably wrong, sometimes legitimate. Reported, never blocking."""
        out: list[str] = []
        if self.problems():
            return out
        if CLIP_WARN_FRACTION <= self.clipped_fraction < CLIP_BLOCK_FRACTION:
            out.append(f"{self.cue_id}: some clipping ({self.clipped_fraction:.2%})")
        if abs(self.dc_offset) > DC_WARN:
            out.append(f"{self.cue_id}: DC offset {self.dc_offset:+.3f}")
        if self.flatness > NOISE_FLATNESS:
            out.append(f"{self.cue_id}: sounds like noise rather than a {self.kind} "
                       f"(flatness {self.flatness:.2f}) - try variations")
        if self.wanted_s is not None and self.wanted_s - self.length_s > SHORT_BED_TOLERANCE_S:
            out.append(f"{self.cue_id}: covers {self.length_s:.0f}s of its {self.wanted_s:.0f}s "
                       "range; the rest is silent - set loop, or use a longer-range engine")
        return out


def measure_stem(path: Path, cue_id: str, kind: str,
                 wanted_s: float | None = None) -> StemQuality:
    samples, sr = read_wav(Path(path))
    stereo = to_stereo(samples)
    mono = stereo.mean(axis=1)
    return StemQuality(
        cue_id=cue_id,
        kind=kind,
        peak_db=peak_dbfs(stereo),
        crest=crest_db(stereo),
        clipped_fraction=float(np.count_nonzero(np.abs(stereo) >= CLIP_LEVEL) / max(stereo.size, 1)),
        dc_offset=float(mono.mean()),
        flatness=spectral_flatness(stereo, sr),
        length_s=stereo.shape[0] / sr,
        wanted_s=wanted_s,
    )


@dataclass
class RenderQuality:
    stems: list[StemQuality] = field(default_factory=list)
    mix_lufs: float | None = None
    mix_flatness: float | None = None
    sfx_events: int | None = None
    sfx_cues: int = 0
    balance_problems: list[str] = field(default_factory=list)

    def problems(self) -> list[str]:
        out: list[str] = []
        for s in self.stems:
            out.extend(s.problems())
        if self.mix_lufs is None or not np.isfinite(self.mix_lufs):
            out.append("the finished mix is silent or could not be measured")
        if self.sfx_cues and self.sfx_events == 0:
            out.append("no effect landed in the mix - every effect cue is silent or past the end")
        return out

    def warnings(self) -> list[str]:
        out: list[str] = []
        for s in self.stems:
            out.extend(s.warnings())
        noisy = [s for s in self.stems if s.flatness > NOISE_FLATNESS and not s.problems()]
        if self.stems and len(noisy) == len(self.stems):
            out.append("every generated sound is noise-like - if this was not a preview, "
                       "something is wrong with the engine")
        if self.mix_flatness is not None and self.mix_flatness > NOISE_FLATNESS:
            out.append(f"the finished mix sounds like noise (flatness {self.mix_flatness:.2f})")
        out.extend(self.balance_problems)
        return out

    @property
    def verdict(self) -> str:
        if self.problems():
            return "fail"
        return "warn" if self.warnings() else "ok"


class QualityError(RuntimeError):
    """Raised when stems fail a block-level check; carries the offending cue ids."""

    def __init__(self, problems: list[str], cue_ids: list[str]):
        self.problems = problems
        self.cue_ids = cue_ids
        listed = "\n".join(f"  - {p}" for p in problems)
        super().__init__(
            "some generated sounds failed quality checks:\n" + listed +
            "\n\nRe-roll the listed cues (or try variations) and render again."
        )
