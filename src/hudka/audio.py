"""Small audio helpers. Deliberately numpy-only — no librosa — to keep installs light."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100


def write_wav(path: Path, samples: np.ndarray, sr: int = SAMPLE_RATE, *,
              subtype: str = "PCM_24") -> Path:
    """Write float32 audio as a stereo WAV, shaped (n, 2).

    Buses are written as FLOAT rather than PCM_24: once cues are normalised and gained
    relative to dialogue, a hard-panned transition can exceed 0 dBFS on one channel, and
    PCM_24 would clip it here - before the mixdown and limiter ever see it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), to_stereo(samples).astype(np.float32), sr, subtype=subtype)
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return samples, sr


def to_stereo(samples: np.ndarray) -> np.ndarray:
    """Coerce mono/(2,n)/(n,2) into (n, 2)."""
    a = np.asarray(samples, dtype=np.float32)
    if a.ndim == 1:
        return np.stack([a, a], axis=-1)
    if a.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D audio, got shape {a.shape}")
    # stable-audio-tools hands back channels-first; soundfile wants frames-first.
    if a.shape[0] in (1, 2) and a.shape[0] < a.shape[1]:
        a = a.T
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    return a[:, :2]


def silence(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros((max(1, int(round(duration * sr))), 2), dtype=np.float32)


def fit_length(samples: np.ndarray, duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Trim or zero-pad to exactly `duration` seconds."""
    a = to_stereo(samples)
    want = max(1, int(round(duration * sr)))
    if a.shape[0] >= want:
        return a[:want]
    return np.pad(a, ((0, want - a.shape[0]), (0, 0)))


def find_onset(samples: np.ndarray, sr: int = SAMPLE_RATE, threshold: float = 0.15) -> float:
    """Seconds of lead-in before the first real transient.

    Generated one-shots commonly carry 50-200ms of near-silence before the actual hit.
    Placing such a clip at a cut makes the sound arrive *late*. The renderer subtracts
    this offset so the transient lands on the cue time instead of the file merely
    starting there.

    Returns 0.0 when the clip has no clear onset (a pad, a bed, near-silence).
    """
    a = to_stereo(samples)
    if a.size == 0:
        return 0.0
    mono = a.mean(axis=1)
    hop = max(1, sr // 1000)  # 1ms resolution
    frames = mono[: len(mono) - len(mono) % hop].reshape(-1, hop)
    if frames.size == 0:
        return 0.0
    envelope = np.abs(frames).max(axis=1)
    peak = float(envelope.max())
    if peak <= 1e-6:
        return 0.0
    loud = np.flatnonzero(envelope >= peak * threshold)
    if loud.size == 0:
        return 0.0
    onset = float(loud[0] * hop) / sr
    # A late "onset" means the clip is a swell or bed, not a hit; don't shift those.
    return onset if onset < 0.5 else 0.0


def loop_to_length(samples: np.ndarray, duration: float, sr: int = SAMPLE_RATE,
                   crossfade: float = 0.25) -> np.ndarray:
    """Repeat a clip to fill `duration`, using an equal-power crossfade at each seam."""
    a = to_stereo(samples)
    want = max(1, int(round(duration * sr)))
    if a.shape[0] >= want:
        return a[:want]

    xf = min(int(crossfade * sr), a.shape[0] // 2)
    if xf <= 0:
        reps = int(np.ceil(want / a.shape[0]))
        return np.tile(a, (reps, 1))[:want]

    # Equal-power (constant energy) curves; a linear fade would dip in the middle.
    t = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
    fade_out, fade_in = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)

    out = a.copy()
    while out.shape[0] < want:
        head, tail = out[:-xf], out[-xf:]
        seam = tail * fade_out + a[:xf] * fade_in
        out = np.concatenate([head, seam, a[xf:]], axis=0)
    return out[:want]


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


# --------------------------------------------------------------------- normalisation

#: One-shots are normalised to a peak, because a transient's RMS says little about how
#: loud it reads. Beds are normalised to loudness (LUFS), measured with ffmpeg.
REF_SFX_PEAK_DBFS = -12.0

#: A sustained "one-shot" - a riser, a long whoosh - shares a peak with a click but is far
#: louder over time. Capping RMS too stops it arriving much hotter than its neighbours.
REF_SFX_RMS_CEILING_DBFS = -26.0

#: Bounds on corrective gain, so a near-silent or clipped generation cannot be dragged to
#: the reference and turned into noise or distortion.
NORMALIZE_GAIN_CLAMP_DB = (-24.0, 18.0)


def peak_dbfs(samples: np.ndarray) -> float:
    return _db(float(np.abs(to_stereo(samples)).max()))


def rms_dbfs(samples: np.ndarray) -> float:
    """RMS of the mono downmix, matching how the balance report measures buses."""
    mono = to_stereo(samples).mean(axis=1)
    return _db(float(np.sqrt((mono**2).mean())))


def _db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-12)))


def normalize_one_shot(
    samples: np.ndarray,
    *,
    peak_target_dbfs: float = REF_SFX_PEAK_DBFS,
    rms_ceiling_dbfs: float = REF_SFX_RMS_CEILING_DBFS,
    clamp_db: tuple[float, float] = NORMALIZE_GAIN_CLAMP_DB,
) -> tuple[np.ndarray, float]:
    """Bring a generated one-shot to a known reference, returning it and the gain applied.

    This is what makes a cue's `gain_db` mean anything. Diffusion output lands at an
    arbitrary level, so applying a fixed attenuation to it produces a different balance
    every run - which is how a bed ended up 27dB under the dialogue.

    The *more attenuating* of the peak and RMS targets wins, so a long sustained cue
    cannot arrive far louder than a click that happens to share its peak.
    """
    audio = to_stereo(samples)
    peak, rms = peak_dbfs(audio), rms_dbfs(audio)
    if peak <= -119.0:  # silence; nothing to normalise toward
        return audio, 0.0

    gain = min(peak_target_dbfs - peak, rms_ceiling_dbfs - rms)
    gain = float(np.clip(gain, clamp_db[0], clamp_db[1]))
    return audio * db_to_gain(gain), gain


# ------------------------------------------------------------------- tone shaping

def shape(
    samples: np.ndarray,
    sr: int = SAMPLE_RATE,
    *,
    highpass_hz: float | None = None,
    lowpass_hz: float | None = None,
    order: int = 2,
) -> np.ndarray:
    """Butterworth-response high/low pass, applied in the frequency domain.

    Filtering is what makes a generated sound *sit* somewhere. A boomy effect competing
    with a voice usually needs its bottom removed rather than its level dropped, and a
    harsh one needs its top rolled off - both leave the sound recognisable where turning
    it down just makes it quiet and still wrong.

    FFT rather than a biquad because these clips are short and a Python IIR loop over two
    million samples is slow enough to be felt in the UI.
    """
    if not highpass_hz and not lowpass_hz:
        return samples

    audio = to_stereo(samples)
    n = audio.shape[0]
    if n < 8:
        return audio

    spectrum = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    gain = np.ones_like(freqs)

    with np.errstate(divide="ignore", invalid="ignore"):
        if highpass_hz and highpass_hz > 0:
            ratio = np.divide(highpass_hz, freqs, out=np.full_like(freqs, np.inf),
                              where=freqs > 0)
            gain *= 1.0 / np.sqrt(1.0 + ratio ** (2 * order))
            gain[freqs == 0] = 0.0
        if lowpass_hz and lowpass_hz > 0:
            gain *= 1.0 / np.sqrt(1.0 + (freqs / lowpass_hz) ** (2 * order))

    return np.fft.irfft(spectrum * gain[:, None], n=n, axis=0).astype(np.float32)


def pitch_shift(samples: np.ndarray, semitones: float) -> np.ndarray:
    """Varispeed: resample so the sound changes pitch and length together.

    Deliberately not pitch-preserving. Slowing a sound down *and* dropping it is how a
    small click becomes a heavy thunk, and that tape-style behaviour is more useful on
    one-shots than a formant-correct shift would be.
    """
    if abs(semitones) < 1e-6:
        return to_stereo(samples)

    audio = to_stereo(samples)
    rate = 2.0 ** (semitones / 12.0)
    src_len = audio.shape[0]
    out_len = max(8, int(round(src_len / rate)))

    positions = np.linspace(0.0, src_len - 1, out_len)
    index = np.arange(src_len, dtype=np.float64)
    return np.stack(
        [np.interp(positions, index, audio[:, ch]) for ch in range(audio.shape[1])],
        axis=-1,
    ).astype(np.float32)


def reverse(samples: np.ndarray) -> np.ndarray:
    """Play the clip backwards. Turns a decay into a swell, which is how risers are made."""
    return to_stereo(samples)[::-1].copy()
