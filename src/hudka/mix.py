"""Stage 3: place, duck, master, mux.

Split by what each tool is actually good at. Placement happens in numpy, because exact
sample positions, transient alignment, fades and loops are easy to get right and easy to
assert on. Dynamics and loudness happen in ffmpeg, because sidechain compression and
EBU R128 normalisation are solved problems there and reimplementing them would be worse.

The result is a mix mastered to the preset's LUFS target with a true-peak ceiling, which
is what stops YouTube from turning the whole thing down on upload.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .audio import (
    SAMPLE_RATE,
    db_to_gain,
    pitch_shift,
    reverse as reverse_audio,
    rms_dbfs,
    shape,
    find_onset,
    fit_length,
    loop_to_length,
    normalize_one_shot,
    read_wav,
    to_stereo,
    write_wav,
)
from .schema import BedCue, CueSheet, SfxCue

#: Loudness every music/ambience bed is brought to before its cue gain applies.
REF_BED_LUFS = -20.0

#: Crossfade at a looped bed's seam. The 0.25s default is right for a looped room tone
#: and wrong for music: a 120s bed under a 218s video produced a hard musical seam at
#: 2:00, mid-phrase. Two seconds is long enough to read as an arrangement, not a splice.
BED_LOOP_CROSSFADE_S = 2.0


class MixError(RuntimeError):
    pass


def measured_lufs(path: Path) -> float | None:
    """Integrated loudness of a file, or None when it cannot be measured.

    Never let a non-finite value reach a gain calculation: NaN propagates silently
    through the multiply and produces a stem of pure silence, which looks exactly like a
    model that generated nothing.
    """
    try:
        value, _ = integrated_lufs(path)
    except Exception:
        return None
    return float(value) if np.isfinite(value) else None


def _run(cmd: list[str], *, text: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=text)
    if proc.returncode != 0:
        err = proc.stderr if text else proc.stderr.decode("utf-8", "replace")
        raise MixError(f"ffmpeg failed:\n{err.strip()[-2000:]}")
    return proc


# --------------------------------------------------------------------------- placement


def _apply_pan(samples: np.ndarray, pan: float) -> np.ndarray:
    """Constant-power pan. -1 is hard left, 0 centre, +1 hard right."""
    if abs(pan) < 1e-6:
        return samples
    angle = (pan + 1.0) * np.pi / 4.0  # 0..pi/2
    out = samples.copy()
    out[:, 0] *= float(np.cos(angle)) * np.sqrt(2)
    out[:, 1] *= float(np.sin(angle)) * np.sqrt(2)
    return out


def _apply_fades(samples: np.ndarray, fade_in: float, fade_out: float) -> np.ndarray:
    out = samples.copy()
    n = out.shape[0]
    n_in = min(int(fade_in * SAMPLE_RATE), n)
    n_out = min(int(fade_out * SAMPLE_RATE), n - n_in)
    if n_in > 0:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)[:, None]
    if n_out > 0:
        out[n - n_out :] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)[:, None]
    return out


def _add_at(bus: np.ndarray, clip: np.ndarray, at_samples: int) -> None:
    """Sum `clip` into `bus` starting at a sample offset, clipping to the bus length."""
    if at_samples >= bus.shape[0]:
        return
    start = max(0, at_samples)
    clip_start = start - at_samples
    length = min(clip.shape[0] - clip_start, bus.shape[0] - start)
    if length <= 0:
        return
    bus[start : start + length] += clip[clip_start : clip_start + length]


def _shaped(clip: np.ndarray, cue) -> np.ndarray:
    """Apply a cue's tone controls, keeping the result level-neutral.

    Filtering removes energy, so a high-pass would otherwise quietly turn a cue down as
    well as thinning it - and the user would reach for the gain to compensate, undoing
    the point of normalised levels. Measuring either side and adding the difference back
    keeps "how it sounds" and "how loud it is" as separate controls.
    """
    if cue.reverse:
        clip = reverse_audio(clip)
    if cue.pitch_semitones:
        clip = pitch_shift(clip, cue.pitch_semitones)

    if cue.highpass_hz or cue.lowpass_hz:
        before = rms_dbfs(clip)
        clip = shape(clip, SAMPLE_RATE,
                     highpass_hz=cue.highpass_hz, lowpass_hz=cue.lowpass_hz)
        after = rms_dbfs(clip)
        if np.isfinite(before) and np.isfinite(after) and after > -110:
            clip = clip * db_to_gain(min(before - after, 24.0))
    return clip


def place_sfx(cues: list[SfxCue], stems: dict[str, Path], total: float, *,
              bus_offset_db: float = 0.0, normalize: bool = True) -> np.ndarray:
    """Sum every one-shot into a single SFX bus.

    Order matters: normalise the raw clip first, then align, fade and pan. Normalising
    after panning would break the reference, since a hard pan adds up to 3 dB on one side.
    """
    bus = np.zeros((int(round(total * SAMPLE_RATE)), 2), dtype=np.float32)

    for cue in cues:
        if cue.muted:
            continue
        samples, _ = read_wav(stems[cue.id])
        clip = to_stereo(samples)
        clip = clip - clip.mean(axis=0)   # a DC offset is a thump at every fade edge
        clip = _shaped(clip, cue)

        if normalize:
            clip, _ = normalize_one_shot(clip)

        # Land the transient on the cue time, not the file start. Without this a clip with
        # 150ms of lead-in arrives audibly late against a cut.
        offset = find_onset(clip) if cue.align_transient else 0.0
        at = cue.at - offset

        # A tiny tail fade always, to avoid a click; the cue's own fades on top of it.
        clip = _apply_fades(clip, cue.fade_in, max(cue.fade_out, min(0.02, cue.duration / 4)))
        clip = _apply_pan(clip, cue.pan) * db_to_gain(cue.gain_db + bus_offset_db)
        _add_at(bus, clip, int(round(at * SAMPLE_RATE)))

    return bus


def place_beds(beds: list[BedCue], stems: dict[str, Path], total: float, *,
               bus_offset_db: float = 0.0, normalize: bool = True,
               reference_lufs: float = REF_BED_LUFS) -> np.ndarray:
    """Sum music or ambience beds into one bus.

    Loudness is measured on the raw file before looping or fading: measuring after fades
    reads low, and measuring after looping pays for the same answer several times over.
    """
    bus = np.zeros((int(round(total * SAMPLE_RATE)), 2), dtype=np.float32)

    for bed in beds:
        if bed.muted:
            continue
        source = stems[bed.id]
        samples, _ = read_wav(source)
        clip = to_stereo(samples)
        clip = clip - clip.mean(axis=0)

        norm_db = 0.0
        if normalize:
            lufs = measured_lufs(source)
            if lufs is not None:
                norm_db = float(np.clip(reference_lufs - lufs, -24.0, 18.0))

        clip = _shaped(clip, bed)

        clip = (loop_to_length(clip, bed.duration, crossfade=BED_LOOP_CROSSFADE_S)
                if bed.loop else fit_length(clip, bed.duration))
        clip = _apply_fades(clip, bed.fade_in, bed.fade_out)
        clip = clip * db_to_gain(norm_db + bed.gain_db + bus_offset_db)
        _add_at(bus, clip, int(round(bed.start * SAMPLE_RATE)))

    return bus


# ----------------------------------------------------------------------------- ffmpeg


def extract_original_audio(video: Path, dest: Path, total: float) -> Path | None:
    """Pull the source audio out as a stem, so dialogue survives the rebuild."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vn", "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s24le", str(dest)],
        capture_output=True,
    )
    if proc.returncode != 0 or not dest.exists():
        return None
    samples, _ = read_wav(dest)
    return write_wav(dest, fit_length(samples, total))


#: The duck threshold sits this far below the key signal's measured loudness, so it
#: actually engages on speech instead of waiting for a level the dialogue never reaches.
DUCK_THRESHOLD_UNDER_KEY_DB = 14.0

#: How far over threshold the key typically sits, calibrated by measuring the real filter
#: against real buses. Converts a requested depth into the ratio that delivers it.
DUCK_EFFECTIVE_OVER_DB = 16.0


def _duck_params(depth_db: float, key_ref_lufs: float | None) -> tuple[float, float]:
    """Threshold and ratio that deliver approximately `depth_db` of gain reduction.

    The previous version used a fixed threshold near -10 dBFS. Dialogue sits closer to
    -17 dBFS, so the compressor almost never opened: measured against the real buses it
    delivered 0.56 dB median during speech while claiming 14. Anchoring the threshold to
    the key's own measured loudness is what makes the requested depth mean something -
    which in turn is why the preset depths had to come down when this was fixed.
    """
    depth = min(abs(depth_db), 15.0)
    ratio = float(np.clip(1.0 / max(1e-6, 1.0 - depth / DUCK_EFFECTIVE_OVER_DB), 1.0, 20.0))

    reference = key_ref_lufs if key_ref_lufs is not None else -17.0
    threshold = float(np.clip(
        10 ** ((reference - DUCK_THRESHOLD_UNDER_KEY_DB) / 20.0), 0.000977, 1.0
    ))
    return threshold, ratio


def mixdown(*, music: Path | None, sfx: Path | None, ambience: Path | None,
            dialogue: Path | None, dest: Path, duck_depth_db: float,
            original_gain_db: float = 0.0, key_ref_lufs: float | None = None) -> Path:
    """Combine the buses, ducking the music under dialogue and effects.

    Dialogue and effects are each needed twice when ducking: once as audio in the mix,
    once as the sidechain key. An ffmpeg input pad can only be consumed once, so those
    stems are explicitly `asplit` into a main leg and a key leg.
    """
    inputs: list[str] = []
    src: dict[str, str] = {}
    for name, path in (("music", music), ("sfx", sfx),
                       ("ambience", ambience), ("dialogue", dialogue)):
        if path is not None and Path(path).exists():
            src[name] = f"{len(inputs) // 2}:a"
            inputs += ["-i", str(path)]

    if not src:
        raise MixError("nothing to mix: no stems were produced")

    graph: list[str] = []
    main: dict[str, str] = {}
    key_legs: list[str] = []

    # Ducking only makes sense when there is a bed to duck and something to duck it under.
    ducking = "music" in src and ({"dialogue", "sfx"} & src.keys())

    for name in ("dialogue", "sfx", "ambience"):
        if name not in src:
            continue
        if ducking and name in ("dialogue", "sfx"):
            graph.append(f"[{src[name]}]asplit=2[{name}_m][{name}_k]")
            main[name] = f"{name}_m"
            key_legs.append(f"{name}_k")
        else:
            main[name] = src[name]

    if "dialogue" in main and original_gain_db:
        graph.append(f"[{main['dialogue']}]volume={original_gain_db}dB[dlg_g]")
        main["dialogue"] = "dlg_g"

    if "music" in src:
        if ducking:
            if len(key_legs) > 1:
                graph.append(
                    f"[{']['.join(key_legs)}]amix=inputs={len(key_legs)}:normalize=0[key]"
                )
                key = "key"
            else:
                key = key_legs[0]
            threshold, ratio = _duck_params(duck_depth_db, key_ref_lufs)
            graph.append(
                f"[{src['music']}][{key}]sidechaincompress="
                f"threshold={threshold:.4f}:ratio={ratio:.2f}:"
                f"attack=25:release=320:makeup=1[ducked]"
            )
            main["music"] = "ducked"
        else:
            main["music"] = src["music"]

    legs = [main[n] for n in ("music", "sfx", "ambience", "dialogue") if n in main]
    if len(legs) == 1:
        graph.append(f"[{legs[0]}]anull[out]")
    else:
        graph.append(f"[{']['.join(legs)}]amix=inputs={len(legs)}:normalize=0[out]")

    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-v", "error", *inputs,
          "-filter_complex", ";".join(graph),
          "-map", "[out]", "-ac", "2", "-ar", str(SAMPLE_RATE),
          "-c:a", "pcm_f32le", str(dest)])
    return dest


def measure_loudness(path: Path, target_lufs: float, true_peak: float) -> dict:
    """Loudnorm analysis pass. Returns the measured values for the correction pass."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path),
         "-af", f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    start = proc.stderr.rfind("{")
    end = proc.stderr.rfind("}")
    if start == -1 or end == -1:
        raise MixError(f"could not read loudness measurement:\n{proc.stderr[-1500:]}")
    return json.loads(proc.stderr[start : end + 1])


def normalize(src: Path, dest: Path, target_lufs: float, true_peak: float) -> Path:
    """Second loudnorm pass, using the measured values.

    Two passes rather than one because single-pass loudnorm works from a running estimate
    and routinely lands a decibel or more off target - which is exactly the error that
    makes a platform re-normalise the upload.
    """
    m = measure_loudness(src, target_lufs, true_peak)
    filt = (
        f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11:"
        f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
        f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
        f"offset={m['target_offset']}:linear=true:print_format=summary"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", filt,
          "-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s24le", str(dest)])
    return dest


def integrated_lufs(path: Path) -> tuple[float, float]:
    """Measure a finished file with ebur128. Returns (integrated LUFS, true peak dBTP)."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path),
         "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    lufs, peak = float("nan"), float("nan")
    tail = proc.stderr.splitlines()
    for i, line in enumerate(tail):
        if "Integrated loudness" in line:
            for follow in tail[i : i + 4]:
                if "I:" in follow:
                    lufs = float(follow.split("I:")[1].split("LUFS")[0].strip())
                    break
        if "True peak" in line:
            for follow in tail[i : i + 4]:
                if "Peak:" in follow:
                    peak = float(follow.split("Peak:")[1].split("dBFS")[0].strip())
                    break
    return lufs, peak


def mux(video: Path, audio: Path, dest: Path) -> Path:
    """Put the finished mix back onto the picture, copying the video stream untouched."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(audio),
          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
          "-c:a", "aac", "-b:a", "320k", "-shortest", str(dest)])
    return dest
