"""Stage 1: turn a video into facts a sound designer can act on.

Everything here is deterministic - ffprobe metadata, shot boundaries, a motion-energy
curve, whether the source already carries speech, and contact sheets with burned-in
timecodes. The contact sheets are what Claude reads in order to write the cue sheet, so
the timecodes matter: they are how a described moment becomes an exact cue time.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .schema import VideoInfo

#: Contact sheet layout. 3x3 keeps each frame legible while covering a lot of ground.
GRID = (3, 3)
TILE_WIDTH = 480


class AnalysisError(RuntimeError):
    pass


@dataclass
class Shot:
    index: int
    start: float
    end: float
    #: Mean inter-frame difference across the shot, 0..1. High means lots of movement.
    motion: float = 0.0
    #: Largest single-frame jump inside the shot - a hit, an impact, a fast reveal.
    peak_motion: float = 0.0
    peak_at: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Analysis:
    video: VideoInfo
    shots: list[Shot]
    #: (time, energy) sampled at a fixed rate, for spotting movement inside a long shot.
    motion_curve: list[tuple[float, float]]
    #: Time ranges where the source audio is loud enough to likely be speech.
    speech_ranges: list[tuple[float, float]]
    contact_sheets: list[str]

    def to_json(self) -> dict:
        return {
            "video": self.video.model_dump(mode="json"),
            "shots": [asdict(s) for s in self.shots],
            "motion_curve": [[round(t, 3), round(e, 4)] for t, e in self.motion_curve],
            "speech_ranges": [[round(a, 3), round(b, 3)] for a, b in self.speech_ranges],
            "contact_sheets": self.contact_sheets,
        }


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AnalysisError(f"{cmd[0]} failed: {proc.stderr.strip()[-800:]}")
    return proc


def probe(path: Path) -> VideoInfo:
    """Read container metadata with ffprobe."""
    path = Path(path)
    if not path.exists():
        raise AnalysisError(f"no such video: {path}")

    proc = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise AnalysisError(f"{path.name} contains no video stream")

    num, _, den = video.get("r_frame_rate", "30/1").partition("/")
    denom = float(den or 1)
    fps = float(num) / denom if denom else 30.0
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise AnalysisError(f"could not determine duration of {path.name}")

    return VideoInfo(
        path=str(path.resolve()),
        duration=duration,
        fps=fps,
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def motion_curve(path: Path, fps: float = 8.0) -> list[tuple[float, float]]:
    """Mean absolute inter-frame difference, sampled at the given rate.

    Decoded straight to tiny greyscale frames through a pipe, so this stays fast even on
    long footage and needs no extra dependencies.
    """
    w, h = 64, 36
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    )
    frame_bytes = w * h
    count = len(proc.stdout) // frame_bytes
    if count < 2:
        return []

    frames = np.frombuffer(proc.stdout[: count * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(count, h * w).astype(np.float32) / 255.0
    diffs = np.abs(np.diff(frames, axis=0)).mean(axis=1)
    return [(round(i / fps, 3), float(d)) for i, d in enumerate(diffs, start=1)]


def _seconds(timecode) -> float:
    """PySceneDetect 0.7 exposes `.seconds`; older releases only have get_seconds()."""
    value = getattr(timecode, "seconds", None)
    return float(value) if value is not None else float(timecode.get_seconds())


def detect_shots(path: Path, info: VideoInfo, threshold: float = 27.0) -> list[Shot]:
    """Shot boundaries via PySceneDetect's content detector, with a whole-video fallback."""
    try:
        from scenedetect import ContentDetector, detect
    except ImportError:  # pragma: no cover - scenedetect is a hard dependency
        return [Shot(index=0, start=0.0, end=info.duration)]

    try:
        scenes = detect(str(path), ContentDetector(threshold=threshold))
    except Exception:
        scenes = []

    if not scenes:
        return [Shot(index=0, start=0.0, end=info.duration)]

    return [
        Shot(index=i, start=_seconds(start), end=_seconds(end))
        for i, (start, end) in enumerate(scenes)
    ]


def annotate_shots(shots: list[Shot], curve: list[tuple[float, float]]) -> None:
    """Attach motion statistics to each shot, in place."""
    if not curve:
        return
    times = np.array([t for t, _ in curve])
    energy = np.array([e for _, e in curve])

    for shot in shots:
        mask = (times >= shot.start) & (times < shot.end)
        # Skip the boundary sample: the cut itself is a huge diff and would swamp the shot.
        drop_first = mask.sum() > 1
        window = energy[mask][1:] if drop_first else energy[mask]
        window_times = times[mask][1:] if drop_first else times[mask]
        if window.size == 0:
            continue
        shot.motion = round(float(window.mean()), 4)
        peak = int(window.argmax())
        shot.peak_motion = round(float(window[peak]), 4)
        shot.peak_at = round(float(window_times[peak]), 3)


def detect_speech(path: Path, info: VideoInfo) -> list[tuple[float, float]]:
    """Time ranges carrying probable speech, derived from ffmpeg silencedetect.

    Deliberately a heuristic - the point is to know where the mix must get out of the way,
    which non-silence answers well enough without loading a speech recognition model.
    """
    if not info.has_audio:
        return []

    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path),
         "-af", "silencedetect=noise=-32dB:d=0.35", "-f", "null", "-"],
        capture_output=True, text=True,
    )

    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.rsplit("silence_start:", 1)[1].strip().split()[0])
        elif "silence_end:" in line and start is not None:
            silences.append((start, float(line.rsplit("silence_end:", 1)[1].strip().split()[0])))
            start = None
    if start is not None:
        silences.append((start, info.duration))

    # Invert: everything that is not silence is candidate speech.
    speech: list[tuple[float, float]] = []
    cursor = 0.0
    for s_start, s_end in silences:
        if s_start - cursor > 0.3:
            speech.append((round(cursor, 3), round(s_start, 3)))
        cursor = s_end
    if info.duration - cursor > 0.3:
        speech.append((round(cursor, 3), round(info.duration, 3)))
    return speech


def _label_font(size: int = 17):
    """A readable label font, falling back to PIL's bitmap default if none is installed.

    Timecodes are the whole point of these sheets - they have to be legible.
    """
    for candidate in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _grab_frame(path: Path, at: float, dest: Path, width: int = TILE_WIDTH) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"scale={width}:-2", str(dest)],
        capture_output=True,
    )
    return proc.returncode == 0 and dest.exists()


def contact_sheets(path: Path, shots: list[Shot], out_dir: Path,
                   per_shot: int = 2) -> list[str]:
    """Build labelled frame grids - the visual input for writing the cue sheet.

    Every tile is stamped with its shot index and exact timecode, so a moment seen in a
    sheet can be turned into a precise cue time without guessing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_frames"
    tmp_dir.mkdir(exist_ok=True)

    # Sample points spread inside each shot, avoiding the transition frames at its edges.
    samples: list[tuple[int, float]] = []
    for shot in shots:
        n = 1 if shot.duration < 1.5 else per_shot
        for k in range(n):
            frac = (k + 1) / (n + 1)
            samples.append((shot.index, shot.start + shot.duration * frac))

    tiles: list[tuple[int, float, Path]] = []
    for shot_idx, at in samples:
        frame = tmp_dir / f"s{shot_idx:03d}_{at:07.2f}.jpg"
        if _grab_frame(path, at, frame):
            tiles.append((shot_idx, at, frame))

    if not tiles:
        raise AnalysisError("could not extract any frames; is the video readable?")

    cols, rows = GRID
    per_sheet = cols * rows
    sheets: list[str] = []

    font = _label_font()

    for sheet_no, start in enumerate(range(0, len(tiles), per_sheet)):
        chunk = tiles[start : start + per_sheet]
        with Image.open(chunk[0][2]) as probe_img:
            tw, th = probe_img.size
        label_h = 30
        # Size to the tiles actually present, so a partial last sheet has no dead space.
        used_rows = (len(chunk) + cols - 1) // cols
        used_cols = min(cols, len(chunk))
        sheet = Image.new("RGB", (used_cols * tw, used_rows * (th + label_h)), (18, 18, 20))
        draw = ImageDraw.Draw(sheet)

        for i, (shot_idx, at, frame_path) in enumerate(chunk):
            col, row = i % cols, i // cols
            x, y = col * tw, row * (th + label_h)
            with Image.open(frame_path) as img:
                sheet.paste(img.resize((tw, th)), (x, y + label_h))
            draw.text((x + 8, y + 7), f"shot {shot_idx}   t={at:.2f}s",
                      fill=(255, 214, 102), font=font)

        dest = out_dir / f"sheet_{sheet_no:02d}.jpg"
        sheet.save(dest, quality=88)
        sheets.append(dest.name)

    for _, _, frame_path in tiles:
        frame_path.unlink(missing_ok=True)
    tmp_dir.rmdir()
    return sheets


def analyze(path: Path, out_dir: Path) -> Analysis:
    """Run the full analysis and write analysis.json plus contact sheets."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(path)
    curve = motion_curve(path)
    shots = detect_shots(path, info)
    annotate_shots(shots, curve)
    speech = detect_speech(path, info)
    info.has_dialogue = bool(speech) and sum(b - a for a, b in speech) > info.duration * 0.15
    sheets = contact_sheets(path, shots, out_dir / "contact")

    result = Analysis(
        video=info, shots=shots, motion_curve=curve,
        speech_ranges=speech, contact_sheets=sheets,
    )
    (out_dir / "analysis.json").write_text(
        json.dumps(result.to_json(), indent=2) + "\n", encoding="utf-8"
    )
    return result
