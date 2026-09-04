"""Shared fixtures.

The fixture video is generated with ffmpeg rather than committed, so the suite runs
anywhere with no binary assets and no network. It has known hard cuts, which is what
makes shot detection and cue placement assertable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

#: Hard cuts at 4s and 8s: three visually distinct 4-second segments.
CUT_TIMES = (4.0, 8.0)
FIXTURE_DURATION = 12.0


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


requires_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")


@pytest.fixture(scope="session")
def fixture_video(tmp_path_factory) -> Path:
    """A 12s silent video: three 4s segments of clearly different content."""
    out = tmp_path_factory.mktemp("fixtures") / "cuts.mp4"
    segments = [
        "testsrc=size=320x180:rate=30:duration=4",
        "smptebars=size=320x180:rate=30:duration=4",
        "testsrc2=size=320x180:rate=30:duration=4",
    ]
    inputs: list[str] = []
    for spec in segments:
        inputs += ["-f", "lavfi", "-i", spec]

    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs,
         "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
         "-map", "[v]", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def fixture_video_with_speech(tmp_path_factory) -> Path:
    """Same picture, plus an audio track that alternates tone and silence.

    Stands in for dialogue: the mixer only needs to know where the source is loud.
    """
    out = tmp_path_factory.mktemp("fixtures") / "spoken.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30:duration=12",
         "-f", "lavfi", "-i",
         "sine=frequency=300:duration=12,volume='if(lt(mod(t,4),2.5),0.4,0.0)':eval=frame",
         "-map", "0:v", "-map", "1:a", "-pix_fmt", "yuv420p", "-shortest", str(out)],
        check=True, capture_output=True,
    )
    return out
