"""Cue sheet validation — the errors that would otherwise surface as a bad mix."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hudka.schema import BedCue, CueSheet, SfxCue, VideoInfo


def video(duration: float = 12.0) -> VideoInfo:
    return VideoInfo(path="x.mp4", duration=duration, fps=30.0)


def bed(**kw) -> BedCue:
    return BedCue(**{"id": "b", "start": 0.0, "end": 10.0, "prompt": "warm pad", **kw})


def sfx(**kw) -> SfxCue:
    return SfxCue(**{"id": "s", "at": 1.0, "prompt": "whoosh", **kw})


def test_accepts_a_reasonable_sheet():
    sheet = CueSheet(video=video(), music=[bed()], sfx=[sfx()])
    assert sheet.music[0].duration == 10.0
    assert sheet.sfx[0].end == 2.5


def test_rejects_bed_ending_before_it_starts():
    with pytest.raises(ValidationError, match="must be after start"):
        CueSheet(video=video(), music=[bed(start=5.0, end=2.0)])


def test_rejects_fades_longer_than_the_bed():
    with pytest.raises(ValidationError, match="exceed"):
        CueSheet(video=video(), music=[bed(start=0.0, end=2.0, fade_in=1.5, fade_out=1.5)])


def test_rejects_overlapping_music_beds():
    """Two music beds at once fight each other and wreck the loudness target."""
    with pytest.raises(ValidationError, match="overlap"):
        CueSheet(video=video(), music=[
            bed(id="a", start=0.0, end=8.0),
            bed(id="b", start=5.0, end=12.0),
        ])


def test_allows_overlapping_ambience_beds():
    """Layering room tones is normal, so ambience is exempt from the overlap rule."""
    sheet = CueSheet(video=video(), ambience=[
        bed(id="a", start=0.0, end=8.0),
        bed(id="b", start=5.0, end=12.0),
    ])
    assert len(sheet.ambience) == 2


def test_rejects_cues_past_the_end_of_the_video():
    with pytest.raises(ValidationError, match="past the"):
        CueSheet(video=video(12.0), sfx=[sfx(at=30.0)])


def test_rejects_duplicate_ids():
    """Ids key the stem cache; duplicates would silently share a file."""
    with pytest.raises(ValidationError, match="duplicate cue id"):
        CueSheet(video=video(), sfx=[sfx(id="dup", at=1.0), sfx(id="dup", at=2.0)])


def test_round_trips_through_disk(tmp_path):
    original = CueSheet(video=video(), music=[bed()], sfx=[sfx()])
    path = tmp_path / "cues.json"
    original.save(path)
    assert CueSheet.load(path) == original


def test_gain_is_bounded():
    """A +40dB typo would clip catastrophically; the schema refuses it."""
    with pytest.raises(ValidationError):
        sfx(gain_db=40.0)
