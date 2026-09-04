"""End-to-end: analysis through to a mastered, muxed video.

Runs entirely on the `silence` stub engine, so the whole pipeline is exercised offline
with no weights and no GPU. What is asserted is the deterministic half — shot detection,
cue placement, loudness, provenance — which is where mistakes would otherwise be silent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from saand import analyze as analyze_mod
from saand import mix, render
from saand.audio import SAMPLE_RATE, read_wav
from saand.schema import BedCue, CueSheet, SfxCue

from .conftest import CUT_TIMES, FIXTURE_DURATION, requires_ffmpeg

pytestmark = requires_ffmpeg


def sheet_for(info, **kw) -> CueSheet:
    defaults = dict(
        preset="short-form",
        music=[BedCue(id="bed", start=0.0, end=info.duration, prompt="steady background bed",
                      engine="silence", gain_db=-18.0, seed=1)],
        sfx=[SfxCue(id="hit1", at=4.0, duration=1.0, prompt="whoosh on the cut",
                    engine="silence", gain_db=-6.0, seed=2),
             SfxCue(id="hit2", at=8.0, duration=1.0, prompt="second whoosh",
                    engine="silence", gain_db=-6.0, seed=3)],
    )
    defaults.update(kw)
    return CueSheet(video=info, **defaults)


class TestAnalysis:
    def test_probe_reads_duration_and_fps(self, fixture_video):
        info = analyze_mod.probe(fixture_video)
        assert info.duration == pytest.approx(FIXTURE_DURATION, abs=0.2)
        assert info.fps == pytest.approx(30.0, abs=0.1)
        assert not info.has_audio

    def test_finds_the_known_cuts(self, fixture_video):
        info = analyze_mod.probe(fixture_video)
        shots = analyze_mod.detect_shots(fixture_video, info)
        assert len(shots) >= 3, "should find the three segments"

        boundaries = [s.start for s in shots[1:]]
        for expected in CUT_TIMES:
            assert any(abs(b - expected) < 0.4 for b in boundaries), \
                f"no shot boundary near the {expected}s cut; found {boundaries}"

    def test_motion_curve_is_produced(self, fixture_video):
        curve = analyze_mod.motion_curve(fixture_video)
        assert len(curve) > 10
        assert all(0.0 <= energy <= 1.0 for _, energy in curve)

    def test_contact_sheets_are_written(self, fixture_video, tmp_path):
        result = analyze_mod.analyze(fixture_video, tmp_path)
        assert result.contact_sheets
        for name in result.contact_sheets:
            assert (tmp_path / "contact" / name).stat().st_size > 0
        assert json.loads((tmp_path / "analysis.json").read_text())["shots"]

    def test_detects_speech_when_present(self, fixture_video_with_speech):
        info = analyze_mod.probe(fixture_video_with_speech)
        assert info.has_audio
        ranges = analyze_mod.detect_speech(fixture_video_with_speech, info)
        assert ranges, "alternating tone should register as speech-like"


class TestPlacement:
    def test_sfx_lands_at_the_cue_time(self, tmp_path):
        """The core timing guarantee: energy appears where the cue says, not elsewhere."""
        from saand.audio import write_wav

        clip = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        clip[:1000] = 0.8  # immediate transient, no lead-in
        stem = write_wav(tmp_path / "s.wav", clip)

        cue = SfxCue(id="s", at=3.0, duration=1.0, prompt="hit",
                     engine="silence", gain_db=0.0, align_transient=False)
        bus = mix.place_sfx([cue], {"s": stem}, total=6.0)

        energy = np.abs(bus).mean(axis=1)
        loud = np.flatnonzero(energy > 0.1)
        assert loud.size, "nothing was placed"
        assert loud[0] / SAMPLE_RATE == pytest.approx(3.0, abs=0.02)

    def test_transient_alignment_compensates_for_lead_in(self, tmp_path):
        """A clip with 200ms of silence up front must still hit exactly on the cue."""
        from saand.audio import write_wav

        clip = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        clip[int(0.2 * SAMPLE_RATE) : int(0.2 * SAMPLE_RATE) + 1000] = 0.8
        stem = write_wav(tmp_path / "s.wav", clip)

        cue = SfxCue(id="s", at=3.0, duration=1.0, prompt="hit",
                     engine="silence", gain_db=0.0, align_transient=True)
        bus = mix.place_sfx([cue], {"s": stem}, total=6.0)

        loud = np.flatnonzero(np.abs(bus).mean(axis=1) > 0.1)
        assert loud[0] / SAMPLE_RATE == pytest.approx(3.0, abs=0.03), \
            "transient did not land on the cue time"

    def test_gain_is_raw_attenuation_when_normalization_is_off(self, tmp_path):
        """`normalize=False` keeps the original meaning, for version-1 cue sheets."""
        from saand.audio import write_wav

        stem = write_wav(tmp_path / "s.wav",
                         np.ones((SAMPLE_RATE, 2), dtype=np.float32) * 0.5)
        cue = SfxCue(id="s", at=0.0, duration=1.0, prompt="tone",
                     engine="silence", gain_db=-6.0, align_transient=False)
        bus = mix.place_sfx([cue], {"s": stem}, total=2.0, normalize=False)
        assert np.abs(bus[:1000]).max() == pytest.approx(0.25, abs=0.02)

    def test_gain_is_relative_to_the_normalized_reference(self, tmp_path):
        """With normalization on, a cue lands at reference + gain whatever it generated at.

        This is the property the whole fix rests on: two stems generated 20 dB apart must
        end up at the same place, so a preset gain means something.
        """
        from saand.audio import REF_SFX_PEAK_DBFS, peak_dbfs, write_wav

        levels = {}
        # 12 dB apart, not 30: a wider spread hits the +-18 dB corrective clamp, which is
        # there on purpose so a near-silent generation is not dragged up into noise.
        for name, amp in (("quiet", 0.15), ("loud", 0.6)):
            clip = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
            clip[:2000] = amp                      # a transient, not a constant tone
            stem = write_wav(tmp_path / f"{name}.wav", clip)
            cue = SfxCue(id=name, at=0.0, duration=1.0, prompt="hit",
                         engine="silence", gain_db=-6.0, align_transient=False)
            bus = mix.place_sfx([cue], {name: stem}, total=2.0)
            levels[name] = peak_dbfs(bus)

        assert levels["quiet"] == pytest.approx(REF_SFX_PEAK_DBFS - 6.0, abs=1.5)
        assert levels["loud"] == pytest.approx(levels["quiet"], abs=0.5),             "stems generated at different levels must land at the same place"

    def test_sustained_cue_is_held_back_by_the_rms_ceiling(self, tmp_path):
        """A long sustained cue must not arrive as loud as a click sharing its peak."""
        from saand.audio import normalize_one_shot, rms_dbfs

        click = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        click[:500] = 0.5
        drone = np.ones((SAMPLE_RATE, 2), dtype=np.float32) * 0.5

        _, click_gain = normalize_one_shot(click)
        normalized_drone, drone_gain = normalize_one_shot(drone)
        assert drone_gain < click_gain, "the sustained cue should be attenuated further"
        assert rms_dbfs(normalized_drone) == pytest.approx(-26.0, abs=0.5)

    def test_cue_past_the_end_is_clipped_not_crashed(self, tmp_path):
        from saand.audio import write_wav

        stem = write_wav(tmp_path / "s.wav", np.ones((2 * SAMPLE_RATE, 2), dtype=np.float32))
        cue = SfxCue(id="s", at=4.5, duration=2.0, prompt="tail",
                     engine="silence", gain_db=0.0, align_transient=False)
        bus = mix.place_sfx([cue], {"s": stem}, total=5.0)
        assert bus.shape[0] == 5 * SAMPLE_RATE


@pytest.fixture(scope="module")
def rendered(fixture_video, tmp_path_factory):
    """One full render, shared across the assertions below."""
    out = tmp_path_factory.mktemp("render")
    info = analyze_mod.probe(fixture_video)
    sheet = sheet_for(info)
    sheet.save(out / "cues.json")
    return render.render(sheet, out), out, sheet


class TestFullRender:
    def test_produces_a_playable_video(self, rendered):
        result, _, _ = rendered
        assert result.final_video.exists()
        assert result.final_video.stat().st_size > 1000

    def test_output_duration_matches_the_source(self, rendered):
        result, _, sheet = rendered
        info = analyze_mod.probe(result.final_video)
        assert info.duration == pytest.approx(sheet.video.duration, abs=0.3)

    def test_masters_to_the_loudness_target(self, rendered):
        """Landing off target is what makes a platform re-normalise the upload."""
        result, _, sheet = rendered
        assert result.measured_lufs == pytest.approx(sheet.target_lufs, abs=1.0)

    def test_respects_the_true_peak_ceiling(self, rendered):
        result, _, sheet = rendered
        assert result.measured_peak_db <= sheet.true_peak_db + 0.5

    def test_every_stem_is_accounted_for(self, rendered):
        """An untracked stem means audio of unknown licence in the mix."""
        result, out, sheet = rendered
        records = json.loads(result.provenance.read_text())["records"]
        assert len(records) == len(sheet.all_cues())
        assert all(r["licence"] for r in records)

        tracked = {r["file"] for r in records}
        on_disk = {p.name for p in (out / "stems").rglob("*.wav")}
        assert on_disk == tracked

    def test_licence_report_is_human_readable(self, rendered):
        result, _, _ = rendered
        report = result.licence_report.read_text(encoding="utf-8")
        assert "# Audio licence report" in report
        assert "Commercial use" in report

    def test_rerender_reuses_cached_stems(self, rendered):
        """Editing one cue must not re-roll the whole mix."""
        _, out, sheet = rendered
        again = render.render(sheet, out)
        assert again.generated_count == 0
        assert again.cached_count == len(sheet.all_cues())

    def test_changing_a_prompt_regenerates_only_that_cue(self, rendered):
        _, out, sheet = rendered
        edited = sheet.model_copy(deep=True)
        edited.sfx[0].prompt = "a completely different sound"
        result = render.render(edited, out)
        assert result.generated_count == 1
        assert result.cached_count == len(edited.all_cues()) - 1


class TestLicenceEnforcementInRender:
    def test_render_refuses_a_restricted_engine_without_opt_in(self, fixture_video, tmp_path):
        from saand.engines.base import LicenceError

        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info, sfx=[
            SfxCue(id="s", at=1.0, duration=1.0, prompt="footsteps",
                   engine="hunyuan-foley", seed=1)
        ])
        with pytest.raises(LicenceError, match="opt-in"):
            render.render(sheet, tmp_path)

    def test_gate_runs_before_any_generation(self, fixture_video, tmp_path):
        """Fail fast: a licence problem must not surface after a long render."""
        from saand.engines.base import LicenceError

        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info, sfx=[
            SfxCue(id="ok", at=1.0, duration=1.0, prompt="fine", engine="silence", seed=1),
            SfxCue(id="bad", at=2.0, duration=1.0, prompt="restricted",
                   engine="hunyuan-foley", seed=2),
        ])
        with pytest.raises(LicenceError):
            render.render(sheet, tmp_path)
        assert not list((tmp_path / "stems").rglob("*.wav")) if (tmp_path / "stems").exists() \
            else True


class TestMixBalance:
    """The layers must actually be audible.

    A render can master perfectly to -14 LUFS and still contain music 27 dB and effects
    20 dB under the dialogue - which is to say, contain neither. Loudness compliance says
    nothing about balance, so balance is asserted separately.
    """

    def test_duck_threshold_tracks_the_key_instead_of_being_fixed(self):
        """A fixed threshold near -10 dBFS never opened on dialogue around -17."""
        quiet_th, _ = mix._duck_params(-6.0, key_ref_lufs=-24.0)
        loud_th, _ = mix._duck_params(-6.0, key_ref_lufs=-10.0)
        assert loud_th > quiet_th, "threshold must follow the key signal's level"

    def test_deeper_duck_asks_for_a_higher_ratio(self):
        _, gentle = mix._duck_params(-4.0, key_ref_lufs=-16.0)
        _, hard = mix._duck_params(-12.0, key_ref_lufs=-16.0)
        assert hard > gentle
        assert gentle >= 1.0, "ffmpeg's minimum ratio is 1; a floor of 2 over-ducks"

    def test_duck_survives_an_unmeasurable_key(self):
        threshold, ratio = mix._duck_params(-6.0, key_ref_lufs=None)
        assert 0 < threshold <= 1.0 and ratio >= 1.0

    def test_unmeasurable_bed_loudness_does_not_silence_it(self, tmp_path, monkeypatch):
        """A NaN loudness reading must not propagate into the gain and zero the stem."""
        from saand.audio import peak_dbfs, write_wav

        clip = np.ones((SAMPLE_RATE, 2), dtype=np.float32) * 0.3
        stem = write_wav(tmp_path / "bed.wav", clip)
        monkeypatch.setattr(mix, "measured_lufs", lambda path: None)

        bed = BedCue(id="bed", start=0.0, end=1.0, prompt="a bed",
                     engine="silence", gain_db=0.0, fade_in=0.0, fade_out=0.0)
        bus = mix.place_beds([bed], {"bed": stem}, total=1.0)
        assert peak_dbfs(bus) > -60.0, "an unmeasurable bed was silenced"

    def test_balance_report_flags_a_buried_mix(self, tmp_path):
        from saand import balance as balance_mod
        from saand.audio import write_wav

        buses = tmp_path / "buses"
        loud = np.random.default_rng(0).standard_normal((SAMPLE_RATE * 2, 2)).astype(np.float32) * 0.3
        write_wav(buses / "original.wav", loud)
        write_wav(buses / "music.wav", loud * 0.002)      # ~54 dB down

        report = balance_mod.measure(tmp_path)
        assert report is not None
        assert any("music" in p for p in report.problems())

    def test_balance_report_passes_a_sane_mix(self, tmp_path):
        from saand import balance as balance_mod
        from saand.audio import write_wav

        buses = tmp_path / "buses"
        rng = np.random.default_rng(0)
        source = rng.standard_normal((SAMPLE_RATE * 2, 2)).astype(np.float32) * 0.3
        write_wav(buses / "original.wav", source)
        write_wav(buses / "music.wav", source * 0.2)      # 14 dB down - audible

        report = balance_mod.measure(tmp_path)
        assert not [p for p in report.problems() if "music" in p]

    def test_silent_footage_has_no_balance_to_report(self, tmp_path):
        from saand import balance as balance_mod

        (tmp_path / "buses").mkdir(parents=True)
        assert balance_mod.measure(tmp_path) is None
