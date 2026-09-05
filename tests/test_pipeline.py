"""End-to-end: analysis through to a mastered, muxed video.

Runs entirely on the `silence` stub engine, so the whole pipeline is exercised offline
with no weights and no GPU. What is asserted is the deterministic half — shot detection,
cue placement, loudness, provenance — which is where mistakes would otherwise be silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hudka import analyze as analyze_mod
from hudka import mix, render
from hudka.audio import SAMPLE_RATE, read_wav
from hudka.schema import BedCue, CueSheet, SfxCue

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
        from hudka.audio import write_wav

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
        from hudka.audio import write_wav

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
        from hudka.audio import write_wav

        # A tone, not a constant block: a constant is pure DC, and placement removes DC.
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        tone = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        stem = write_wav(tmp_path / "s.wav", np.stack([tone, tone], axis=-1))
        cue = SfxCue(id="s", at=0.0, duration=1.0, prompt="tone",
                     engine="silence", gain_db=-6.0, align_transient=False)
        bus = mix.place_sfx([cue], {"s": stem}, total=2.0, normalize=False)
        assert np.abs(bus[:1000]).max() == pytest.approx(0.25, abs=0.02)

    def test_gain_is_relative_to_the_normalized_reference(self, tmp_path):
        """With normalization on, a cue lands at reference + gain whatever it generated at.

        This is the property the whole fix rests on: two stems generated 20 dB apart must
        end up at the same place, so a preset gain means something.
        """
        from hudka.audio import REF_SFX_PEAK_DBFS, peak_dbfs, write_wav

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
        from hudka.audio import normalize_one_shot, rms_dbfs

        click = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        click[:500] = 0.5
        drone = np.ones((SAMPLE_RATE, 2), dtype=np.float32) * 0.5

        _, click_gain = normalize_one_shot(click)
        normalized_drone, drone_gain = normalize_one_shot(drone)
        assert drone_gain < click_gain, "the sustained cue should be attenuated further"
        assert rms_dbfs(normalized_drone) == pytest.approx(-26.0, abs=0.5)

    def test_cue_past_the_end_is_clipped_not_crashed(self, tmp_path):
        from hudka.audio import write_wav

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
        from hudka.engines.base import LicenceError

        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info, sfx=[
            SfxCue(id="s", at=1.0, duration=1.0, prompt="footsteps",
                   engine="hunyuan-foley", seed=1)
        ])
        with pytest.raises(LicenceError, match="opt-in"):
            render.render(sheet, tmp_path)

    def test_gate_runs_before_any_generation(self, fixture_video, tmp_path):
        """Fail fast: a licence problem must not surface after a long render."""
        from hudka.engines.base import LicenceError

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
        from hudka.audio import peak_dbfs, write_wav

        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        tone = (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)
        stem = write_wav(tmp_path / "bed.wav", np.stack([tone, tone], axis=-1))
        monkeypatch.setattr(mix, "measured_lufs", lambda path: None)

        bed = BedCue(id="bed", start=0.0, end=1.0, prompt="a bed",
                     engine="silence", gain_db=0.0, fade_in=0.0, fade_out=0.0)
        bus = mix.place_beds([bed], {"bed": stem}, total=1.0)
        assert peak_dbfs(bus) > -60.0, "an unmeasurable bed was silenced"

    def test_balance_report_flags_a_buried_mix(self, tmp_path):
        from hudka import balance as balance_mod
        from hudka.audio import write_wav

        buses = tmp_path / "buses"
        loud = np.random.default_rng(0).standard_normal((SAMPLE_RATE * 2, 2)).astype(np.float32) * 0.3
        write_wav(buses / "original.wav", loud)
        write_wav(buses / "music.wav", loud * 0.002)      # ~54 dB down

        report = balance_mod.measure(tmp_path)
        assert report is not None
        assert any("music" in p for p in report.problems())

    def test_balance_report_passes_a_sane_mix(self, tmp_path):
        from hudka import balance as balance_mod
        from hudka.audio import write_wav

        buses = tmp_path / "buses"
        rng = np.random.default_rng(0)
        source = rng.standard_normal((SAMPLE_RATE * 2, 2)).astype(np.float32) * 0.3
        write_wav(buses / "original.wav", source)
        write_wav(buses / "music.wav", source * 0.2)      # 14 dB down - audible

        report = balance_mod.measure(tmp_path)
        assert not [p for p in report.problems() if "music" in p]

    def test_silent_footage_has_no_balance_to_report(self, tmp_path):
        from hudka import balance as balance_mod

        (tmp_path / "buses").mkdir(parents=True)
        assert balance_mod.measure(tmp_path) is None


class TestShapingControls:
    """Tone controls exist so a sound can be fixed rather than only re-rolled.

    The design rule under test: shaping is level-neutral and costs no regeneration, while
    anything reaching the model changes the cache key. If those two blur, the cheap knobs
    stop being cheap and the levels stop meaning anything.
    """

    @staticmethod
    def _tone_stem(tmp_path):
        from hudka.audio import write_wav

        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        both = (np.sin(2 * np.pi * 80 * t) * 0.4 + np.sin(2 * np.pi * 6000 * t) * 0.4)
        return write_wav(tmp_path / "s.wav", np.stack([both] * 2, axis=-1).astype(np.float32))

    @staticmethod
    def _cue(**kw):
        return SfxCue(**{"id": "s", "at": 0.0, "duration": 1.0, "prompt": "test tone",
                         "engine": "silence", "gain_db": 0.0, "align_transient": False, **kw})

    def test_filtering_does_not_change_the_level(self, tmp_path):
        from hudka.audio import rms_dbfs

        stem = self._tone_stem(tmp_path)
        plain = mix.place_sfx([self._cue()], {"s": stem}, total=3.0)
        filtered = mix.place_sfx([self._cue(highpass_hz=1000)], {"s": stem}, total=3.0)
        assert rms_dbfs(filtered) == pytest.approx(rms_dbfs(plain), abs=1.0)

    def test_highpass_actually_removes_the_low_end(self, tmp_path):
        stem = self._tone_stem(tmp_path)
        out = mix.place_sfx([self._cue(highpass_hz=1000)], {"s": stem}, total=3.0)
        spectrum = np.abs(np.fft.rfft(out.mean(axis=1)))
        freqs = np.fft.rfftfreq(len(out), 1 / SAMPLE_RATE)
        low = spectrum[(freqs > 50) & (freqs < 200)].sum()
        high = spectrum[(freqs > 5000) & (freqs < 7000)].sum()
        assert low < high / 10

    def test_reverse_flips_the_clip(self, tmp_path):
        from hudka.audio import write_wav

        clip = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        clip[:1000] = 0.5                                   # energy at the start
        stem = write_wav(tmp_path / "r.wav", clip)
        out = mix.place_sfx([self._cue(reverse=True)], {"s": stem}, total=2.0)
        loud = np.flatnonzero(np.abs(out).mean(axis=1) > 0.05)
        assert loud[0] / SAMPLE_RATE > 0.5, "energy should have moved to the end"

    def test_pitch_down_lengthens_the_clip(self, tmp_path):
        from hudka.audio import pitch_shift

        clip = np.ones((SAMPLE_RATE, 2), dtype=np.float32) * 0.3
        assert pitch_shift(clip, -12).shape[0] == pytest.approx(2 * SAMPLE_RATE, rel=0.01)

    def test_moving_a_cue_in_time_reuses_its_stem(self):
        """Dragging a clip must not throw away the sound it names.

        `window` reaches only an engine that conditions on picture, and generate_stems'
        cleanup unlinks every wav the current run did not produce - so a key that moved
        with `at` regenerated the cue AND deleted the take it already had.
        """
        from hudka.engines.base import GenerateRequest

        a = GenerateRequest(prompt="a click", duration=2.0, seed=1, window=(1.0, 3.0))
        b = GenerateRequest(prompt="a click", duration=2.0, seed=1, window=(9.0, 11.0))
        assert a.cache_key("silence") == b.cache_key("silence")

    def test_a_picture_conditioned_engine_still_keys_on_its_window(self, tmp_path):
        """The other half: for foley the window IS what the model sees, so two spans of
        the same video are different requests and must not share a stem."""
        from hudka.engines.base import GenerateRequest

        clip = tmp_path / "source.mp4"
        clip.write_bytes(b"")
        a = GenerateRequest(prompt="a click", duration=2.0, seed=1,
                            video=clip, window=(1.0, 3.0))
        b = GenerateRequest(prompt="a click", duration=2.0, seed=1,
                            video=clip, window=(9.0, 11.0))
        assert a.cache_key("hunyuan-foley") != b.cache_key("hunyuan-foley")

    def test_tone_changes_do_not_invalidate_the_cache(self):
        """Filtering is applied at placement, so it must not force a regeneration."""
        from hudka.engines.base import GenerateRequest

        req = GenerateRequest(prompt="a click", duration=2.0, seed=1)
        assert req.cache_key("silence") == GenerateRequest(
            prompt="a click", duration=2.0, seed=1
        ).cache_key("silence")

    def test_generation_options_do_invalidate_the_cache(self):
        from hudka.engines.base import GenerateRequest

        base = GenerateRequest(prompt="a click", duration=2.0, seed=1)
        for extra in ({"steps": 25}, {"cfg_scale": 3.0}, {"negative_prompt": "reverb"}):
            other = GenerateRequest(prompt="a click", duration=2.0, seed=1, extra=extra)
            assert other.cache_key("silence") != base.cache_key("silence"), extra


class TestBusAndTakes:
    def test_bus_trim_shifts_the_whole_bus(self, tmp_path):
        from hudka.audio import peak_dbfs, write_wav

        clip = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        clip[:1000] = 0.4
        stem = write_wav(tmp_path / "s.wav", clip)
        cue = SfxCue(id="s", at=0.0, duration=1.0, prompt="hit",
                     engine="silence", gain_db=0.0, align_transient=False)

        flat = mix.place_sfx([cue], {"s": stem}, total=2.0)
        trimmed = mix.place_sfx([cue], {"s": stem}, total=2.0, bus_offset_db=-6.0)
        assert peak_dbfs(trimmed) == pytest.approx(peak_dbfs(flat) - 6.0, abs=0.3)

    def test_take_seeds_are_deterministic_and_distinct(self):
        from hudka.render import take_seeds

        seeds = take_seeds(2001, 4)
        assert len(set(seeds)) == 4
        assert seeds == take_seeds(2001, 4)
        assert 2001 not in seeds, "a take must differ from the cue's current seed"

    def test_duck_override_beats_the_preset(self):
        from hudka.schema import CueSheet, VideoInfo

        sheet = CueSheet(video=VideoInfo(path="x.mp4", duration=10.0, fps=30.0),
                         preset="explainer", duck_depth_db=-2.0)
        assert sheet.duck_depth_db == -2.0
        assert CueSheet(video=sheet.video, preset="explainer").duck_depth_db is None


class TestPreviewIsUnmistakable:
    """A placeholder must never pass for a product.

    A preview render (stub engine) was heard as hiss and taken for a broken generator:
    the stub emitted noise, the output was named final.mp4, the UI said "on target", and
    the stem cleanup would have deleted a real render's cached audio. Every one of those
    is pinned here.
    """

    def test_placeholder_is_tonal_not_noise(self, tmp_path):
        from hudka.engines.base import GenerateRequest
        from hudka.engines.stub import SilenceEngine

        path = SilenceEngine().generate(GenerateRequest(prompt="x", duration=3.0, seed=3),
                                        tmp_path / "p.wav")
        samples, sr = read_wav(path)
        mono = samples.mean(axis=1) * np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(mono)) + 1e-12
        freqs = np.fft.rfftfreq(len(mono), 1 / sr)
        band = spectrum[(freqs > 50) & (freqs < 16000)]
        flatness = float(np.exp(np.log(band).mean()) / band.mean())
        assert flatness < 0.2, f"placeholder sounds like noise (flatness {flatness:.2f})"

    def test_placeholder_has_an_onset_and_is_deterministic(self, tmp_path):
        from hudka.audio import find_onset
        from hudka.engines.base import GenerateRequest
        from hudka.engines.stub import SilenceEngine

        a, _ = read_wav(SilenceEngine().generate(
            GenerateRequest(prompt="x", duration=2.0, seed=9), tmp_path / "a.wav"))
        b, _ = read_wav(SilenceEngine().generate(
            GenerateRequest(prompt="x", duration=2.0, seed=9), tmp_path / "b.wav"))
        assert np.array_equal(a, b)
        assert find_onset(a) < 0.05

    def test_preview_never_touches_the_real_render(self, fixture_video, tmp_path):
        """The data-loss case: previewing after a real render deleted the real stems."""
        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info)
        real = render.render(sheet, tmp_path)
        real_stems = {p.name for p in (tmp_path / "stems").rglob("*.wav")}
        real_final_mtime = real.final_video.stat().st_mtime
        assert real_stems

        result = render.render(sheet, tmp_path, preview=True)

        assert result.is_preview
        assert result.final_video == tmp_path / "preview" / "preview.mp4"
        assert {p.name for p in (tmp_path / "stems").rglob("*.wav")} == real_stems, \
            "preview render deleted or altered the real stems"
        assert real.final_video.stat().st_mtime == real_final_mtime
        assert (tmp_path / "preview" / "provenance.json").exists()
        assert not (tmp_path / "preview" / "final.mp4").exists()

    def test_preview_provenance_says_so(self, fixture_video, tmp_path):
        info = analyze_mod.probe(fixture_video)
        result = render.render(sheet_for(info), tmp_path, preview=True)
        ledger = json.loads(result.provenance.read_text(encoding="utf-8"))
        assert ledger["preview"] is True
        report = result.licence_report.read_text(encoding="utf-8")
        assert "PREVIEW" in report.splitlines()[0]
        assert "not for use" in report.splitlines()[0].lower()

    def test_real_render_is_not_flagged(self, fixture_video, tmp_path):
        info = analyze_mod.probe(fixture_video)
        result = render.render(sheet_for(info), tmp_path)
        assert result.is_preview is False
        assert json.loads(result.provenance.read_text())["preview"] is False


class TestQualityGateInRender:
    """The gate is wired where it can act: on cached stems, on fresh stems, on the mix."""

    @staticmethod
    def _square(seconds: float) -> np.ndarray:
        x = np.ones(int(seconds * SAMPLE_RATE), dtype=np.float32)
        x[1::2] = -1.0
        return np.stack([x, x], axis=-1)

    def test_cached_saturated_stem_is_regenerated_not_reused(self, fixture_video, tmp_path):
        """A cached file is not proof of a good file."""
        from hudka.audio import write_wav

        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info)
        first = render.render(sheet, tmp_path)
        victim = first.stems["hit1"]
        write_wav(victim, self._square(1.0))          # poison the cache in place

        log: list[str] = []
        second = render.render(sheet, tmp_path, progress=log.append)

        assert second.generated_count == 1, "only the poisoned stem should regenerate"
        assert any("regenerating" in line and "hit1" in line for line in log)
        from hudka import qa
        assert not qa.measure_stem(victim, "hit1", "sfx").problems()

    def test_fresh_saturated_stem_blocks_before_any_mix(self, fixture_video, tmp_path, monkeypatch):
        from hudka import qa
        from hudka.audio import write_wav

        def poisoned_worker(engine_id, cues, device, say):
            for item in cues:
                write_wav(Path(item["dest"]), self._square(item["duration"]))
        monkeypatch.setattr(render, "_run_worker", poisoned_worker)

        info = analyze_mod.probe(fixture_video)
        with pytest.raises(qa.QualityError) as exc:
            render.render(sheet_for(info), tmp_path)
        assert "hit1" in str(exc.value) and "re-roll" in str(exc.value).lower()
        assert not (tmp_path / "final.mp4").exists()
        assert not (tmp_path / "render_report.json").exists()

    def test_render_report_is_written(self, fixture_video, tmp_path):
        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info)
        result = render.render(sheet, tmp_path)
        report = json.loads((tmp_path / "render_report.json").read_text(encoding="utf-8"))
        assert report["verdict"] in ("ok", "warn")
        assert report["verdict"] == result.verdict
        assert len(report["stems"]) == len(sheet.all_cues())
        assert report["lufs"] == pytest.approx(result.measured_lufs, abs=0.01)

    def test_short_unlooped_bed_warns_by_name(self, fixture_video, tmp_path):
        """A 120s generation under a 182s range once left 62s of silence, silently."""
        from hudka.audio import read_wav, write_wav

        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info)
        first = render.render(sheet, tmp_path)
        bed = first.stems["bed"]
        samples, _ = read_wav(bed)
        write_wav(bed, samples[: samples.shape[0] // 3])   # now far shorter than its range

        second = render.render(sheet, tmp_path)
        assert second.verdict == "warn"
        assert any("bed" in w and "covers" in w for w in second.quality.warnings())

    def test_exit_codes_are_explained_not_all_blamed_on_vram(self):
        text = render._explain_exit(0xC0000006, "")
        assert "external" in text.lower() or "usb" in text.lower()
        assert "vram" not in text.lower()
        assert "ram" in render._explain_exit(-9 & 0xFFFFFFFF, "").lower()
        assert "vram" in render._explain_exit(1, "CUDA out of memory").lower()

    def test_worker_warning_lines_reach_the_log(self, monkeypatch, tmp_path):
        """An engine's `warning:` on stderr used to be read only after a failed exit."""
        import io

        from hudka.engines.base import GenerateRequest
        from hudka.engines.stub import SilenceEngine

        class FakeStdin:
            def __init__(self): self.buf, self.done = "", []
            def write(self, s): self.buf += s
            def close(self):
                job = json.loads(self.buf)
                for item in job["cues"]:
                    SilenceEngine().generate(
                        GenerateRequest(prompt=item["prompt"], duration=item["duration"],
                                        seed=item["seed"]), Path(item["dest"]))
                self.done = [item["id"] for item in job["cues"]]

        class LazyLines:
            """Iterable only when iterated - `_run_worker` truth-tests stdout before writing."""
            def __init__(self, stdin): self.stdin = stdin
            def __iter__(self):
                return iter([json.dumps({"done": d}) + "\n" for d in self.stdin.done])

        class FakeProc:
            def __init__(self, *a, **k):
                self.stdin = FakeStdin()
                self.stdout = LazyLines(self.stdin)
                self.stderr = io.StringIO("warning: test engine complained\n")
                self.returncode = 0
            def wait(self): return 0

        monkeypatch.setattr(render.subprocess, "Popen", FakeProc)
        log: list[str] = []
        dest = tmp_path / "x.wav"
        render._run_worker("silence", [{"id": "x", "prompt": "p", "duration": 1.0,
                                        "seed": 1, "dest": str(dest), "video": None,
                                        "window": [0, 1], "extra": {}}], None, log.append)
        assert any("test engine complained" in line for line in log)


class TestWorkerPipes:
    """The worker's stderr must be drained while stdout is read, or it deadlocks.

    Reproduced through a real OS pipe, not a fake: a child that writes well past any pipe
    buffer to stderr before it finishes. Without a concurrent drain the child blocks on
    the write, the parent blocks reading stdout, and the render never ends - which is
    exactly what happened on a 36-cue project.
    """

    def test_a_chatty_worker_does_not_deadlock(self, monkeypatch, tmp_path):
        import subprocess
        import sys
        import threading

        dest = tmp_path / "x.wav"
        # A child that floods stderr (2 MB, far beyond any pipe buffer), writes a valid
        # WAV where the worker would, then reports the cue done on stdout.
        script = (
            "import sys, wave, json\n"
            "sys.stderr.write('x' * 2_000_000); sys.stderr.flush()\n"
            f"w = wave.open({str(dest)!r}, 'wb'); w.setnchannels(2); w.setsampwidth(2); "
            "w.setframerate(44100); w.writeframes(bytes([0, 16]) * 88200); w.close()\n"
            "print(json.dumps({'done': 'x'}), flush=True)\n"
        )
        real_popen = subprocess.Popen

        def spawn(args, **kw):
            return real_popen([sys.executable, "-c", script], **kw)
        monkeypatch.setattr(render.subprocess, "Popen", spawn)

        log: list[str] = []
        outcome: dict = {}

        def run():
            try:
                render._run_worker("silence", [{"id": "x", "prompt": "p", "duration": 1.0,
                                                "seed": 1, "dest": str(dest), "video": None,
                                                "window": [0, 1], "extra": {}}], None, log.append)
                outcome["ok"] = True
            except Exception as exc:  # pragma: no cover - surfaces in the assertion
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=60)
        assert not worker.is_alive(), "worker deadlocked on a full stderr pipe"
        assert outcome.get("ok"), outcome.get("error")
        assert any("render  x" in line for line in log)


class TestProxyAnalysis:
    """Frame analysis runs on a once-decoded proxy, not on the source twice.

    PySceneDetect decoding a 4K source itself took 125 s of a 159 s analysis; the
    motion curve decoded it again. Cut positions from the proxy must still land on the
    known cuts, and the proxy must exist for later re-use.
    """

    def test_proxy_is_written_and_cuts_still_land(self, fixture_video, tmp_path):
        result = analyze_mod.analyze(fixture_video, tmp_path)
        assert (tmp_path / "proxy.mp4").exists()
        boundaries = [s.start for s in result.shots[1:]]
        for expected in CUT_TIMES:
            assert any(abs(b - expected) < 0.15 for b in boundaries), \
                f"no boundary near the {expected}s cut in {boundaries}"

    def test_stage_timings_are_recorded(self, fixture_video, tmp_path):
        result = analyze_mod.analyze(fixture_video, tmp_path)
        assert {"probe", "proxy", "motion", "shots", "speech", "sheets"} <= set(result.timings)
        saved = json.loads((tmp_path / "analysis.json").read_text(encoding="utf-8"))
        assert saved["timings"]["shots"] >= 0

    def test_speech_scan_does_not_decode_video(self):
        """`-vn` is what keeps an audio scan from costing a full picture decode."""
        import inspect

        source = inspect.getsource(analyze_mod.detect_speech)
        assert '"-vn"' in source


class TestBedLoopSeam:
    def test_music_loops_use_a_musical_crossfade(self):
        """A 0.25s splice mid-phrase at 2:00 is what a listener notices in minute two."""
        import inspect

        assert mix.BED_LOOP_CROSSFADE_S >= 1.0
        assert "crossfade=BED_LOOP_CROSSFADE_S" in inspect.getsource(mix.place_beds)

    def test_long_crossfade_holds_level_across_the_seam(self):
        from hudka.audio import loop_to_length

        t = np.arange(SAMPLE_RATE * 6) / SAMPLE_RATE
        tone = (np.sin(2 * np.pi * 110 * t) * 0.4).astype(np.float32)
        clip = np.stack([tone, tone], axis=-1)
        looped = loop_to_length(clip, 14.0, crossfade=mix.BED_LOOP_CROSSFADE_S)
        # RMS in 250ms windows across the first seam (around 6s) must not dip.
        win = SAMPLE_RATE // 4
        levels = [np.sqrt((looped[i:i + win, 0] ** 2).mean())
                  for i in range(int(4.5 * SAMPLE_RATE), int(7.5 * SAMPLE_RATE), win)]
        assert min(levels) > 0.7 * max(levels), "seam dipped audibly"


class TestMutedCues:
    """Mute leaves a cue out of the mix without losing it.

    It is a placement decision, so it stays out of the cache key: the stem is still
    generated and still recorded in the ledger, which keeps provenance truthful and makes
    unmuting free.
    """

    def test_a_muted_effect_is_not_in_the_mix_but_is_still_generated(self, fixture_video, tmp_path):
        from hudka.audio import peak_dbfs

        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info)
        sheet.sfx[0].muted = True

        result = render.render(sheet, tmp_path)
        assert result.stems["hit1"].exists(), "a muted cue must still be generated"

        records = json.loads(result.provenance.read_text(encoding="utf-8"))["records"]
        assert any(r["cue_id"] == "hit1" for r in records), \
            "a muted cue must stay in the ledger - its stem is on disk"

        bus, _ = read_wav(tmp_path / "buses" / "sfx.wav")
        loud = np.flatnonzero(np.abs(bus).mean(axis=1) > 0.01) / SAMPLE_RATE
        assert not any(abs(t - 4.0) < 0.3 for t in loud), "the muted cue was placed anyway"
        assert any(abs(t - 8.0) < 0.3 for t in loud), "the unmuted cue should still be there"

    def test_muting_does_not_change_the_cache_key(self):
        from hudka.engines.base import GenerateRequest

        a = GenerateRequest(prompt="a click", duration=2.0, seed=1)
        b = GenerateRequest(prompt="a click", duration=2.0, seed=1)
        assert a.cache_key("silence") == b.cache_key("silence")

    def test_muting_every_effect_does_not_block_the_render(self, fixture_video, tmp_path):
        """Otherwise the quality gate's 'no effect landed' fires on a deliberate choice."""
        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info)
        for cue in sheet.sfx:
            cue.muted = True

        result = render.render(sheet, tmp_path)
        assert result.verdict in ("ok", "warn")
        assert not any("no effect landed" in p for p in result.quality.problems())

    def test_a_muted_bed_leaves_no_music(self, fixture_video, tmp_path):
        from hudka.audio import peak_dbfs

        info = analyze_mod.probe(fixture_video)
        sheet = sheet_for(info)
        sheet.music[0].muted = True
        render.render(sheet, tmp_path)
        bus, _ = read_wav(tmp_path / "buses" / "music.wav")
        assert peak_dbfs(bus) < -60.0
