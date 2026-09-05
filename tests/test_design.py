"""The scaffold designer.

The case that matters here is the one that shipped broken: footage with no hard cuts and
very low motion - a screen recording, a locked-off talking head. A fixed motion threshold
tuned for edited video matches nothing on that material and yields a cue sheet with no
effects at all.
"""

from __future__ import annotations

import pytest

from hudka import design
from hudka.schema import VideoInfo


def analysis(*, duration=48.0, width=3840, height=2160, shots=None, curve=None,
             speech=None, has_audio=True, path="x.mp4"):
    return {
        "video": {"path": path, "duration": duration, "fps": 60.0,
                  "width": width, "height": height, "has_audio": has_audio,
                  "has_dialogue": bool(speech)},
        "shots": shots if shots is not None else [
            {"index": 0, "start": 0.0, "end": duration, "motion": 0.0005,
             "peak_motion": 0.036, "peak_at": 27.9}],
        "motion_curve": curve or [],
        "speech_ranges": speech or [],
    }


def flat_curve(duration, base, spikes):
    """A low-motion curve with a few spikes, like a screen recording."""
    curve = [[round(t * 0.125, 3), base] for t in range(int(duration * 8))]
    for at, value in spikes:
        idx = min(int(at * 8), len(curve) - 1)
        curve[idx][1] = value
    return curve


class TestPresetChoice:
    def test_portrait_is_always_short_form(self):
        info = VideoInfo(path="x", duration=30, fps=30, width=1080, height=1920)
        assert design.suggest_preset(info, analysis()) == "short-form"

    def test_narration_heavy_landscape_becomes_explainer(self):
        """84% speech means the music must sit under a voice, not carry the edit."""
        a = analysis(speech=[(0, 40.6)])
        info = VideoInfo.model_validate(a["video"])
        assert design.suggest_preset(info, a) == "explainer"

    def test_long_quiet_landscape_becomes_cinematic(self):
        a = analysis(duration=200, speech=[])
        info = VideoInfo.model_validate(a["video"])
        assert design.suggest_preset(info, a) == "cinematic"

    def test_short_quiet_landscape_stays_short_form(self):
        a = analysis(duration=30, speech=[])
        info = VideoInfo.model_validate(a["video"])
        assert design.suggest_preset(info, a) == "short-form"


class TestLowMotionFootage:
    """The regression: a screen recording produced zero effects."""

    def test_screen_recording_still_gets_effects(self):
        a = analysis(
            curve=flat_curve(48, 0.0005, [(11.5, 0.011), (25.6, 0.012),
                                          (27.9, 0.036), (38.9, 0.036)]),
            speech=[(0, 40.6)],
        )
        sheet = design.scaffold(a)
        assert sheet.sfx, "low-motion footage produced no cues at all"
        assert sheet.preset == "explainer"

    def test_anchors_land_on_the_activity_peaks(self):
        peaks = [11.5, 25.6, 27.9, 38.9]
        a = analysis(curve=flat_curve(48, 0.0005, [(p, 0.03) for p in peaks]))
        sheet = design.scaffold(a)
        for cue in sheet.sfx:
            assert any(abs(cue.at - p) < 1.0 for p in peaks), \
                f"cue at {cue.at}s does not match any activity peak"

    def test_cues_are_spaced_apart(self):
        """Anchors closer than the minimum gap turn the mix to mush."""
        clustered = [(10.0 + i * 0.2, 0.03) for i in range(12)]
        a = analysis(curve=flat_curve(48, 0.0005, clustered))
        times = sorted(c.at for c in design.scaffold(a).sfx)
        for earlier, later in zip(times, times[1:]):
            assert later - earlier >= design.MIN_GAP_SECONDS - 0.01

    def test_pure_noise_produces_no_false_anchors(self):
        a = analysis(curve=flat_curve(48, 0.0001, []))
        assert design.scaffold(a).sfx == []

    def test_a_music_bed_is_always_present(self):
        sheet = design.scaffold(analysis(curve=flat_curve(48, 0.0001, [])))
        assert len(sheet.music) == 1
        assert sheet.music[0].end == pytest.approx(48.0, abs=0.1)


class TestEditedFootage:
    def test_cuts_become_transition_cues(self):
        shots = [{"index": i, "start": i * 4.0, "end": (i + 1) * 4.0,
                  "motion": 0.2, "peak_motion": 0.3, "peak_at": i * 4.0 + 2}
                 for i in range(4)]
        sheet = design.scaffold(analysis(duration=16, shots=shots), preset="short-form")
        cut_cues = [c for c in sheet.sfx if c.id.startswith("cut")]
        assert len(cut_cues) == 3, "one accent per cut, not counting the first frame"
        assert [c.at for c in cut_cues] == [4.0, 8.0, 12.0]

    def test_speech_thins_the_design_out(self):
        shots = [{"index": i, "start": i * 2.0, "end": (i + 1) * 2.0,
                  "motion": 0.2, "peak_motion": 0.3, "peak_at": i * 2.0 + 1}
                 for i in range(20)]
        curve = flat_curve(40, 0.05, [(i * 2.0 + 1, 0.4) for i in range(20)])
        quiet = design.scaffold(analysis(duration=40, shots=shots, curve=curve,
                                         speech=[]), preset="short-form")
        talky = design.scaffold(analysis(duration=40, shots=shots, curve=curve,
                                         speech=[(0, 36)]), preset="short-form")
        assert len(talky.sfx) <= len(quiet.sfx)
        motion_cues = [c for c in talky.sfx if c.id.startswith("hit")]
        assert all(c.gain_db <= -14.0 for c in motion_cues), \
            "cues over speech should be quieter"


class TestSpeechCoverage:
    def test_measures_the_talking_fraction(self):
        assert design.speech_coverage({"speech_ranges": [(0, 40.6)]}, 48.6) == \
            pytest.approx(0.835, abs=0.01)

    def test_silent_video_is_zero(self):
        assert design.speech_coverage({"speech_ranges": []}, 48.0) == 0.0

    def test_zero_duration_does_not_divide_by_zero(self):
        assert design.speech_coverage({"speech_ranges": [(0, 5)]}, 0) == 0.0


class TestScaffoldDefaults:
    def test_omitting_the_preset_lets_the_designer_choose(self):
        """A concrete default would override the auto-choice and never be noticed."""
        from hudka.ui.server import ScaffoldOptions

        assert ScaffoldOptions().preset is None

    def test_explicit_preset_is_respected(self):
        a = analysis(speech=[(0, 40.6)])  # would otherwise pick explainer
        assert design.scaffold(a, preset="cinematic").preset == "cinematic"


class TestGenerationLength:
    """Short generations come back saturated about half the time.

    Measured on small-sfx: 0.8s, 1.2s and 1.8s all returned audio pinned to +-1.0 with a
    crest factor near 1.5 dB, while 1.0s, 1.5s and everything from 2.0s up were clean. It
    is not a threshold, so cues are generated at a safe length and trimmed after.
    """

    def test_scaffolded_one_shots_clear_the_unreliable_range(self):
        from hudka.design import SFX_DURATION_SECONDS
        from hudka.engines.stable_audio3 import MIN_RELIABLE_SECONDS

        assert SFX_DURATION_SECONDS >= MIN_RELIABLE_SECONDS

    def test_short_cues_are_generated_at_a_safe_length(self):
        """A 0.5s cue must not be *generated* at 0.5s, whatever it is placed at."""
        from hudka.engines.stable_audio3 import MIN_RELIABLE_SECONDS

        requested = 0.5
        assert max(requested, MIN_RELIABLE_SECONDS) == MIN_RELIABLE_SECONDS

    def test_saturation_detector_separates_a_hit_from_a_square_wave(self):
        import numpy as np

        from hudka.engines.stable_audio3 import SATURATION_CREST_DB, _crest_db

        hit = np.zeros((2, 44100), dtype=np.float32)
        hit[:, :400] = 0.6
        saturated = np.ones((2, 44100), dtype=np.float32)
        saturated[:, 1::2] = -1.0

        assert _crest_db(hit) > SATURATION_CREST_DB
        assert _crest_db(saturated) < SATURATION_CREST_DB

    def test_silence_does_not_divide_by_zero(self):
        import numpy as np

        from hudka.engines.stable_audio3 import _crest_db

        assert _crest_db(np.zeros((2, 1000), dtype=np.float32)) == 0.0


class TestPromptVariety:
    def test_scaffolded_effects_are_not_all_identical(self):
        """Every effect used to carry one hardcoded string, so they all sounded the same."""
        curve = flat_curve(48, 0.0005, [(t, 0.02 + i * 0.004)
                                        for i, t in enumerate((6.0, 12.0, 18.0, 24.0, 30.0, 36.0))])
        sheet = design.scaffold(analysis(curve=curve, speech=[(0, 40.6)]))
        prompts = [c.prompt for c in sheet.sfx]
        assert len(set(prompts)) > 1, "all scaffolded effects share one prompt"

    def test_stronger_anchors_become_transitions(self):
        curve = flat_curve(48, 0.0005, [(10.0, 0.04), (20.0, 0.004), (30.0, 0.038)])
        sheet = design.scaffold(analysis(curve=curve))
        roles = [c.note.split(",")[0] for c in sheet.sfx]
        assert "transition" in roles

    def test_gains_come_from_the_preset_not_hardcoded(self):
        from hudka import presets

        curve = flat_curve(48, 0.0005, [(10.0, 0.04), (20.0, 0.03), (30.0, 0.02)])
        sheet = design.scaffold(analysis(curve=curve), preset="explainer")
        pre = presets.get("explainer")
        allowed = {pre.sfx_gain_db + trim for trim in presets.SFX_TRIM_DB.values()}
        for cue in sheet.sfx:
            assert cue.gain_db in allowed, f"{cue.id} gain {cue.gain_db} is not preset-derived"


class TestMusicVariety:
    """Every project used to get the same background music - not similar, the same.

    The bed prompt was one of two fixed strings chosen by a single boolean and the seed
    was hardcoded to 7. Prompt, seed and engine are the whole cache key, so two clips of
    the same length produced byte-identical music, and every clip whatever its length got
    the same key, tempo and instruments. Three real projects on this machine shared one
    prompt hash and one seed.
    """

    @staticmethod
    def _bed(**kwargs):
        return design.scaffold(analysis(**kwargs)).music[0]

    def test_two_projects_from_the_same_file_get_different_music(self):
        """The case that made it obvious: one video imported twice, same footage, same
        length - and Hudka wrote the same underscore into both."""
        a = self._bed(path=r"out\one\source\clip.mp4", speech=[(0, 40)])
        b = self._bed(path=r"out\two\source\clip.mp4", speech=[(0, 40)])
        assert (a.prompt, a.seed) != (b.prompt, b.seed)

    def test_the_same_project_always_gets_the_same_music(self):
        """Re-scaffolding must not silently replace the music. The literal is the point:
        Python's hash() is salted per process, so an implementation built on it would
        pass a same-process comparison and still change the bed on every restart."""
        first = self._bed(path="a.mp4", speech=[(0, 40)])
        second = self._bed(path="a.mp4", speech=[(0, 40)])
        assert (first.prompt, first.seed) == (second.prompt, second.seed)
        assert first.seed == 50113, "the seed must be a stable digest, not a salted hash"

    def test_the_bed_follows_how_busy_the_picture_is(self):
        still = analysis(curve=flat_curve(48, 0.0002, []))
        busy = analysis(
            curve=flat_curve(48, 0.03, []),
            shots=[{"index": i, "start": i * 4.0, "end": i * 4.0 + 4.0, "motion": 0.03,
                    "peak_motion": 0.05, "peak_at": i * 4.0 + 1} for i in range(12)])
        assert design.bed_pace(still, 48.0) == "slow"
        assert design.bed_pace(busy, 48.0) == "fast"
        assert self._bed(curve=flat_curve(48, 0.0002, [])).prompt != \
            design.scaffold(busy).music[0].prompt

    def test_a_narrated_clip_gets_a_bed_written_to_sit_under_the_voice(self):
        """A bed with a lead melody fights a voiceover instead of supporting it."""
        narrated = self._bed(speech=[(0.0, 44.0)])
        assert "no lead melody" in narrated.prompt
        assert "no vocals" in narrated.prompt

    def test_every_bed_keeps_the_voice_room_it_was_written_for(self):
        for bed in design.MUSIC_BEDS:
            assert "no vocals" in bed.prompt, f"{bed.mood} could come back with singing"
            # A darker, more mono bed competes with centre-panned speech instead of
            # sitting under it.
            assert "wide stereo image" in bed.prompt and "airy top end" in bed.prompt
            if bed.narration:
                assert "no lead melody" in bed.prompt and "steady dynamics" in bed.prompt

    def test_the_library_covers_every_combination_it_is_asked_for(self):
        pairs = {(b.narration, b.pace) for b in design.MUSIC_BEDS}
        for narration in (True, False):
            for pace in ("slow", "medium", "fast"):
                assert (narration, pace) in pairs, f"nothing for {narration}/{pace}"
                pool = [b for b in design.MUSIC_BEDS
                        if b.narration is narration and b.pace == pace]
                assert len(pool) > 1, "one candidate is how this became one bed for everyone"


class TestHardwareAdaptation:
    """The tool has to be usable without an NVIDIA GPU, not merely runnable.

    Measured: a 50-step 20s bed is 2.2s on a 4070 and 228s on the CPU. Shipping the GPU
    setting to a CPU user turns a 66-second render into something they abandon.
    """

    def test_cpu_caps_the_expensive_bed_steps(self, monkeypatch):
        from hudka.engines.stable_audio3 import DEFAULT_STEPS, StableAudio3Engine

        engine = StableAudio3Engine("stable-audio-3-small-music", device="cpu")
        assert engine.steps == 50, "the GPU setting should still be what is configured"
        assert engine._effective_steps() == DEFAULT_STEPS

    def test_gpu_keeps_the_richer_setting(self):
        from hudka.engines.stable_audio3 import BED_STEPS, StableAudio3Engine

        engine = StableAudio3Engine("stable-audio-3-small-music", device="cuda")
        assert engine._effective_steps() == BED_STEPS

    def test_one_shots_are_unaffected_either_way(self):
        from hudka.engines.stable_audio3 import DEFAULT_STEPS, StableAudio3Engine

        for device in ("cpu", "cuda"):
            engine = StableAudio3Engine("stable-audio-3-small-sfx", device=device)
            assert engine._effective_steps() == DEFAULT_STEPS


class TestPlacementRules:
    """Rules that stop the scaffold placing sounds a listener hears as mistakes."""

    @staticmethod
    def _shots(*spans):
        shots, t = [], 0.0
        for i, length in enumerate(spans):
            shots.append({"index": i, "start": t, "end": t + length, "motion": 0.2,
                          "peak_motion": 0.3, "peak_at": t + length / 2})
            t += length
        return shots, t

    def test_cuts_closer_than_the_gap_get_one_transition(self):
        """Two page loads a second apart used to get a whoosh each - a stutter."""
        shots, total = self._shots(8.0, 1.0, 8.0, 8.0)      # cuts at 8.0, 9.0, 17.0
        sheet = design.scaffold(analysis(duration=total, shots=shots), preset="short-form")
        cuts = sorted(c.at for c in sheet.sfx if c.id.startswith("cut"))
        assert 8.0 in cuts and 17.0 in cuts
        assert 9.0 not in cuts, "the second of two adjacent cuts must not get its own whoosh"

    def test_a_flash_frame_marks_one_transition_at_its_start(self):
        """A flash between two scenes is one event: the whoosh lands where the picture
        first changes, and the flash's exit a moment later gets nothing."""
        shots, total = self._shots(8.0, 0.9, 8.0)           # the 0.9s shot is a flash
        sheet = design.scaffold(analysis(duration=total, shots=shots), preset="short-form")
        cuts = sorted(c.at for c in sheet.sfx if c.id.startswith("cut"))
        assert cuts == [8.0]

    def test_heavy_narration_uses_the_low_end_of_density(self):
        """96% speech on a 3.6-minute clip produced 35 effects, mostly cursor jitter."""
        from hudka import presets

        minutes = 3.6
        spikes = [(t, 0.03) for t in range(2, int(minutes * 60) - 2, 3)]   # plenty of anchors
        curve = flat_curve(minutes * 60, 0.0005, spikes)
        talky = design.scaffold(analysis(duration=minutes * 60, curve=curve,
                                         speech=[(0, minutes * 60 * 0.96)]),
                                preset="explainer")
        low, high = presets.get("explainer").sfx_per_minute
        assert len(talky.sfx) <= round(low * minutes) + 1
        assert len(talky.sfx) < round(((low + high) / 2) * minutes * 0.8)
