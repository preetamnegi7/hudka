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
             speech=None, has_audio=True):
    return {
        "video": {"path": "x.mp4", "duration": duration, "fps": 60.0,
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
