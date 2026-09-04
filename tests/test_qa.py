"""Content quality gates.

The thresholds are pinned here against synthesised signals, because a flatness number
means nothing without the estimator that produced it (amplitude- vs power-spectrum
estimators differ by ~1.5x on the same noise). If the estimator changes, these fail and
the constants get re-measured on purpose rather than drifting.
"""

from __future__ import annotations

import numpy as np
import pytest

from hudka import qa
from hudka.audio import SAMPLE_RATE, write_wav


def sine(seconds=2.0, hz=440.0, amp=0.5):
    """A clean tone with a decaying envelope - crest ~11 dB, like real generated audio.

    Deliberately not a steady sine: that has a crest factor of 3.01 dB, which the gate
    correctly reads as saturated. Nothing these models emit is a steady sine, so the
    clean reference here has dynamics, as every real stem measured so far does.
    """
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    env = np.exp(-3.0 * t / max(seconds, 1e-6))
    x = (np.sin(2 * np.pi * hz * t) * env * amp).astype(np.float32)
    return np.stack([x, x], axis=-1)


def noise(seconds=2.0, amp=0.3, seed=0):
    x = (np.random.default_rng(seed).standard_normal(int(seconds * SAMPLE_RATE)) * amp).astype(np.float32)
    x = np.clip(x, -0.99, 0.99)
    return np.stack([x, x], axis=-1)


def square(seconds=1.0):
    x = np.ones(int(seconds * SAMPLE_RATE), dtype=np.float32)
    x[1::2] = -1.0
    return np.stack([x, x], axis=-1)


class TestEstimators:
    def test_flatness_separates_noise_from_tone_by_a_wide_margin(self):
        tone = qa.spectral_flatness(sine())
        hiss = qa.spectral_flatness(noise())
        assert tone < 0.05
        assert hiss > 0.45
        # The threshold must sit clearly between the two, not near either.
        assert tone < qa.NOISE_FLATNESS < hiss
        assert hiss - qa.NOISE_FLATNESS > 0.1 and qa.NOISE_FLATNESS - tone > 0.2

    def test_flatness_ignores_silence_around_a_hit(self):
        """A one-shot is mostly silence; the estimate must come from the sound, not the gap."""
        clip = np.zeros((SAMPLE_RATE * 2, 2), dtype=np.float32)
        clip[:4000] = sine(4000 / SAMPLE_RATE)[:4000]
        assert qa.spectral_flatness(clip) < 0.05

    def test_crest_separates_real_from_saturated(self):
        assert qa.crest_db(sine()) > 2.5          # a sine is 3.01 dB
        assert qa.crest_db(square()) < 0.5
        hit = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
        hit[:500] = 0.6
        assert qa.crest_db(hit) > qa.SATURATION_CREST_DB

    def test_placeholder_engine_is_not_noise_like(self, tmp_path):
        """The preview stub was rewritten to be tonal; a preview must not warn as noise."""
        from hudka.engines.base import GenerateRequest
        from hudka.engines.stub import SilenceEngine

        path = SilenceEngine().generate(GenerateRequest(prompt="x", duration=3.0, seed=1),
                                        tmp_path / "p.wav")
        assert qa.measure_stem(path, "p", "music").flatness < qa.NOISE_FLATNESS


class TestStemGate:
    def test_silence_blocks(self, tmp_path):
        q = qa.measure_stem(write_wav(tmp_path / "s.wav", np.zeros((SAMPLE_RATE, 2), np.float32)), "s", "sfx")
        assert any("silent" in p for p in q.problems())

    def test_saturation_blocks(self, tmp_path):
        q = qa.measure_stem(write_wav(tmp_path / "s.wav", square()), "s", "sfx")
        assert any("saturated" in p for p in q.problems())

    def test_heavy_clipping_blocks_light_clipping_warns(self, tmp_path):
        heavy = np.clip(sine(amp=1.0) * 4.0, -1.0, 1.0)         # flat-topped for most of it
        q = qa.measure_stem(write_wav(tmp_path / "h.wav", heavy), "h", "sfx")
        assert any("clipped" in p for p in q.problems())

        light = sine(amp=0.5)
        light[:60] = 1.0                                       # ~0.03% of samples
        q = qa.measure_stem(write_wav(tmp_path / "l.wav", light), "l", "sfx")
        assert not q.problems()
        assert q.clipped_fraction < qa.CLIP_BLOCK_FRACTION

    def test_a_few_full_scale_samples_in_a_bed_do_not_block(self, tmp_path):
        """The library clamps to +-1; a real 48s bed carries ~20 such samples."""
        bed = sine(seconds=20.0, amp=0.5)
        bed[1000:1020] = 1.0
        q = qa.measure_stem(write_wav(tmp_path / "b.wav", bed), "bed", "music")
        assert not q.problems()

    def test_noise_warns_but_does_not_block(self, tmp_path):
        """Wind, rain and air are legitimately noise-like."""
        q = qa.measure_stem(write_wav(tmp_path / "n.wav", noise()), "n", "sfx")
        assert not q.problems()
        assert any("noise" in w for w in q.warnings())

    def test_dc_offset_warns(self, tmp_path):
        q = qa.measure_stem(write_wav(tmp_path / "d.wav", sine() + 0.05), "d", "sfx")
        assert any("DC offset" in w for w in q.warnings())

    def test_short_bed_warns_and_names_the_shortfall(self, tmp_path):
        q = qa.measure_stem(write_wav(tmp_path / "b.wav", sine(seconds=10.0)), "bed", "music",
                            wanted_s=30.0)
        assert any("covers 10s of its 30s" in w for w in q.warnings())

    def test_a_clean_stem_is_clean(self, tmp_path):
        q = qa.measure_stem(write_wav(tmp_path / "c.wav", sine()), "c", "sfx")
        assert not q.problems() and not q.warnings()


class TestRenderVerdict:
    def _stem(self, tmp_path, name, audio, kind="sfx"):
        return qa.measure_stem(write_wav(tmp_path / f"{name}.wav", audio), name, kind)

    def test_ok_when_everything_is_clean(self, tmp_path):
        r = qa.RenderQuality(stems=[self._stem(tmp_path, "a", sine())], mix_lufs=-14.0,
                             mix_flatness=0.02, sfx_events=1, sfx_cues=1)
        assert r.verdict == "ok"

    def test_warn_on_noise_like_stem(self, tmp_path):
        r = qa.RenderQuality(stems=[self._stem(tmp_path, "n", noise())], mix_lufs=-14.0,
                             mix_flatness=0.02, sfx_events=1, sfx_cues=1)
        assert r.verdict == "warn"

    def test_fail_on_saturated_stem(self, tmp_path):
        r = qa.RenderQuality(stems=[self._stem(tmp_path, "s", square())], mix_lufs=-14.0,
                             mix_flatness=0.02, sfx_events=1, sfx_cues=1)
        assert r.verdict == "fail"

    def test_unmeasurable_mix_fails(self):
        assert qa.RenderQuality(mix_lufs=float("nan")).verdict == "fail"

    def test_no_effect_landing_fails_when_effects_were_asked_for(self):
        r = qa.RenderQuality(mix_lufs=-14.0, sfx_events=0, sfx_cues=3)
        assert any("no effect landed" in p for p in r.problems())
        assert qa.RenderQuality(mix_lufs=-14.0, sfx_events=0, sfx_cues=0).verdict == "ok"

    def test_all_noise_escalates_the_message(self, tmp_path):
        r = qa.RenderQuality(stems=[self._stem(tmp_path, "a", noise(seed=1)),
                                    self._stem(tmp_path, "b", noise(seed=2))],
                             mix_lufs=-14.0, mix_flatness=0.5)
        assert any("every generated sound is noise-like" in w for w in r.warnings())

    def test_quality_error_names_cues_and_the_remedy(self):
        err = qa.QualityError(["hit01: saturated"], ["hit01"])
        assert "hit01" in str(err) and "re-roll" in str(err).lower()
