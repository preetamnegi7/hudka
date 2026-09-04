"""Audio helpers — the parts that decide whether a sound lands where it should."""

from __future__ import annotations

import numpy as np

from hudka.audio import (
    SAMPLE_RATE,
    db_to_gain,
    find_onset,
    fit_length,
    loop_to_length,
    to_stereo,
)


def burst(lead_in: float, length: float = 1.0) -> np.ndarray:
    """A clip with `lead_in` seconds of silence before a loud decaying hit."""
    silence = np.zeros(int(lead_in * SAMPLE_RATE), dtype=np.float32)
    n = int((length - lead_in) * SAMPLE_RATE)
    hit = np.exp(-np.linspace(0, 8, n, dtype=np.float32))
    return np.stack([np.concatenate([silence, hit])] * 2, axis=-1)


class TestOnset:
    def test_finds_lead_in_before_a_transient(self):
        """This offset is what stops a generated hit arriving late against a cut."""
        assert find_onset(burst(0.15)) == np.float32(0.15).item() or abs(
            find_onset(burst(0.15)) - 0.15
        ) < 0.01

    def test_reports_zero_when_the_hit_is_immediate(self):
        assert find_onset(burst(0.0)) < 0.005

    def test_ignores_slow_swells(self):
        """A late peak means a bed or swell, not a hit — shifting those would be wrong."""
        n = SAMPLE_RATE * 3
        swell = np.linspace(0, 1, n, dtype=np.float32) ** 2
        assert find_onset(np.stack([swell] * 2, axis=-1)) == 0.0

    def test_handles_silence(self):
        assert find_onset(np.zeros((SAMPLE_RATE, 2), dtype=np.float32)) == 0.0


class TestShaping:
    def test_to_stereo_expands_mono(self):
        out = to_stereo(np.ones(100, dtype=np.float32))
        assert out.shape == (100, 2)

    def test_to_stereo_transposes_channels_first(self):
        """stable-audio-tools returns (channels, samples); soundfile wants the reverse."""
        assert to_stereo(np.ones((2, 500), dtype=np.float32)).shape == (500, 2)

    def test_fit_length_pads_and_trims(self):
        short = np.ones((SAMPLE_RATE, 2), dtype=np.float32)
        assert fit_length(short, 2.0).shape[0] == 2 * SAMPLE_RATE
        assert fit_length(short, 0.5).shape[0] == SAMPLE_RATE // 2

    def test_db_to_gain(self):
        assert abs(db_to_gain(0.0) - 1.0) < 1e-6
        assert abs(db_to_gain(-6.0) - 0.5012) < 1e-3


class TestLooping:
    def test_fills_the_requested_length(self):
        clip = np.ones((SAMPLE_RATE, 2), dtype=np.float32) * 0.5
        assert loop_to_length(clip, 3.5).shape[0] == int(3.5 * SAMPLE_RATE)

    def test_trims_when_already_long_enough(self):
        clip = np.ones((4 * SAMPLE_RATE, 2), dtype=np.float32)
        assert loop_to_length(clip, 2.0).shape[0] == 2 * SAMPLE_RATE

    def test_crossfade_does_not_produce_a_gap(self):
        """An equal-power seam should hold level; a linear one would dip audibly."""
        clip = np.ones((SAMPLE_RATE, 2), dtype=np.float32) * 0.5
        looped = loop_to_length(clip, 2.5)
        envelope = np.abs(looped).mean(axis=1)
        # Ignore the very start and end; the seam sits in the middle.
        interior = envelope[1000:-1000]
        assert interior.min() > 0.4, "crossfade seam dipped"
