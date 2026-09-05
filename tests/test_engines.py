"""Engine-layer hazards that no test used to cover."""

from __future__ import annotations

import importlib
import sys

import pytest

from hudka import engines
from hudka.engines.stable_audio3 import BED_STEPS, StableAudio3Engine
from hudka.schema import BedCue


class TestAceStepModule:
    def test_the_module_parses_and_the_engine_builds(self):
        """The file carried a raw newline inside a string literal at HEAD, so
        build("acestep-1.5") raised SyntaxError instead of the intended "not installed"."""
        module = importlib.import_module("hudka.engines.acestep")
        engine = engines.build("acestep-1.5")
        assert isinstance(engine, module.AceStepEngine)
        assert engine.id == "acestep-1.5"

    def test_missing_package_is_reported_not_crashed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "acestep", None)     # makes `import acestep` fail
        engine = engines.build("acestep-1.5")
        with pytest.raises(RuntimeError, match="not installed"):
            engine.preflight()


class TestSafeDefaults:
    def test_a_bed_defaults_to_the_engine_every_machine_can_run(self):
        """The default was the medium model - the one the code says kills a 12 GB card -
        so any hand-written or programmatic bed inherited it."""
        assert BedCue(id="b", start=0.0, end=10.0, prompt="a bed").engine == "stable-audio-3-small-music"

    def test_an_explicit_eight_steps_is_honoured(self):
        """`steps: int = 8` made an explicit 8 indistinguishable from unset, so it was
        silently promoted to the variant's 50."""
        assert StableAudio3Engine("stable-audio-3-small-music", steps=8).steps == 8
        assert StableAudio3Engine("stable-audio-3-small-music").steps == BED_STEPS
        assert StableAudio3Engine("stable-audio-3-small-music", steps=None).steps == BED_STEPS
