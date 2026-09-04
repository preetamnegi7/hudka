"""The licence gate.

This is the constraint the whole project exists to satisfy, so it gets asserted rather
than trusted: a non-commercial model must not be reachable by accident.
"""

from __future__ import annotations

import pytest

from hudka import engines
from hudka.engines.base import LicenceError, require_usable
from hudka.engines.licences import (
    CC_BY_NC,
    MIT,
    STABILITY_COMMUNITY,
    TENCENT_HUNYUAN_COMMUNITY,
)
from hudka.engines.stub import SilenceEngine


class FakeEngine:
    def __init__(self, engine_id, licence):
        self.id = engine_id
        self.licence = licence
        self.kinds = ("sfx",)


class TestGate:
    def test_permissive_engine_passes(self):
        require_usable(FakeEngine("acestep-1.5", MIT),
                       allow_noncommercial=False, opted_in=set())

    def test_stability_engine_passes_without_opt_in(self):
        """The default stack must work with no flags — that is what makes it the default."""
        require_usable(FakeEngine("stable-audio-3-medium", STABILITY_COMMUNITY),
                       allow_noncommercial=False, opted_in=set())

    def test_noncommercial_engine_is_blocked_by_default(self):
        with pytest.raises(LicenceError, match="forbids commercial use"):
            require_usable(FakeEngine("mmaudio", CC_BY_NC),
                           allow_noncommercial=False, opted_in=set())

    def test_noncommercial_engine_needs_an_explicit_flag(self):
        require_usable(FakeEngine("mmaudio", CC_BY_NC),
                       allow_noncommercial=True, opted_in=set())

    def test_territory_restricted_engine_needs_opt_in(self):
        with pytest.raises(LicenceError, match="needs an explicit opt-in"):
            require_usable(FakeEngine("hunyuan-foley", TENCENT_HUNYUAN_COMMUNITY),
                           allow_noncommercial=False, opted_in=set())

    def test_opt_in_message_names_the_excluded_territories(self):
        """A user has to be told what they are accepting, not just that it is restricted."""
        with pytest.raises(LicenceError) as exc:
            require_usable(FakeEngine("hunyuan-foley", TENCENT_HUNYUAN_COMMUNITY),
                           allow_noncommercial=False, opted_in=set())
        message = str(exc.value)
        assert "European Union" in message and "South Korea" in message

    def test_opted_in_engine_passes(self):
        require_usable(FakeEngine("hunyuan-foley", TENCENT_HUNYUAN_COMMUNITY),
                       allow_noncommercial=False, opted_in={"hunyuan-foley"})


class TestRegistry:
    def test_mmaudio_cannot_be_constructed(self):
        """The trap this project is designed to avoid: MIT code, non-commercial weights."""
        with pytest.raises(LicenceError, match="CC-BY-NC"):
            engines.build("mmaudio")

    def test_unknown_engine_is_rejected(self):
        with pytest.raises(ValueError, match="unknown engine"):
            engines.build("definitely-not-real")

    def test_default_engines_are_all_unrestricted(self):
        """Nothing needing a flag may appear in the defaults."""
        for engine_id in engines.DEFAULT_ENGINES.values():
            lic = engines.LICENCE_TABLE[engine_id]
            assert lic.commercial, f"{engine_id} is not commercially usable"
            assert not lic.requires_optin, f"{engine_id} needs an opt-in; unfit as a default"

    def test_stub_engine_builds(self):
        assert isinstance(engines.build("silence"), SilenceEngine)


class TestLicenceFacts:
    def test_stability_terms_are_recorded(self):
        assert STABILITY_COMMUNITY.revenue_cap_usd == 1_000_000
        assert "Freesound" in STABILITY_COMMUNITY.training_data
        assert not STABILITY_COMMUNITY.territory_exclusions

    def test_hunyuan_territory_exclusions_are_recorded(self):
        assert len(TENCENT_HUNYUAN_COMMUNITY.territory_exclusions) == 3
        assert TENCENT_HUNYUAN_COMMUNITY.requires_optin

    def test_summary_flags_noncommercial_clearly(self):
        assert "NON-COMMERCIAL" in CC_BY_NC.summary()


class TestPreflight:
    """A missing engine must be caught before any generation starts.

    Discovering it on the first cue means the user watches a render begin, then fail on
    something knowable up front. These simulate the missing package rather than depending
    on whether it happens to be installed here.
    """

    @staticmethod
    def _hide_engine(monkeypatch):
        """Make `import stable_audio_3` fail, however it is installed."""
        import sys

        monkeypatch.setitem(sys.modules, "stable_audio_3", None)

    def test_missing_engine_is_caught_before_any_stem_is_written(self, tmp_path, monkeypatch):
        import pytest

        from hudka import render
        from hudka.schema import BedCue, CueSheet, VideoInfo

        self._hide_engine(monkeypatch)
        sheet = CueSheet(
            video=VideoInfo(path="x.mp4", duration=10.0, fps=30.0),
            music=[BedCue(id="bed", start=0.0, end=10.0, prompt="a bed",
                          engine="stable-audio-3-medium", seed=1)],
        )
        with pytest.raises(RuntimeError, match="not installed"):
            render.render(sheet, tmp_path)

        stems = tmp_path / "stems"
        assert not stems.exists() or not list(stems.rglob("*.wav")),             "generation started before the engine was known to be usable"

    def test_the_message_names_the_real_install_command(self, monkeypatch):
        import pytest

        from hudka.engines.stable_audio3 import StableAudio3Engine

        self._hide_engine(monkeypatch)
        with pytest.raises(RuntimeError) as exc:
            StableAudio3Engine("stable-audio-3-medium").preflight()
        text = str(exc.value)
        assert "git+https://github.com/Stability-AI/stable-audio-3" in text
        assert "--extra stableaudio" not in text, "stale flag that no longer exists"
        assert "preview" in text, "should point at the offline fallback"

    def test_stub_engine_passes_preflight(self):
        from hudka import engines

        engine = engines.build("silence")
        preflight = getattr(engine, "preflight", None)
        if callable(preflight):
            preflight()

    def test_licence_is_checked_before_availability(self, tmp_path, monkeypatch):
        """A missing install must not mask a licence violation on another engine.

        The licence is the guarantee this project exists to make; a missing package is
        only an inconvenience. Checking them in the wrong order hides the important one.
        """
        import pytest

        from hudka import render
        from hudka.engines.base import LicenceError
        from hudka.schema import BedCue, CueSheet, SfxCue, VideoInfo

        self._hide_engine(monkeypatch)
        sheet = CueSheet(
            video=VideoInfo(path="x.mp4", duration=10.0, fps=30.0),
            # Uninstalled engine listed first, restricted engine second.
            music=[BedCue(id="bed", start=0.0, end=10.0, prompt="a bed",
                          engine="stable-audio-3-medium", seed=1)],
            sfx=[SfxCue(id="s1", at=1.0, duration=1.0, prompt="footsteps",
                        engine="hunyuan-foley", seed=2)],
        )
        with pytest.raises(LicenceError, match="opt-in"):
            render.render(sheet, tmp_path)


class TestModelCacheLocation:
    """Weights belong off the system drive, but the login must survive the redirect.

    Setting HF_HOME moves the token store along with the blob cache, which silently
    un-authenticates an already logged-in machine and produces a 401 that looks exactly
    like a licence that was never accepted.
    """

    def test_cache_redirect_does_not_move_the_token_store(self):
        import inspect

        from hudka import engines

        source = inspect.getsource(engines._point_hf_cache_at_model_dir)
        body = source.split('"""')[-1]
        assert 'os.environ["HF_HUB_CACHE"]' in body
        assert 'os.environ["HF_HOME"]' not in body, \
            "HF_HOME relocates the token store and hides an existing login"

    def test_weights_are_cached_off_the_system_drive(self):
        import os

        from hudka import engines

        cache = os.environ.get("HF_HUB_CACHE")
        assert cache, "model downloads should be redirected away from the default cache"
        assert str(engines.model_dir()) in cache


class TestModelDirSelection:
    """Weights must not land on a removable or exFAT volume.

    They are read by memory-mapping multi-gigabyte files. Doing that from a USB external
    disk fails with STATUS_IN_PAGE_ERROR: the process dies with no Python traceback,
    which is close to undiagnosable from the symptom alone. Free space is the wrong thing
    to choose on, because the external backup drive usually has the most of it.
    """

    def test_exfat_is_rejected_however_much_space_it_has(self, monkeypatch):
        import shutil

        from hudka import engines

        monkeypatch.setattr(engines, "_filesystem_of", lambda drive: "exFAT")
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda drive: type("U", (), {"free": 8_000_000_000_000})(),
        )
        assert not engines._is_suitable("E:\\", need_gb=40.0)

    def test_ntfs_with_room_is_accepted(self, monkeypatch):
        import shutil

        from hudka import engines

        monkeypatch.setattr(engines, "_filesystem_of", lambda drive: "NTFS")
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda drive: type("U", (), {"free": 500_000_000_000})(),
        )
        assert engines._is_suitable("C:\\", need_gb=40.0)

    def test_ntfs_without_room_is_rejected(self, monkeypatch):
        import shutil

        from hudka import engines

        monkeypatch.setattr(engines, "_filesystem_of", lambda drive: "NTFS")
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda drive: type("U", (), {"free": 5_000_000_000})(),
        )
        assert not engines._is_suitable("C:\\", need_gb=40.0)

    def test_explicit_override_wins(self, monkeypatch, tmp_path):
        from hudka import engines

        monkeypatch.setenv("HUDKA_MODEL_DIR", str(tmp_path))
        assert engines.model_dir() == tmp_path

    def test_chosen_directory_is_usable_in_practice(self):
        """Whatever it picks on this machine must pass its own suitability check."""
        import os

        from hudka import engines

        chosen = engines.model_dir()
        if os.name == "nt" and not os.environ.get("HUDKA_MODEL_DIR"):
            drive = str(chosen)[:3]
            assert engines._is_suitable(drive, need_gb=40.0), \
                f"{drive} was selected but is not suitable for memory-mapped weights"
