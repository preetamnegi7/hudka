"""Adopting weights that are already on the machine.

The medium checkpoint was found fully downloaded in the default Hugging Face cache on the
development machine - 9.2 GB the app never looked at, because it redirects HF_HUB_CACHE
off the system drive. The same cache also holds a CC-BY-NC model that must never be
touched, so the adoption rule is structural, not a comment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hudka import engines
from hudka.engines.stable_audio3 import REQUIRED_FILES

MEDIUM = "stabilityai/stable-audio-3-medium"
SHA = "27b5a21bdeadbeef"


def fake_repo(root: Path, repo_id: str, *, complete: bool = True, files=REQUIRED_FILES) -> Path:
    """An HF cache folder with tiny files. Snapshot entries are regular files, which is
    what huggingface_hub writes on a Windows machine without symlink privilege."""
    repo = root / engines._repo_folder(repo_id)
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(SHA, encoding="utf-8")
    (repo / "blobs").mkdir()
    snap = repo / "snapshots" / SHA
    for i, rel in enumerate(files):
        target = snap / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"weights-{i}-".encode() * 50)
        (repo / "blobs" / f"blob{i}").write_bytes(target.read_bytes())
    if not complete:
        (repo / "blobs" / "blob9.incomplete").write_bytes(b"partial")
    return repo


@pytest.fixture
def caches(tmp_path, monkeypatch):
    default, app = tmp_path / "default", tmp_path / "app"
    default.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(app))
    monkeypatch.setattr(engines, "_default_hf_cache", lambda: default)
    return default, app


def listing(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class TestAdoption:
    def test_copies_a_complete_snapshot_and_leaves_the_original(self, caches):
        default, app = caches
        src = fake_repo(default, MEDIUM)
        said = []

        dst = engines.adopt_from_default_cache(MEDIUM, progress=said.append)

        assert dst == app / engines._repo_folder(MEDIUM)
        assert engines.snapshot_complete(dst, REQUIRED_FILES) is not None
        assert listing(src) == listing(dst), "byte-for-byte, and the original still there"
        assert any("copying, the original stays" in s for s in said)
        assert not (app / (engines._repo_folder(MEDIUM) + ".adopting")).exists()

    def test_refuses_an_incomplete_download(self, caches):
        default, app = caches
        fake_repo(default, MEDIUM, complete=False)
        assert engines.adopt_from_default_cache(MEDIUM) is None
        assert not (app / engines._repo_folder(MEDIUM)).exists()

    def test_refuses_a_snapshot_missing_its_text_encoder(self, caches):
        default, app = caches
        fake_repo(default, MEDIUM, files=("model.safetensors", "model_config.json"))
        assert engines.adopt_from_default_cache(MEDIUM) is None
        assert not (app / engines._repo_folder(MEDIUM)).exists()

    def test_will_not_touch_a_non_commercial_model(self, caches):
        """musicgen-small sits in the same cache on the development machine. Its weights
        are CC-BY-NC; adoption must refuse by construction, not by remembering."""
        default, _app = caches
        musicgen = fake_repo(default, "facebook/musicgen-small")
        before = {p: p.stat().st_mtime_ns for p in musicgen.rglob("*")}
        with pytest.raises(ValueError, match="not adoptable"):
            engines.adopt_from_default_cache("facebook/musicgen-small")
        assert {p: p.stat().st_mtime_ns for p in musicgen.rglob("*")} == before

    def test_does_nothing_when_the_app_cache_already_has_it(self, caches):
        default, app = caches
        fake_repo(default, MEDIUM)
        fake_repo(app, MEDIUM)
        said = []
        assert engines.adopt_from_default_cache(MEDIUM, progress=said.append) is None
        assert said == []

    def test_a_failure_mid_copy_leaves_nothing_half_visible(self, caches, monkeypatch):
        default, app = caches
        fake_repo(default, MEDIUM)
        calls = {"n": 0}
        real = engines.shutil.copy2

        def flaky(src, dst, *a, **k):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("disk went away")
            return real(src, dst, *a, **k)

        monkeypatch.setattr(engines.shutil, "copy2", flaky)
        with pytest.raises(OSError):
            engines.adopt_from_default_cache(MEDIUM)
        assert not (app / engines._repo_folder(MEDIUM)).exists()
        assert not (app / (engines._repo_folder(MEDIUM) + ".adopting")).exists()

    def test_preflight_adopts_but_a_failure_is_only_a_warning(self, caches, monkeypatch):
        from hudka.engines.stable_audio3 import StableAudio3Engine

        engine = StableAudio3Engine("stable-audio-3-medium")
        monkeypatch.setattr(engine, "_require_package", lambda: object())

        def broken(repo_id, progress=None):
            raise OSError("no permission")

        monkeypatch.setattr(engines, "adopt_from_default_cache", broken)
        said = []
        engine.preflight(progress=said.append)          # must not raise
        assert any("could not adopt weights" in s for s in said)


class TestSnapshotComplete:
    def test_missing_refs_is_incomplete(self, tmp_path):
        assert engines.snapshot_complete(tmp_path / "nothing", REQUIRED_FILES) is None

    def test_empty_file_is_incomplete(self, tmp_path):
        repo = fake_repo(tmp_path, MEDIUM)
        (repo / "snapshots" / SHA / "model.safetensors").write_bytes(b"")
        assert engines.snapshot_complete(repo, REQUIRED_FILES) is None
