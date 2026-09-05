"""The GUI server.

This process has the user's filesystem underneath it and accepts uploads, so path
handling and validation are asserted rather than assumed. Jobs are driven to completion
so the async paths are covered too, not just the request that starts them.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from hudka.ui.server import create_app

from .conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def wait_for(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    """Block until a background job finishes, then return it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/job/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(tmp_path_factory.mktemp("workspace")))


@pytest.fixture(scope="module")
def project(client, fixture_video):
    """A project imported by path, analysed, scaffolded on the stub engine, and rendered.

    A *real* render on the stub, not a preview: a preview is confined to preview/ and is
    deliberately not a render, so it would leave the project at the wrong stage for
    everything downstream that expects final.mp4.
    """
    res = client.post("/api/import/path", json={"path": str(fixture_video)})
    assert res.status_code == 200
    name = res.json()["name"]

    job = client.post(f"/api/project/{name}/analyze").json()
    assert wait_for(client, job["id"])["status"] == "done"

    assert client.post(f"/api/project/{name}/scaffold",
                       json={"preset": "short-form", "engine": "silence"}).status_code == 200

    job = client.post(f"/api/project/{name}/render", json={}).json()
    assert wait_for(client, job["id"])["status"] == "done"
    return name


class TestImport:
    def test_serves_the_page(self, client):
        res = client.get("/")
        assert res.status_code == 200 and "Hudka" in res.text

    def test_import_by_path(self, client, fixture_video):
        res = client.post("/api/import/path", json={"path": str(fixture_video)})
        assert res.status_code == 200
        assert res.json()["name"]

    def test_import_by_upload(self, client, fixture_video):
        with open(fixture_video, "rb") as handle:
            res = client.post("/api/import/upload",
                              files={"file": ("clip.mp4", handle, "video/mp4")})
        assert res.status_code == 200
        name = res.json()["name"]
        assert client.get(f"/media/{name}/source").status_code == 200

    def test_rejects_a_non_video_upload(self, client):
        res = client.post("/api/import/upload",
                          files={"file": ("notes.txt", b"hello", "text/plain")})
        assert res.status_code == 415

    def test_missing_path_is_reported(self, client):
        res = client.post("/api/import/path", json={"path": "C:/nope/missing.mp4"})
        assert res.status_code == 404

    def test_names_do_not_collide(self, client, fixture_video):
        first = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        second = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        assert first != second


class TestWorkflow:
    def test_library_lists_the_project(self, client, project):
        data = client.get("/api/library").json()
        assert any(p["name"] == project for p in data["projects"])
        assert "short-form" in data["presets"]
        assert data["engine_available"]["silence"] is True

    def test_stage_advances_to_rendered(self, client, project):
        assert client.get(f"/api/project/{project}").json()["stage"] == "rendered"

    def test_analysis_results_are_exposed(self, client, project):
        data = client.get(f"/api/project/{project}").json()
        assert len(data["shots"]) >= 3
        assert data["contact_sheets"]

    def test_scaffold_produces_cues(self, client, project):
        cues = client.get(f"/api/project/{project}").json()["cues"]
        assert cues["music"] and cues["sfx"]

    def test_render_reports_loudness(self, client, project):
        job = [j for j in client.get("/api/library").json()["jobs"]
               if j["project"] == project and j["kind"] == "render"][0]
        assert job["result"]["lufs"] == pytest.approx(-14.0, abs=1.5)

    def test_preview_mode_leaves_the_saved_sheet_alone(self, client, fixture_video):
        """Preview swaps engines at render time; it must not rewrite cues.json."""
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        wait_for(client, client.post(f"/api/project/{name}/analyze").json()["id"])
        client.post(f"/api/project/{name}/scaffold", json={"preset": "short-form"})
        before = client.get(f"/api/project/{name}").json()["cues"]
        assert all(c["engine"] != "silence" for c in before["sfx"])

        wait_for(client, client.post(f"/api/project/{name}/render",
                                     json={"preview": True}).json()["id"])

        after = client.get(f"/api/project/{name}").json()["cues"]
        assert [c["engine"] for c in after["sfx"]] == [c["engine"] for c in before["sfx"]], \
            "preview mode overwrote the saved engines"

    def test_scaffold_before_analysis_is_refused(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        res = client.post(f"/api/project/{name}/scaffold", json={"preset": "short-form"})
        assert res.status_code == 409

    def test_render_without_cues_is_refused(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        res = client.post(f"/api/project/{name}/render", json={})
        assert res.status_code == 409


class TestMedia:
    def test_serves_video_and_stems(self, client, project):
        assert client.get(f"/media/{project}/video").status_code == 200
        assert client.get(f"/media/{project}/source").status_code == 200
        stems = client.get(f"/api/project/{project}").json()["stems"]
        assert stems
        assert client.get(f"/media/{project}/stem/{stems[0]}").status_code == 200

    def test_downloads_are_attachments(self, client, project):
        res = client.get(f"/download/{project}/video")
        assert res.status_code == 200
        assert "attachment" in res.headers.get("content-disposition", "")
        assert client.get(f"/download/{project}/licence").status_code == 200

    def test_unknown_stem_is_404(self, client, project):
        assert client.get(f"/media/{project}/stem/nope").status_code == 404


class TestSafety:
    def test_project_name_cannot_escape_the_workspace(self, client):
        """A crafted project name must not reach outside the workspace directory.

        Percent-encoded separators only: an unencoded `../..` is collapsed by the HTTP
        client before the request is sent, so it never exercises the handler.
        """
        for attempt in ("..%2F..", "..%5C..", "%2e%2e%2f%2e%2e", "....//", "%2e%2e"):
            res = client.get(f"/api/project/{attempt}")
            assert res.status_code == 404, f"{attempt} reached the handler"

    def test_contact_sheet_path_traversal_is_refused(self, client, project):
        for attempt in ("..%2F..%2Fcues.json", "..%5Ccues.json", "....//cues.json"):
            assert client.get(f"/media/{project}/contact/{attempt}").status_code == 404

    def test_invalid_cue_sheet_is_rejected_with_a_readable_reason(self, client, project):
        cues = client.get(f"/api/project/{project}").json()["cues"]
        cues["sfx"][0]["at"] = 9999.0
        res = client.put(f"/api/project/{project}/cues", json=cues)
        assert res.status_code == 422
        # Errors come back as a list so the page can point at the cue that is wrong.
        assert "past the" in " ".join(e["message"] for e in res.json()["detail"])

    def test_valid_cue_sheet_round_trips(self, client, project):
        cues = client.get(f"/api/project/{project}").json()["cues"]
        cues["sfx"][0]["prompt"] = "a reworded whoosh"
        assert client.put(f"/api/project/{project}/cues", json=cues).status_code == 200
        after = client.get(f"/api/project/{project}").json()["cues"]
        assert after["sfx"][0]["prompt"] == "a reworded whoosh"

    def test_restricted_engine_is_refused_at_render(self, client, project):
        """The licence gate must hold through the GUI, not just the CLI."""
        cues = client.get(f"/api/project/{project}").json()["cues"]
        original = cues["sfx"][0]["engine"]
        cues["sfx"][0]["engine"] = "hunyuan-foley"
        client.put(f"/api/project/{project}/cues", json=cues)

        job = client.post(f"/api/project/{project}/render", json={}).json()
        finished = wait_for(client, job["id"])
        assert finished["status"] == "error"
        assert "opt-in" in finished["error"]

        cues["sfx"][0]["engine"] = original
        client.put(f"/api/project/{project}/cues", json=cues)

    def test_deleting_a_project_removes_it(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        assert client.delete(f"/api/project/{name}").status_code == 200
        assert client.get(f"/api/project/{name}").status_code == 404


class TestFreshProjectContract:
    """A just-imported project reports empty analysis, and the page must handle that.

    An empty list is truthy in JavaScript. Gating the Analyse step on `proj.shots`
    rather than `proj.shots.length` skipped the step entirely, so the page offered
    "Create cue sheet" on an unanalysed project and the server refused it with no way
    forward. These pin both halves of that contract.
    """

    def test_fresh_project_reports_empty_analysis(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        data = client.get(f"/api/project/{name}").json()
        assert data["stage"] == "imported"
        assert data["shots"] == []
        assert data["cues"] is None

    def test_page_gates_the_analyse_step_on_length(self):
        from hudka.ui.server import HERE

        page = (HERE / "index.html").read_text(encoding="utf-8")
        assert "!(proj.shots || []).length" in page,             "the Analyse step must test array length, not array truthiness"
        assert "if (!proj.shots)" not in page,             "truthiness check on proj.shots reintroduces the stuck-on-import bug"


class TestValidationFeedback:
    """A save that fails must say which cue is wrong and what to do about it.

    Render saves before rendering, so an unreadable validation failure made the Render
    button appear to do nothing at all.
    """

    def test_empty_prompt_names_the_offending_cue(self, client, project):
        cues = client.get(f"/api/project/{project}").json()["cues"]
        cues["sfx"].append({
            "id": "sfx99", "at": 1.0, "duration": 1.5, "prompt": "",
            "engine": "silence", "gain_db": -6, "pan": 0, "seed": 5,
            "align_transient": True, "shot": None, "note": "",
        })
        res = client.put(f"/api/project/{project}/cues", json=cues)
        assert res.status_code == 422

        detail = res.json()["detail"]
        assert isinstance(detail, list)
        entry = next(e for e in detail if e["cue"] == "sfx99")
        assert "description of the sound" in entry["message"]

    def test_every_bad_cue_is_reported_not_just_the_first(self, client, project):
        cues = client.get(f"/api/project/{project}").json()["cues"]
        for n in (1, 2, 3):
            cues["sfx"].append({
                "id": f"bad{n}", "at": float(n), "duration": 1.0, "prompt": "",
                "engine": "silence", "gain_db": -6, "pan": 0, "seed": n,
                "align_transient": True, "shot": None, "note": "",
            })
        detail = client.put(f"/api/project/{project}/cues", json=cues).json()["detail"]
        assert {"bad1", "bad2", "bad3"} <= {e["cue"] for e in detail}

    def test_overlapping_music_beds_explain_themselves(self, client, project):
        cues = client.get(f"/api/project/{project}").json()["cues"]
        span = cues["video"]["duration"]
        cues["music"] = [
            {"id": "m1", "start": 0.0, "end": span, "prompt": "a bed",
             "engine": "silence", "gain_db": -18, "fade_in": 0.5, "fade_out": 1.0,
             "duck": True, "loop": False, "seed": 1, "note": ""},
            {"id": "m2", "start": 2.0, "end": span, "prompt": "another bed",
             "engine": "silence", "gain_db": -18, "fade_in": 0.5, "fade_out": 1.0,
             "duck": True, "loop": False, "seed": 2, "note": ""},
        ]
        res = client.put(f"/api/project/{project}/cues", json=cues)
        assert res.status_code == 422
        assert "overlap" in " ".join(e["message"] for e in res.json()["detail"])

    def test_page_gives_new_cues_a_valid_starter_prompt(self):
        """Otherwise "+ effect" creates a cue that can never be saved."""
        from hudka.ui.server import HERE

        page = (HERE / "index.html").read_text(encoding="utf-8")
        assert "STARTER_PROMPT" in page
        assert "prompt: ''" not in page, "a new cue must not start with an empty prompt"


class TestEngineAuthReporting:
    """The gated weights are the last blocker, so the page must state it, not discover it.

    `model_info` succeeds on a gated repo it cannot download from, so the check has to be
    `auth_check` - otherwise the app reports everything is fine until a render dies.
    """

    def test_library_reports_auth_state(self, client):
        auth = client.get("/api/library").json()["engine_auth"]
        assert set(auth) >= {"needed", "logged_in", "authorised"}

    def test_auth_check_is_used_rather_than_model_info(self):
        import inspect

        from hudka.engines import stable_audio3

        source = inspect.getsource(stable_audio3.StableAudio3Engine.is_authorised)
        assert "auth_check(" in source
        assert "model_info(" not in source,             "model_info succeeds on a gated repo that cannot actually be downloaded"

    def test_gated_failure_explains_both_steps(self):
        from hudka.engines.stable_audio3 import StableAudio3Engine

        message = StableAudio3Engine("stable-audio-3-medium")._gated_message()
        assert "huggingface.co/stabilityai/stable-audio-3-medium" in message
        assert "hf auth login" in message
        assert "preview" in message

    def test_gated_errors_are_recognised(self):
        """Detection works on the message text, so it survives hub version changes."""
        import inspect

        from hudka.engines import stable_audio3
        from hudka.engines.stable_audio3 import _is_gated

        assert _is_gated(RuntimeError("401 Client Error"))
        assert _is_gated(RuntimeError("Cannot access gated repo for url ..."))
        assert not _is_gated(RuntimeError("CUDA out of memory"))
        # The typed check is the primary path; the text match is the fallback.
        assert "GatedRepoError" in inspect.getsource(stable_audio3._is_gated)


class TestOfflineOperation:
    """Generation, mixing and mastering must work with no network at all.

    The only thing that ever needs the internet is the one-time weight download. Anything
    else reaching out turns a local tool into one that stalls or leaks when it should not.
    """

    def test_no_outbound_calls_once_weights_are_cached(self, monkeypatch, tmp_path):
        import hudka.ui.server as server

        weights = tmp_path / "models--stabilityai--stable-audio-3-small-sfx" / "snapshots" / "abc"
        weights.mkdir(parents=True)
        (weights / "model.safetensors").write_bytes(b"x")
        monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setattr(server, "_AUTH_CACHE", None)

        def explode(*a, **k):
            raise AssertionError("reached the network when weights were already local")

        monkeypatch.setattr("huggingface_hub.HfApi.auth_check", explode)
        assert server._engine_auth()["needed"] is False

    def test_offline_flag_short_circuits_the_check(self, monkeypatch):
        import hudka.ui.server as server

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr(server, "_AUTH_CACHE", None)
        monkeypatch.setattr(
            "huggingface_hub.HfApi.auth_check",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("called while offline")),
        )
        result = server._engine_auth()
        assert result["offline"] is True and result["needed"] is False

    def test_result_is_cached_across_page_loads(self, monkeypatch):
        import hudka.ui.server as server

        monkeypatch.setattr(server, "_AUTH_CACHE", None)
        calls = []
        monkeypatch.setattr("huggingface_hub.HfApi.auth_check",
                            lambda self, repo, **k: calls.append(repo))
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setenv("HF_HUB_CACHE", "/nonexistent-so-the-check-runs")

        server._engine_auth()
        server._engine_auth()
        server._engine_auth()
        assert len(calls) <= 1, "the page reached out once per load instead of once per process"


class TestSheetForwardCompatibility:
    """A cue sheet written before a control existed must still drive that control.

    The page binds inputs directly to cue fields. Serving the raw file means an older
    sheet hands the UI `undefined` for every new field, and the controls silently do
    nothing - so sheets are validated on the way out and filled with current defaults.
    """

    def test_older_sheet_is_served_with_current_defaults(self, client, project, tmp_path):
        import json

        from hudka.ui.server import create_app

        cues = client.get(f"/api/project/{project}").json()["cues"]

        # Strip every field added after the sheet format settled, as an old file would.
        trimmed = dict(cues)
        for key in ("music_bus_db", "sfx_bus_db", "ambience_bus_db", "duck_depth_db"):
            trimmed.pop(key, None)
        for cue in trimmed["sfx"]:
            for key in ("highpass_hz", "lowpass_hz", "pitch_semitones", "reverse",
                        "fade_in", "fade_out", "steps", "cfg_scale", "negative_prompt"):
                cue.pop(key, None)

        from fastapi.testclient import TestClient
        import shutil

        workspace = tmp_path / "ws"
        (workspace / project).mkdir(parents=True)
        (workspace / project / "cues.json").write_text(json.dumps(trimmed), encoding="utf-8")
        old_client = TestClient(create_app(workspace))

        served = old_client.get(f"/api/project/{project}").json()["cues"]
        assert served["music_bus_db"] == 0.0
        assert served["sfx"][0]["pitch_semitones"] == 0.0
        assert served["sfx"][0]["reverse"] is False
        assert served["sfx"][0]["steps"] is None

    def test_unparseable_sheet_is_still_served(self, tmp_path):
        import json

        from fastapi.testclient import TestClient

        from hudka.ui.server import create_app

        workspace = tmp_path / "ws"
        (workspace / "broken").mkdir(parents=True)
        (workspace / "broken" / "cues.json").write_text(
            json.dumps({"nonsense": True}), encoding="utf-8")

        client = TestClient(create_app(workspace))
        res = client.get("/api/project/broken")
        assert res.status_code == 200, "a broken sheet must stay openable, not 500"


class TestPreviewThroughTheGui:
    """The page must present a preview as a preview at every point it can be seen."""

    def test_preview_does_not_advance_the_stage(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        assert wait_for(client, client.post(f"/api/project/{name}/analyze").json()["id"])["status"] == "done"
        client.post(f"/api/project/{name}/scaffold", json={"preset": "short-form"})

        job = wait_for(client, client.post(f"/api/project/{name}/render",
                                           json={"preview": True}).json()["id"])
        assert job["status"] == "done"
        assert job["result"]["preview"] is True

        data = client.get(f"/api/project/{name}").json()
        assert data["stage"] == "designed", "a preview is not a render"
        assert data["has_preview"] is True

    def test_preview_can_be_watched_but_not_downloaded(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        wait_for(client, client.post(f"/api/project/{name}/analyze").json()["id"])
        client.post(f"/api/project/{name}/scaffold", json={"preset": "short-form"})
        wait_for(client, client.post(f"/api/project/{name}/render", json={"preview": True}).json()["id"])

        assert client.get(f"/media/{name}/preview").status_code == 200
        assert client.get(f"/media/{name}/video").status_code == 404
        res = client.get(f"/download/{name}/video")
        assert res.status_code == 409
        assert "preview" in res.json()["detail"]

    def test_page_resets_the_checkbox_and_never_says_on_target_for_a_preview(self):
        from hudka.ui.server import HERE

        page = (HERE / "index.html").read_text(encoding="utf-8")
        assert "$('#previewMode').checked = false" in page, "preview must not be sticky"
        assert "placeholder tones, not real audio" in page
        assert "previewBanner" in page
        # The green success path must be unreachable for a preview result.
        assert "if (r.preview) {" in page

    def test_a_preview_over_a_finished_render_still_announces_itself(self, client, project):
        """The case that got through: every other preview test starts from a project that
        was never rendered, where the stage stays 'designed'. Preview an already-rendered
        project and the stage stays 'rendered' - and the page's banner used to be gated on
        `proj.stage !== 'rendered'`, so the one warning that says "this is not real audio"
        was wiped by the refresh that runs immediately after the render finishes."""
        wait_for(client, client.post(f"/api/project/{project}/render", json={}).json()["id"])
        first = client.get(f"/api/project/{project}").json()
        assert first["stage"] == "rendered"
        assert first["rendered_at"] is not None
        assert first["has_preview"] is False

        job = wait_for(client, client.post(f"/api/project/{project}/render",
                                           json={"preview": True}).json()["id"])
        assert job["result"]["preview"] is True

        after = client.get(f"/api/project/{project}").json()
        assert after["stage"] == "rendered", "a preview must not undo a finished render"
        assert after["has_preview"] is True
        assert after["rendered_at"] == first["rendered_at"], \
            "a preview must not touch final.mp4"
        assert after["preview_at"] is not None

        from hudka.ui.server import HERE

        page = (HERE / "index.html").read_text(encoding="utf-8")
        assert "if (viewPreview && proj.has_preview) {" in page, \
            "the preview banner must not be gated on the stage"
        # The two videos must cache-bust independently, or a re-render serves the old cut.
        assert "src.endsWith('/preview') ? proj.preview_at : proj.rendered_at" in page

    def test_a_preview_is_still_there_after_the_project_is_reopened(self, client, fixture_video):
        """The defect behind "preview never worked": viewPreview had exactly one writer -
        startPoll, when a render job finished - so a preview could be seen only between
        that job ending and the next reload. Reopening went back to the untouched source
        with no control anywhere to ask for the preview again."""
        from hudka.ui.server import HERE

        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        wait_for(client, client.post(f"/api/project/{name}/analyze").json()["id"])
        client.post(f"/api/project/{name}/scaffold", json={"preset": "short-form"})
        wait_for(client, client.post(f"/api/project/{name}/render",
                                     json={"preview": True}).json()["id"])

        reopened = client.get(f"/api/project/{name}").json()
        assert reopened["stage"] == "designed" and reopened["has_preview"] is True
        assert reopened["preview_report"]["preview"] is True
        assert reopened["preview_provenance"], "the preview keeps its own ledger"
        assert reopened["stems"] == [], "a preview must not write into the real stem tree"
        assert reopened["preview_stems"], "the preview's own stems must be reachable"
        assert reopened["cues_at"] is not None

        cue = reopened["preview_stems"][0]
        assert client.get(f"/media/{name}/preview-stem/{cue}").status_code == 200
        assert client.get(f"/media/{name}/stem/{cue}").status_code == 404,             "the trees must not be merged - a mixture plays real audio for some cues and a"
        assert client.get(f"/download/{name}/preview-licence").status_code == 200

        page = (HERE / "index.html").read_text(encoding="utf-8")
        assert "viewPreview = !!(proj.has_preview && proj.stage !== 'rendered')" in page
        assert "proj.preview_report" in page and "proj.preview_stems" in page

        assert client.delete(f"/api/project/{name}/preview").status_code == 200
        gone = client.get(f"/api/project/{name}").json()
        assert gone["has_preview"] is False and gone["preview_stems"] == []
        assert client.get(f"/download/{name}/preview-licence").status_code == 404


class TestLiveMix:
    """M and S were never broken - they were inaudible. These pin the graph that fixed
    that, and the three measured constants that are silently wrong if they drift."""

    @staticmethod
    def _page() -> str:
        from hudka.ui.server import HERE

        return (HERE / "index.html").read_text(encoding="utf-8")

    def test_mute_and_solo_reach_the_audition(self):
        page = self._page()
        assert "function auditionApply()" in page
        assert page.count("auditionApply();") >= 4, "M, S, gain and pan must all apply"
        assert "dims every other track" not in page, "solo is audible now, not cosmetic"

    def test_the_pan_law_is_the_mixers_not_the_browsers(self):
        """StereoPannerNode folds one channel into the other: +6.02 dB at hard pan where
        mix._apply_pan gives +3.01. Measured, not assumed."""
        page = self._page()
        assert "createStereoPanner" not in page
        assert "Math.cos(a) * Math.SQRT2" in page

    def test_the_cache_key_is_never_recomputed_in_the_page(self):
        """engines.base joins the parts with a 0x1f separator. A second implementation in
        JavaScript would be free to drift from the one the renderer uses, and the obvious
        JS version gets that separator wrong - which would mark every cue permanently
        stale. The page asks the server instead."""
        assert "crypto.subtle" not in self._page()

    def test_the_media_source_is_created_once_and_only_from_the_button(self):
        """createMediaElementSource permanently reroutes the video's audio and throws on a
        second call for the same element, so it must be unreachable from the Space
        handler - the first Space press would otherwise mute the video for good."""
        page = self._page()
        assert page.count("createMediaElementSource") == 1
        space = page.split("case ' ':")[1].split("case ")[0]
        assert "armAudition" not in space and "ensureCtx" in space

    def test_the_live_mix_plays_the_source_not_the_finished_render(self):
        """final.mp4 has the dialogue already fused with the music and effects; only the
        source carries the untouched original that mix.extract_original_audio pulls out."""
        assert "if (!wantOriginal && !AUD.on) {" in self._page()

    def test_normalisation_honours_both_constraints(self):
        """audio.normalize_one_shot takes the MORE ATTENUATING of the peak and RMS
        targets, so a long riser is caught by RMS where a click is caught by peak."""
        page = self._page()
        assert "REF_SFX_PEAK_DBFS = -12.0" in page
        assert "REF_SFX_RMS_CEILING_DBFS = -26.0" in page
        assert "Math.min(REF_SFX_PEAK_DBFS - peak, REF_SFX_RMS_CEILING_DBFS - rms)" in page

    def test_you_can_hear_a_cue_sheet_without_rendering_it(self, client, fixture_video):
        """Generation is the slow part that answers "does this sound right". Placing,
        mixing, mastering to a loudness target and re-muxing a 4K video are not - and they
        were the price of admission before you were allowed to hear anything.

        Its own project, not the shared fixture: that one is already rendered, and the
        whole point here is the state where nothing has been.
        """
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        wait_for(client, client.post(f"/api/project/{name}/analyze").json()["id"])
        client.post(f"/api/project/{name}/scaffold",
                    json={"preset": "short-form", "engine": "silence"})

        before = client.get(f"/api/project/{name}").json()
        assert before["stage"] == "designed"
        assert before["stems"] == [], "nothing to listen to yet"

        job = wait_for(client, client.post(f"/api/project/{name}/generate", json={}).json()["id"])
        assert job["status"] == "done", job.get("error")
        assert job["result"]["generated"] > 0

        after = client.get(f"/api/project/{name}").json()
        assert after["stems"], "the stems the live mix plays"
        assert after["stem_info"], "and the levels it needs to place them at"
        assert after["stage"] == "designed", "generating is not rendering"
        assert client.get(f"/media/{name}/video").status_code == 404, "no video was made"
        assert client.get(f"/media/{name}/stem/{after['stems'][0]}").status_code == 200

        # Rendering afterwards must reuse every one of them, or this cost the user twice.
        rendered = wait_for(client, client.post(f"/api/project/{name}/render", json={}).json()["id"])
        assert rendered["result"]["generated"] == 0
        assert rendered["result"]["cached"] == len(after["stems"])

    def test_generating_offers_itself_when_there_is_nothing_to_hear(self):
        page = self._page()
        assert "function cuesNeedingStems()" in page
        assert "'Generate & listen'" in page
        assert "/generate`" in page

    def test_the_server_serves_what_the_graph_needs(self, client, project):
        wait_for(client, client.post(f"/api/project/{project}/render", json={}).json()["id"])
        data = client.get(f"/api/project/{project}").json()

        info = data["stem_info"]
        assert info, "every rendered stem needs its content hash and kind"
        for entry in info.values():
            assert entry["key"], "the hash is what tells the page a stem is stale"
            if entry["kind"] in ("music", "ambience"):
                assert entry["lufs"] is not None or entry["lufs"] is None
            else:
                assert entry["lufs"] is None, "effects are normalised in the page, not measured"

        assert "lufs" in client.get(f"/api/project/{project}/dialogue_lufs").json()

        cue = data["cues"]["sfx"][0]
        first = client.post(f"/api/project/{project}/cue_key", json={"cue": cue}).json()["key"]
        assert first == info[cue["id"]]["key"], "an untouched cue is not stale"

        # Moving it must NOT change the key (the stem survives a drag), but rewording it
        # must - that is the whole point of the staleness check.
        moved = dict(cue, at=cue["at"] + 5)
        assert client.post(f"/api/project/{project}/cue_key",
                           json={"cue": moved}).json()["key"] == first
        reworded = dict(cue, prompt=cue["prompt"] + " and a low thump")
        assert client.post(f"/api/project/{project}/cue_key",
                           json={"cue": reworded}).json()["key"] != first


class TestNewMusic:
    def test_new_music_changes_the_style_and_nothing_else(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        wait_for(client, client.post(f"/api/project/{name}/analyze").json()["id"])
        client.post(f"/api/project/{name}/scaffold", json={"preset": "short-form", "engine": "silence"})
        before = client.get(f"/api/project/{name}").json()["cues"]["music"][0]
        before["gain_db"] = -9.5                      # a user edit that must survive
        client.put(f"/api/project/{name}/cues",
                   json=client.get(f"/api/project/{name}").json()["cues"] | {
                       "music": [before]})

        res = client.post(f"/api/project/{name}/cue/{before['id']}/new_music")
        assert res.status_code == 200
        after = client.get(f"/api/project/{name}").json()["cues"]["music"][0]
        assert after["prompt"] != before["prompt"] and after["seed"] != before["seed"]
        assert res.json()["mood"] in after["note"]
        for key in ("start", "end", "gain_db", "fade_in", "fade_out", "engine", "loop"):
            assert after[key] == before[key], key

    def test_recreating_the_cue_sheet_changes_the_music(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        wait_for(client, client.post(f"/api/project/{name}/analyze").json()["id"])
        client.post(f"/api/project/{name}/scaffold", json={"preset": "short-form"})
        first = client.get(f"/api/project/{name}").json()["cues"]["music"][0]
        client.post(f"/api/project/{name}/scaffold", json={"preset": "short-form"})
        second = client.get(f"/api/project/{name}").json()["cues"]["music"][0]
        assert (second["prompt"], second["seed"]) != (first["prompt"], first["seed"])

    def test_unknown_bed_is_404(self, client, project):
        assert client.post(f"/api/project/{project}/cue/nope/new_music").status_code == 404

    def test_page_offers_new_music_on_bed_cards(self):
        from hudka.ui.server import HERE

        page = (HERE / "index.html").read_text(encoding="utf-8")
        assert "mk('new music'" in page and "/new_music`" in page


class TestMachineInTheGui:
    """The page used to expose no hardware information at all, and listed the medium
    engine as a plain selectable option on a card that would die loading it."""

    @staticmethod
    def _hw(**kw):
        from hudka.engines.hardware import Hardware

        base = dict(device="cuda", gpu_name="NVIDIA GeForce RTX 4070", total_vram_gb=12.9,
                    free_vram_gb=9.0, bf16=True, ram_gb=64.0, cores=24, torch_build="2.7.1+cu128")
        base.update(kw)
        return Hardware(**base)

    def test_library_reports_the_machine(self, client):
        data = client.get("/api/library").json()
        hw = data["hardware"]
        assert hw["tier"] in ("cpu", "gpu-lite", "gpu-medium", "gpu-large")
        assert hw["summary"] and hw["reason"]
        assert set(hw["plan"]) == {"music", "ambience", "sfx"}
        assert data["engine_status"]["silence"]["runnable"] is True

    def test_medium_is_labelled_when_the_card_is_busy(self, monkeypatch):
        from hudka.ui import server as server_mod

        monkeypatch.setattr(server_mod.hardware, "detect", lambda refresh=False: self._hw(free_vram_gb=5.1))
        monkeypatch.setattr(server_mod, "_engine_availability",
                            lambda: {"silence": True, "stable-audio-3-medium": True,
                                     "stable-audio-3-small-sfx": True, "acestep-1.5": False,
                                     "hunyuan-foley": False})
        status = server_mod._engine_status()
        medium = status["stable-audio-3-medium"]
        assert medium["runnable"] is False
        assert medium["reason"] == "needs 8.4 GB free VRAM · 5.1 GB now"
        assert status["stable-audio-3-small-sfx"]["runnable"] is True
        assert status["acestep-1.5"] == {"installed": False, "runnable": False, "reason": "not installed"}

    def test_medium_is_runnable_when_it_fits(self, monkeypatch):
        from hudka.ui import server as server_mod

        monkeypatch.setattr(server_mod.hardware, "detect", lambda refresh=False: self._hw(free_vram_gb=9.0))
        monkeypatch.setattr(server_mod, "_engine_availability",
                            lambda: {"silence": True, "stable-audio-3-medium": True})
        assert server_mod._engine_status()["stable-audio-3-medium"]["runnable"] is True

    def test_the_payload_names_what_holds_the_card(self, monkeypatch):
        from hudka.ui import server as server_mod

        monkeypatch.setattr(server_mod.hardware, "detect", lambda refresh=False: self._hw(free_vram_gb=5.1))
        hw = server_mod._hardware_payload()
        assert hw["tier"] == "gpu-lite" and hw["held_by_others_gb"] == 7.8
        assert "other applications" in hw["reason"]
        assert hw["plan"]["music"]["engine"] == "stable-audio-3-small-music"

    def test_page_disables_engines_that_cannot_run_here(self):
        from hudka.ui.server import HERE

        page = (HERE / "index.html").read_text(encoding="utf-8")
        assert "engineOptions(cue.engine)" in page, "the current value must stay enabled"
        assert "opt.disabled ? ' disabled'" in page
        assert "lib.engine_status" in page
        assert 'id="hwLine"' in page and "hw.summary" in page
        assert "other applications are holding" in page

    def test_page_carries_the_quality_control(self):
        from hudka.ui.server import HERE

        page = (HERE / "index.html").read_text(encoding="utf-8")
        assert 'id="quality"' in page
        assert "proj.cues.quality = $('#quality').value" in page
        assert "$('#quality').value = proj.cues.quality || 'auto'" in page


class TestVerdictThroughTheGui:
    def test_render_result_and_project_carry_the_verdict(self, client, project):
        job = wait_for(client, client.post(f"/api/project/{project}/render", json={}).json()["id"])
        assert job["result"]["verdict"] in ("ok", "warn")
        assert isinstance(job["result"]["warnings"], list)

        data = client.get(f"/api/project/{project}").json()
        assert data["report"]["verdict"] == job["result"]["verdict"]
        assert data["verdict"] == job["result"]["verdict"]

        lib = client.get("/api/library").json()
        mine = next(p for p in lib["projects"] if p["name"] == project)
        assert mine["verdict"] == job["result"]["verdict"]


class TestDeleteIsRecoverable:
    """Delete moves a project to the trash; only a second, separate step destroys it.

    Three projects holding a rendered video and dozens of cached stems vanished after a
    confirmed click, straight past the Recycle Bin. A project is hours of GPU work and
    possibly a hand-tuned cue sheet; one click must not be the end of it.
    """

    def test_deleted_project_is_restorable(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        assert client.delete(f"/api/project/{name}").status_code == 200
        assert client.get(f"/api/project/{name}").status_code == 404

        trash = client.get("/api/trash").json()["items"]
        entry = next(t for t in trash if t["name"] == name)
        assert client.post(f"/api/trash/{entry['trashed_as']}/restore").status_code == 200
        assert client.get(f"/api/project/{name}").status_code == 200

    def test_trash_is_not_addressable_as_a_project(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        client.delete(f"/api/project/{name}")
        assert client.get("/api/project/.trash").status_code == 404
        assert all(p["name"] != ".trash" for p in client.get("/api/library").json()["projects"])

    def test_purge_is_the_only_irreversible_step(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        client.delete(f"/api/project/{name}")
        entry = next(t for t in client.get("/api/trash").json()["items"] if t["name"] == name)
        assert client.delete(f"/api/trash/{entry['trashed_as']}").status_code == 200
        assert all(t["trashed_as"] != entry["trashed_as"]
                   for t in client.get("/api/trash").json()["items"])

    def test_restore_does_not_clobber_a_reused_name(self, client, fixture_video):
        name = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        client.delete(f"/api/project/{name}")
        entry = next(t for t in client.get("/api/trash").json()["items"] if t["name"] == name)
        # The same name gets created again in the meantime.
        again = client.post("/api/import/path", json={"path": str(fixture_video)}).json()["name"]
        assert again == name
        restored = client.post(f"/api/trash/{entry['trashed_as']}/restore").json()["name"]
        assert restored != name and restored.startswith(name)


class TestEditorAffordances:
    """Small page bugs that each made a correct action look like a failure.

    Static assertions on the page source, the pattern already used above: these are
    vanilla-JS behaviours with no server side to exercise through TestClient.
    """

    @staticmethod
    def _page() -> str:
        from hudka.ui.server import HERE

        return (HERE / "index.html").read_text(encoding="utf-8")

    def test_new_cue_ids_avoid_collisions(self):
        """length+1 gave a second sfx03 after deleting sfx02; the schema then refused
        the sheet and blamed the cue the user had just created."""
        page = self._page()
        assert "while (used.has(" in page
        assert "(proj.cues[kind] || []).length + 1" not in page

    def test_unsaved_edits_change_the_save_button(self):
        page = self._page()
        assert "function markClean()" in page
        assert "'Save changes'" in page

    def test_ctrl_s_saves(self):
        page = self._page()
        assert "e.key === 's'" in page

    def test_choosing_a_preset_moves_the_loudness_target(self):
        """Otherwise a preset picked for its guidance keeps the previous target."""
        page = self._page()
        assert "if (id === 'preset')" in page
        assert "pre.target_lufs" in page

    def test_numeric_fields_ignore_unparsable_intermediates(self):
        """Clearing 'at' to retype it used to read as 0 and snap the cue to the start."""
        page = self._page()
        assert "Number.isFinite(parseFloat(input.value))" in page

    def test_ab_label_does_not_accumulate(self):
        page = self._page()
        assert "$('#abWrap').lastChild.textContent" in page
        assert "$('#abWrap').firstChild.textContent = ''" not in page


class TestTimeline:
    """The multi-track timeline. Vanilla JS with no server side, so these are static
    assertions on the page, the pattern used throughout this file."""

    @staticmethod
    def _page() -> str:
        from hudka.ui.server import HERE

        return (HERE / "index.html").read_text(encoding="utf-8")

    def test_lanes_are_packed_so_overlapping_effects_stack(self):
        page = self._page()
        assert "function packLanes(" in page
        assert "tl-lane" in page and "fx ${i + 1}" in page

    def test_clips_can_be_dragged_and_beds_trimmed(self):
        page = self._page()
        assert "function dragClip(" in page and "function trimBed(" in page
        assert "tl-grip" in page

    def test_dragging_snaps_to_cuts_speech_edges_and_the_playhead(self):
        page = self._page()
        assert "function snap(" in page
        assert "proj.speech_ranges" in page and "currentTime" in page
        assert "ev.altKey" in page, "Alt should allow free placement"

    def test_lane_labels_sit_outside_the_scrolling_tracks(self):
        """Overlaid labels hid any clip starting at t=0 - where a music bed starts."""
        page = self._page()
        assert "tlnames" in page
        assert ".tl-label {" not in page

    def test_selecting_a_clip_selects_its_card(self):
        page = self._page()
        assert "function selectCue(" in page
        assert "card.dataset.id = cue.id" in page

    def test_a_project_has_a_url(self):
        """Refreshing mid-edit used to drop you back to the library."""
        page = self._page()
        assert "function routeFromHash(" in page
        assert "hashchange" in page
        assert "#/p/" in page


class TestFullWidthLayout:
    @staticmethod
    def _page() -> str:
        from hudka.ui.server import HERE

        return (HERE / "index.html").read_text(encoding="utf-8")

    def test_the_app_uses_the_whole_window(self):
        """A 1500px cap wasted most of a wide monitor."""
        page = self._page()
        assert "max-width:1500px" not in page

    def test_the_timeline_is_a_full_width_dock(self):
        page = self._page()
        assert ".tldock" in page and "position:sticky; bottom:0" in page

    def test_the_dock_publishes_its_height_for_the_layout(self):
        """The sticky preview panel has to stop where the dock starts, and the dock's
        height changes with lane count, collapse state and window size."""
        page = self._page()
        assert "--dock-h" in page
        assert "function reserveDockRoom(" in page

    def test_the_save_button_stays_visible_when_dirty(self):
        """.ghost is defined after .primary, so keeping both left dark text on a
        transparent background - an invisible button, exactly when it matters."""
        page = self._page()
        assert "btn.classList.remove('ghost')" in page
        assert "btn.classList.add('ghost')" in page

    def test_the_video_reserves_its_shape_before_metadata_loads(self):
        page = self._page()
        assert "aspect-ratio:16/9" in page
        assert "player.style.aspectRatio" in page
