"""The GUI server.

Covers the whole workflow, so nothing needs a terminal: import a video, analyse it,
build and edit a cue sheet, render, audition every stem, and download the result.

Local-only by design. It binds to 127.0.0.1 and works inside a single workspace
directory; paths coming from the page are resolved and checked against it, because this
process has the user's filesystem underneath it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from .. import analyze as analyze_mod
from .. import design, engines, presets, render as render_mod
from ..engines.base import LicenceError
from ..schema import CueSheet, VideoInfo
from .jobs import JobRunner

HERE = Path(__file__).parent
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv"}


class ImportPath(BaseModel):
    path: str


class RenderOptions(BaseModel):
    opted_in: list[str] = []
    allow_noncommercial: bool = False
    #: Swap every engine for the silence stub, to exercise timing and mixing before the
    #: real models are installed. Does not touch the saved cue sheet.
    preview: bool = False


class VariationOptions(BaseModel):
    cue_id: str
    count: int = 4


class ScaffoldOptions(BaseModel):
    #: None means "let the designer choose from the video" - a concrete default here
    #: silently overrides that, which is how a narration-heavy clip ended up short-form.
    preset: str | None = None
    engine: str | None = None


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-._")
    return cleaned[:60] or "project"


def create_app(workspace: Path) -> FastAPI:
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Hudka", docs_url=None, redoc_url=None)
    runner = JobRunner()

    # ---------------------------------------------------------------- helpers

    def project_dir(name: str) -> Path:
        """Resolve a project by name, refusing anything outside the workspace."""
        target = (workspace / name).resolve()
        if target.parent != workspace or not target.is_dir():
            raise HTTPException(status_code=404, detail=f"no project {name!r}")
        return target

    def source_video(project: Path) -> Path | None:
        cues = project / "cues.json"
        if cues.exists():
            # A hand-edited or truncated sheet must not make the project unopenable -
            # that would leave a text editor as the only way to repair it.
            try:
                recorded = json.loads(cues.read_text(encoding="utf-8"))["video"]["path"]
            except (json.JSONDecodeError, KeyError, TypeError, OSError):
                recorded = None
            if recorded:
                path = Path(recorded)
                if path.exists():
                    return path
        found = sorted(p for p in (project / "source").glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
        return found[0] if found else None

    def read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def summarise(project: Path) -> dict:
        analysis = read_json(project / "analysis.json")
        cues = read_json(project / "cues.json")
        video = analysis.get("video") or cues.get("video") or {}
        source = source_video(project)
        active = runner.active_for(project.name)

        if (project / "final.mp4").exists():
            stage = "rendered"
        elif cues:
            stage = "designed"
        elif analysis:
            stage = "analysed"
        else:
            stage = "imported"

        preview_file = project / render_mod.PREVIEW_DIR / "preview.mp4"
        report = read_json(project / "render_report.json")
        return {
            "name": project.name,
            "stage": stage,
            "verdict": report.get("verdict") if report else None,
            "has_preview": preview_file.exists(),
            "preview_at": preview_file.stat().st_mtime if preview_file.exists() else None,
            "busy": bool(active),
            "job": active.as_dict() if active else None,
            "duration": video.get("duration", 0),
            "width": video.get("width", 0),
            "height": video.get("height", 0),
            "has_dialogue": video.get("has_dialogue", False),
            "shots": len(analysis.get("shots", [])),
            "cue_count": len(cues.get("sfx", [])) + len(cues.get("music", []))
                         + len(cues.get("ambience", [])),
            "source_name": source.name if source else None,
            "missing_source": source is None,
        }

    def stem_index(project: Path) -> dict[str, Path]:
        """Map cue id to its stem file. Names are `<cue id>_<content hash>.wav`."""
        return {wav.stem.rsplit("_", 1)[0]: wav for wav in (project / "stems").rglob("*.wav")}

    # ------------------------------------------------------------------ page

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (HERE / "index.html").read_text(encoding="utf-8")

    @app.get("/api/library")
    def api_library() -> JSONResponse:
        projects = sorted(
            (p for p in workspace.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        return JSONResponse({
            "workspace": str(workspace),
            "projects": [summarise(p) for p in projects],
            "presets": {
                name: {"description": pre.description, "guidance": pre.guidance,
                       "target_lufs": pre.target_lufs}
                for name, pre in presets.PRESETS.items()
            },
            "engines": {
                engine_id: {
                    "licence": lic.name, "commercial": lic.commercial,
                    "requires_optin": lic.requires_optin,
                    "territory_exclusions": list(lic.territory_exclusions),
                    "revenue_cap_usd": lic.revenue_cap_usd,
                    "training_data": lic.training_data, "notes": lic.notes,
                }
                for engine_id, lic in engines.LICENCE_TABLE.items()
            },
            "engine_available": _engine_availability(),
            "engine_auth": _engine_auth(),
            "jobs": runner.recent(),
        })

    # ---------------------------------------------------------------- import

    def _new_project(stem: str) -> Path:
        base = _slug(stem)
        target = workspace / base
        n = 2
        while target.exists():
            target = workspace / f"{base}-{n}"
            n += 1
        (target / "source").mkdir(parents=True)
        return target

    @app.post("/api/import/upload")
    async def api_import_upload(file: UploadFile = File(...)) -> JSONResponse:
        name = Path(file.filename or "video.mp4").name
        if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail=f"{Path(name).suffix or 'that file type'} is not a video "
                       f"({', '.join(sorted(VIDEO_SUFFIXES))})",
            )
        project = _new_project(Path(name).stem)
        dest = project / "source" / _slug(name)
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(file.file, out, length=1024 * 1024)
        finally:
            await file.close()
        return JSONResponse({"name": project.name})

    @app.post("/api/import/path")
    def api_import_path(payload: ImportPath) -> JSONResponse:
        """Reference a file already on disk, rather than copying gigabytes through the browser."""
        src = Path(payload.path.strip().strip('"')).expanduser()
        if not src.is_file():
            raise HTTPException(status_code=404, detail=f"no such file: {src}")
        if src.suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(status_code=415, detail=f"{src.suffix} is not a video file")

        project = _new_project(src.stem)
        # A pointer file, so large sources are read in place instead of duplicated.
        (project / "source" / "source.txt").write_text(str(src.resolve()), encoding="utf-8")
        (project / "analysis.json").unlink(missing_ok=True)
        _record_external_source(project, src)
        return JSONResponse({"name": project.name})

    def _record_external_source(project: Path, src: Path) -> None:
        (project / "external_source.json").write_text(
            json.dumps({"path": str(src.resolve())}, indent=2), encoding="utf-8"
        )

    def _resolve_source(project: Path) -> Path:
        external = project / "external_source.json"
        if external.exists():
            path = Path(read_json(external)["path"])
            if not path.exists():
                raise HTTPException(
                    status_code=410,
                    detail=f"the source video has moved or been deleted:\n{path}",
                )
            return path
        found = source_video(project)
        if found is None:
            raise HTTPException(status_code=404, detail="this project has no source video")
        return found

    @app.delete("/api/project/{name}")
    def api_delete(name: str) -> JSONResponse:
        target = project_dir(name)
        if runner.active_for(name):
            raise HTTPException(status_code=409, detail="a job is still running on this project")
        shutil.rmtree(target)
        return JSONResponse({"ok": True})

    # ----------------------------------------------------------------- stages

    @app.post("/api/project/{name}/analyze")
    def api_analyze(name: str) -> JSONResponse:
        project = project_dir(name)
        if runner.active_for(name):
            raise HTTPException(status_code=409, detail="already working on this project")
        video = _resolve_source(project)

        def work(say):
            say(f"analysing {video.name}")
            result = analyze_mod.analyze(video, project)
            say(f"{len(result.shots)} shot(s), {len(result.contact_sheets)} contact sheet(s)")
            if result.video.has_dialogue:
                say("speech detected - music will duck under it")
            return {"shots": len(result.shots)}

        return JSONResponse(runner.submit("analyze", name, work).as_dict())

    @app.post("/api/project/{name}/scaffold")
    def api_scaffold(name: str, opts: ScaffoldOptions) -> JSONResponse:
        """Mechanical starting cue sheet. See `hudka.design` for how anchors are chosen."""
        project = project_dir(name)
        analysis = read_json(project / "analysis.json")
        if not analysis:
            raise HTTPException(status_code=409, detail="analyse the video first")

        sheet = design.scaffold(analysis, preset=opts.preset, engine=opts.engine)
        sheet.save(project / "cues.json")
        return JSONResponse({
            "ok": True,
            "preset": sheet.preset,
            "sfx": len(sheet.sfx),
            "cues": len(sheet.all_cues()),
        })

    @app.post("/api/project/{name}/render")
    def api_render(name: str, opts: RenderOptions) -> JSONResponse:
        project = project_dir(name)
        if runner.active_for(name):
            raise HTTPException(status_code=409, detail="already working on this project")
        if not (project / "cues.json").exists():
            raise HTTPException(status_code=409, detail="there is no cue sheet to render")

        sheet = CueSheet.load(project / "cues.json")

        def work(say):
            try:
                result = render_mod.render(
                    sheet, project, allow_noncommercial=opts.allow_noncommercial,
                    opted_in=set(opts.opted_in), progress=say, preview=opts.preview,
                )
            except LicenceError as exc:
                # Surface the licence explanation verbatim; it tells the user what to do.
                raise RuntimeError(str(exc)) from exc

            off = abs(result.measured_lufs - sheet.target_lufs)
            say(f"{result.measured_lufs:.2f} LUFS, peak {result.measured_peak_db:.2f} dBTP"
                + ("" if off <= 0.5 else f"  ({off:.2f} LU off target)"))
            return {
                "lufs": round(result.measured_lufs, 2),
                "peak": round(result.measured_peak_db, 2),
                "on_target": off <= 0.5,
                "generated": result.generated_count,
                "cached": result.cached_count,
                "preview": result.is_preview,
                # Content verdict. "on_target" alone is not a verdict: a render of pure
                # noise once hit the loudness target exactly.
                "verdict": result.verdict,
                "warnings": result.quality.warnings() if result.quality else [],
            }

        return JSONResponse(runner.submit("render", name, work).as_dict())

    @app.post("/api/project/{name}/variations")
    def api_variations(name: str, opts: VariationOptions) -> JSONResponse:
        """Render alternative takes of one cue, so the choice can be made by ear."""
        project = project_dir(name)
        if runner.active_for(name):
            raise HTTPException(status_code=409, detail="already working on this project")
        if not (project / "cues.json").exists():
            raise HTTPException(status_code=409, detail="there is no cue sheet yet")

        sheet = CueSheet.load(project / "cues.json")
        count = max(2, min(opts.count, 8))

        def work(say):
            takes = render_mod.generate_variations(
                sheet, project, opts.cue_id, count=count, progress=say,
            )
            return {"cue": opts.cue_id, "takes": takes}

        return JSONResponse(runner.submit("variations", name, work).as_dict())

    @app.get("/media/{name}/take/{cue_id}/{seed}")
    def media_take(name: str, cue_id: str, seed: str) -> FileResponse:
        takes_dir = (project_dir(name) / "takes" / cue_id).resolve()
        target = (takes_dir / f"{seed}.wav").resolve()
        if not target.is_file() or takes_dir not in target.parents:
            raise HTTPException(status_code=404, detail="no such take")
        return FileResponse(target, media_type="audio/wav")

    @app.get("/api/job/{job_id}")
    def api_job(job_id: str) -> JSONResponse:
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return JSONResponse(job.as_dict())

    # ------------------------------------------------------------------ cues

    @app.get("/api/project/{name}")
    def api_project(name: str) -> JSONResponse:
        project = project_dir(name)
        analysis = read_json(project / "analysis.json")
        active = runner.active_for(name)

        # Validate on the way out rather than serving the raw file. A sheet written before
        # a field existed simply omits it, and the page would then bind controls to
        # undefined - so every cue sheet is filled out with current defaults first.
        cues = None
        cues_path = project / "cues.json"
        if cues_path.exists():
            try:
                cues = CueSheet.load(cues_path).model_dump(mode="json")
            except Exception:
                # A hand-edited sheet that no longer validates still has to be openable,
                # or the only way to fix it is a text editor.
                cues = read_json(cues_path)

        return JSONResponse({
            **summarise(project),
            "cues": cues,
            "shots": analysis.get("shots", []),
            "speech_ranges": analysis.get("speech_ranges", []),
            "contact_sheets": analysis.get("contact_sheets", []),
            "suggested_preset": (
                design.suggest_preset(VideoInfo.model_validate(analysis["video"]), analysis)
                if analysis else None
            ),
            "speech_coverage": round(
                design.speech_coverage(analysis, analysis.get("video", {}).get("duration", 0)), 3
            ) if analysis else 0.0,
            "takes": {
                d.name: sorted(int(f.stem) for f in d.glob("*.wav") if f.stem.isdigit())
                for d in (project / "takes").iterdir()
            } if (project / "takes").is_dir() else {},
            "stems": sorted(stem_index(project)),
            "report": read_json(project / "render_report.json") or None,
            "provenance": read_json(project / "provenance.json").get("records", []),
            "job": active.as_dict() if active else None,
        })

    @app.put("/api/project/{name}/cues")
    def api_save_cues(name: str, payload: dict) -> JSONResponse:
        project = project_dir(name)
        try:
            sheet = CueSheet.model_validate(payload)
        except ValidationError as exc:
            # Report per cue. "3 validation errors for CueSheet" tells the user nothing
            # about which cue is wrong or what to do, so the page can neither explain the
            # problem nor point at it.
            raise HTTPException(status_code=422, detail=_cue_errors(exc, payload)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=[{"cue": None, "message": str(exc).splitlines()[0]}],
            ) from exc
        sheet.save(project / "cues.json")
        return JSONResponse({"ok": True})

    # ----------------------------------------------------------------- media

    @app.get("/media/{name}/video")
    def media_video(name: str) -> FileResponse:
        final = project_dir(name) / "final.mp4"
        if not final.exists():
            raise HTTPException(status_code=404, detail="not rendered yet")
        return FileResponse(final, media_type="video/mp4")

    @app.get("/media/{name}/preview")
    def media_preview(name: str) -> FileResponse:
        preview = project_dir(name) / render_mod.PREVIEW_DIR / "preview.mp4"
        if not preview.exists():
            raise HTTPException(status_code=404, detail="no preview rendered")
        return FileResponse(preview, media_type="video/mp4")

    @app.get("/media/{name}/source")
    def media_source(name: str) -> FileResponse:
        return FileResponse(_resolve_source(project_dir(name)), media_type="video/mp4")

    @app.get("/media/{name}/stem/{cue_id}")
    def media_stem(name: str, cue_id: str) -> FileResponse:
        stem = stem_index(project_dir(name)).get(cue_id)
        if stem is None:
            raise HTTPException(status_code=404, detail=f"no stem for {cue_id}")
        return FileResponse(stem, media_type="audio/wav")

    @app.get("/media/{name}/contact/{sheet_name}")
    def media_contact(name: str, sheet_name: str) -> FileResponse:
        contact_dir = (project_dir(name) / "contact").resolve()
        target = (contact_dir / sheet_name).resolve()
        # Keep a crafted name from escaping the project directory.
        if not target.is_file() or contact_dir not in target.parents:
            raise HTTPException(status_code=404, detail="no such contact sheet")
        return FileResponse(target, media_type="image/jpeg")

    @app.get("/download/{name}/video")
    def download_video(name: str) -> FileResponse:
        project = project_dir(name)
        final = project / "final.mp4"
        if not final.exists():
            if (project / render_mod.PREVIEW_DIR / "preview.mp4").exists():
                # A preview is placeholder tones. Handing it out as the deliverable is
                # exactly how one was mistaken for a broken product.
                raise HTTPException(
                    status_code=409,
                    detail="only a preview exists - untick preview and render for real",
                )
            raise HTTPException(status_code=404, detail="not rendered yet")
        return FileResponse(final, media_type="video/mp4",
                            filename=f"{name}-with-sound.mp4")

    @app.get("/download/{name}/licence")
    def download_licence(name: str) -> FileResponse:
        report = project_dir(name) / "LICENSE-REPORT.md"
        if not report.exists():
            raise HTTPException(status_code=404, detail="not rendered yet")
        return FileResponse(report, media_type="text/markdown",
                            filename=f"{name}-LICENSE-REPORT.md")

    return app


#: Pydantic error types translated into something a person can act on.
_MESSAGES = {
    "string_too_short": "needs a description of the sound",
    "missing": "is required",
    "greater_than_equal": "is too low",
    "less_than_equal": "is too high",
}


def _cue_errors(exc: ValidationError, payload: dict) -> list[dict]:
    """Turn a ValidationError into one readable entry per offending cue.

    Each entry carries the cue id so the page can highlight the exact card, rather than
    showing a count the user has to go hunting through.
    """
    out: list[dict] = []
    for err in exc.errors():
        loc = err["loc"]
        kind = cue_id = None
        field = ".".join(str(part) for part in loc)

        if len(loc) >= 3 and isinstance(loc[1], int) and isinstance(loc[0], str):
            kind, index, field = loc[0], loc[1], str(loc[2])
            items = payload.get(kind) or []
            if index < len(items):
                cue_id = items[index].get("id") or f"{kind}[{index}]"

        detail = _MESSAGES.get(err["type"])
        if detail is None:
            # Model-level checks (overlapping beds, cue past the end) carry their own text.
            detail = err.get("msg", "is invalid").replace("Value error, ", "")
            if cue_id is None:
                out.append({"cue": None, "field": None, "message": detail})
                continue

        out.append({"cue": cue_id, "kind": kind, "field": field, "message": detail})

    # Collapse duplicates so one bad field is not reported several times over.
    seen, unique = set(), []
    for item in out:
        key = (item["cue"], item.get("field"), item["message"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


#: Cached so the page does not reach out to Hugging Face on every load. Authorisation
#: changes at most once, when the user accepts a licence and logs in - polling it per
#: page view makes an otherwise fully local tool chatter over the network, and stalls the
#: library screen when the machine is offline.
_AUTH_CACHE: dict | None = None


def _engine_auth(refresh: bool = False) -> dict:
    """Whether the gated Stable Audio weights are reachable from this machine.

    Resolved once per process so the page can say what is needed before a render starts,
    rather than after it fails. Skipped entirely when the weights are already cached
    locally or HF_HUB_OFFLINE is set - in both cases there is nothing to ask about, and
    asking would be the only network call this application makes.
    """
    global _AUTH_CACHE
    if _AUTH_CACHE is not None and not refresh:
        return _AUTH_CACHE

    try:
        import huggingface_hub as hub

        from ..engines.stable_audio3 import StableAudio3Engine
    except ImportError:
        _AUTH_CACHE = {"needed": False, "logged_in": False, "authorised": None}
        return _AUTH_CACHE

    engine = StableAudio3Engine("stable-audio-3-small-sfx")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    cached = _weights_present()

    if offline or cached:
        # Nothing to authorise: the weights are already here, or we are told not to look.
        _AUTH_CACHE = {
            "needed": False, "logged_in": bool(hub.get_token()),
            "authorised": True, "repo": engine.repo_id, "offline": offline,
        }
        return _AUTH_CACHE

    _AUTH_CACHE = {
        "needed": True,
        "logged_in": bool(hub.get_token()),
        "authorised": engine.is_authorised(),
        "repo": engine.repo_id,
        "offline": False,
    }
    return _AUTH_CACHE


def _weights_present() -> bool:
    """Whether model weights are already on disk, making any auth check moot."""
    cache = Path(os.environ.get("HF_HUB_CACHE", "")) if os.environ.get("HF_HUB_CACHE") else None
    if cache is None or not cache.is_dir():
        return False
    return any(cache.glob("models--stabilityai--stable-audio-3-*/snapshots/*/*.safetensors"))


def _engine_availability() -> dict[str, bool]:
    """Which engines can actually run, so the page can warn before a render fails."""
    available: dict[str, bool] = {"silence": True}
    for module, ids in (
        ("stable_audio_3", ("stable-audio-3-medium", "stable-audio-3-small-sfx",
                            "stable-audio-3-small-music")),
        ("acestep", ("acestep-1.5",)),
    ):
        try:
            __import__(module)
            present = True
        except ImportError:
            present = False
        for engine_id in ids:
            available[engine_id] = present
    available["hunyuan-foley"] = (engines.model_dir() / "hunyuan-foley").exists()
    return available
