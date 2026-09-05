"""Stage 2 + orchestration: cue sheet in, finished video out.

Two things drive the structure here:

* **Engine staging.** 12GB of VRAM cannot hold every model at once, so cues are grouped
  by engine, each model loads once, generates everything it owns, then unloads before the
  next one starts.
* **Content-hash caching.** A stem is keyed on (engine, prompt, duration, seed), so
  editing one cue and re-rendering regenerates that cue only. This is what makes
  iterating in the audition UI bearable rather than a full re-render each time.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import balance, engines, mix, presets, qa
from .audio import write_wav
from .engines.base import GenerateRequest, require_usable
from .provenance import Ledger
from .schema import BedCue, CueSheet, SfxCue

Progress = Callable[[str], None]


@dataclass
class RenderResult:
    final_video: Path
    mix_wav: Path
    provenance: Path
    licence_report: Path
    measured_lufs: float
    measured_peak_db: float
    stems: dict[str, Path]
    cached_count: int
    generated_count: int
    #: Bus levels relative to the source audio. None for silent footage, where there is
    #: no reference to balance against.
    balance: "balance.Balance | None" = None
    #: Placeholder tones, no model. Named and reported differently at every layer.
    is_preview: bool = False
    #: Content checks on every stem and on the finished mix. Level compliance alone once
    #: let pure noise through with a green tick; this is what says whether it is sound.
    quality: "qa.RenderQuality | None" = None
    verdict: str = "ok"


def _kind_of(cue: SfxCue | BedCue, sheet: CueSheet) -> str:
    if any(cue.id == m.id for m in sheet.music):
        return "music"
    if any(cue.id == a.id for a in sheet.ambience):
        return "ambience"
    return "sfx"


def generate_stems(
    sheet: CueSheet,
    out_dir: Path,
    ledger: Ledger,
    *,
    allow_noncommercial: bool = False,
    opted_in: Iterable[str] = (),
    device: str | None = None,
    progress: Progress | None = None,
) -> tuple[dict[str, Path], list[qa.StemQuality]]:
    """Generate one WAV per cue, reusing cached stems where nothing changed.

    Every stem is measured on the way out - cached ones included. A cached file is not
    proof of a good file: thirteen saturated stems once matched their cache keys exactly
    and would have been reused verbatim, because the fix that stopped the saturation
    lived inside the engine rather than in the key. A cached stem that fails a blocking
    check is deleted and regenerated with the same seed, which keeps provenance truthful.
    """
    opted = set(opted_in)
    say = progress or (lambda _: None)
    qualities: list[qa.StemQuality] = []

    cache_dir = out_dir / "stems"
    cache_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(sheet.video.path)

    by_engine: dict[str, list[SfxCue | BedCue]] = defaultdict(list)
    for cue in sheet.all_cues():
        by_engine[cue.engine].append(cue)

    stems: dict[str, Path] = {}
    cached = generated = 0

    # Gate every engine up front, so a licence problem or a missing dependency surfaces
    # before any weights load rather than part-way through a long render.
    #
    # Two passes, and the order matters: licences are checked across *every* engine before
    # any availability check runs. Otherwise a missing install on one engine masks a
    # licence violation on another, and the licence is the thing that must never slip
    # through - a missing package is merely inconvenient.
    built = {engine_id: engines.build(engine_id, device) for engine_id in by_engine}

    for engine in built.values():
        require_usable(engine, allow_noncommercial=allow_noncommercial, opted_in=opted)

    for engine in built.values():
        preflight = getattr(engine, "preflight", None)
        if callable(preflight):
            preflight()

    for engine_id, cues in by_engine.items():
        engine = built[engine_id]
        say(f"{engine_id}: {len(cues)} cue(s)")
        pending: list[dict] = []

        for cue in cues:
            kind = _kind_of(cue, sheet)
            duration = cue.duration if isinstance(cue, BedCue) else cue.duration
            window = ((cue.start, cue.end) if isinstance(cue, BedCue)
                      else (cue.at, cue.at + cue.duration))

            # Only non-default generation options enter `extra`, so a sheet that sets
            # none of them keeps the cache keys it already had.
            extra = {}
            if cue.steps is not None:
                extra["steps"] = cue.steps
            if cue.cfg_scale is not None:
                extra["cfg_scale"] = cue.cfg_scale
            if cue.negative_prompt:
                extra["negative_prompt"] = cue.negative_prompt

            req = GenerateRequest(
                prompt=cue.prompt,
                duration=duration,
                seed=cue.seed,
                video=video_path if engine_id == "hunyuan-foley" else None,
                window=window,
                extra=extra,
            )
            key = req.cache_key(engine_id)
            dest = cache_dir / kind / f"{cue.id}_{key}.wav"

            wanted = (cue.duration if isinstance(cue, BedCue) and not cue.loop else None)
            was_cached = dest.exists()
            if was_cached:
                check = qa.measure_stem(dest, cue.id, kind, wanted_s=wanted)
                if check.problems():
                    # Regenerating with the same seed is the one honest auto-retake: the
                    # request is identical, only the engine has moved on.
                    say(f"  cached stem for {cue.id} fails checks "
                        f"({check.problems()[0].split(': ', 1)[-1]}); regenerating")
                    dest.unlink()
                    was_cached = False
                else:
                    qualities.append(check)
            if was_cached:
                cached += 1
                say(f"  cached  {cue.id}  {cue.prompt[:52]}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                pending.append({
                    "id": cue.id, "prompt": cue.prompt, "duration": duration,
                    "seed": cue.seed, "dest": str(dest),
                    "video": str(video_path) if engine_id == "hunyuan-foley" else None,
                    "window": list(window),
                    "extra": extra,
                })

            stems[cue.id] = dest
            ledger.add(
                cue_id=cue.id, kind=kind, file=dest, engine_id=engine_id,
                licence=engine.licence, prompt=cue.prompt, seed=cue.seed,
                duration=duration, cached=was_cached,
            )

        if pending:
            _run_worker(engine_id, pending, device, say)
            generated += len(pending)
            for item in pending:
                cue = next(c for c in cues if c.id == item["id"])
                wanted = (cue.duration if isinstance(cue, BedCue) and not cue.loop else None)
                qualities.append(qa.measure_stem(Path(item["dest"]), cue.id,
                                                 _kind_of(cue, sheet), wanted_s=wanted))

    # Fresh stems that fail a blocking check stop the render here, before any mixing:
    # no final.mp4 is made from material that is silence or distortion.
    failed = [q for q in qualities if q.problems()]
    if failed:
        raise qa.QualityError([p for q in failed for p in q.problems()],
                              [q.cue_id for q in failed])

    # Stale stems from earlier renders would otherwise trip the provenance check, and
    # would quietly bloat the project directory.
    keep = {p.resolve() for p in stems.values()}
    for old in cache_dir.rglob("*.wav"):
        if old.resolve() not in keep:
            old.unlink()

    say(f"stems: {generated} generated, {cached} reused")
    return stems, qualities


#: Everything a preview writes lives under this subdirectory of the project.
PREVIEW_DIR = "preview"


def render(
    sheet: CueSheet,
    out_dir: Path,
    *,
    allow_noncommercial: bool = False,
    opted_in: Iterable[str] = (),
    device: str | None = None,
    progress: Progress | None = None,
    preview: bool = False,
) -> RenderResult:
    """Full render: generate, place, duck, master, mux, and write the licence report.

    `preview` swaps every engine for the placeholder and confines all output to
    `out_dir/preview/`. The confinement is not cosmetic: stem cleanup in
    `generate_stems` deletes any WAV the current run did not produce, and preview stems
    carry different cache keys, so a preview run over the real stem tree would destroy a
    finished render's cached audio.
    """
    say = progress or (lambda _: None)
    out_dir = Path(out_dir)

    if preview:
        sheet = sheet.model_copy(deep=True)
        for cue in sheet.all_cues():
            cue.engine = "silence"
        out_dir = out_dir / PREVIEW_DIR
        say("PREVIEW: placeholder tones, no model - not real audio")

    out_dir.mkdir(parents=True, exist_ok=True)

    video = Path(sheet.video.path)
    total = sheet.video.duration
    preset = presets.get(sheet.preset)

    # A stale report behind a fresh failure would show the last render's green state.
    (out_dir / "render_report.json").unlink(missing_ok=True)

    ledger = Ledger(video=str(video), preview=preview)
    stems, qualities = generate_stems(
        sheet, out_dir, ledger,
        allow_noncommercial=allow_noncommercial, opted_in=opted_in,
        device=device, progress=progress,
    )

    buses = out_dir / "buses"
    bus_paths: dict[str, Path | None] = {"music": None, "sfx": None, "ambience": None}

    # The dialogue must be extracted and measured BEFORE the generated buses are placed:
    # its loudness is the reference everything else is positioned against. Placing first
    # and measuring after leaves the anchor at zero and quietly undoes the whole point.
    dialogue = None
    anchor_db = 0.0
    key_ref_lufs: float | None = None

    if sheet.keep_original_audio and sheet.video.has_audio:
        dialogue = mix.extract_original_audio(video, buses / "original.wav", total)
        if dialogue:
            measured = mix.measured_lufs(dialogue)
            if measured is not None:
                key_ref_lufs = measured + sheet.original_gain_db
                anchor_db = float(np.clip(
                    key_ref_lufs - presets.NOMINAL_DIALOGUE_LUFS, -6.0, 6.0
                ))
                say(f"dialogue at {key_ref_lufs:.1f} LUFS, anchoring buses {anchor_db:+.1f} dB")
            else:
                say("could not measure the source loudness; buses left unanchored")

    say("placing cues")
    # FLOAT buses: normalised, gained and panned cues can pass 0 dBFS before the limiter,
    # and a fixed-point bus would clip them here rather than in the mixdown.
    if sheet.music:
        bus_paths["music"] = write_wav(
            buses / "music.wav",
            mix.place_beds(sheet.music, stems, total,
                           bus_offset_db=anchor_db + sheet.music_bus_db,
                           normalize=sheet.normalizes_stems),
            subtype="FLOAT",
        )
    if sheet.ambience:
        bus_paths["ambience"] = write_wav(
            buses / "ambience.wav",
            mix.place_beds(sheet.ambience, stems, total,
                           bus_offset_db=anchor_db + sheet.ambience_bus_db,
                           normalize=sheet.normalizes_stems),
            subtype="FLOAT",
        )
    if sheet.sfx:
        bus_paths["sfx"] = write_wav(
            buses / "sfx.wav",
            mix.place_sfx(sheet.sfx, stems, total,
                          bus_offset_db=anchor_db + sheet.sfx_bus_db,
                          normalize=sheet.normalizes_stems),
            subtype="FLOAT",
        )

    report = balance.measure(out_dir)
    if report:
        for problem in report.problems():
            say(f"  balance: {problem}")

    say("mixing and ducking")
    raw = mix.mixdown(
        music=bus_paths["music"], sfx=bus_paths["sfx"], ambience=bus_paths["ambience"],
        dialogue=dialogue, dest=out_dir / "mix_raw.wav",
        duck_depth_db=(sheet.duck_depth_db if sheet.duck_depth_db is not None
                       else preset.duck_depth_db),
        original_gain_db=sheet.original_gain_db,
        key_ref_lufs=key_ref_lufs,
    )

    say(f"mastering to {sheet.target_lufs} LUFS / {sheet.true_peak_db} dBTP")
    mastered = mix.normalize(raw, out_dir / "mix.wav", sheet.target_lufs, sheet.true_peak_db)
    lufs, peak = mix.integrated_lufs(mastered)

    mix_samples, _ = mix.read_wav(mastered)
    quality = qa.RenderQuality(
        stems=qualities,
        mix_lufs=lufs,
        mix_flatness=qa.spectral_flatness(mix_samples),
        sfx_events=report.sfx_events if report else None,
        sfx_cues=len([c for c in sheet.sfx if not c.muted]),
        balance_problems=report.problems() if report else [],
    )
    if quality.problems():
        # Mix-level blocks: a silent or unmeasurable master, or effects that never landed.
        raise qa.QualityError(quality.problems(), [])
    for line in quality.warnings():
        say(f"  check: {line}")

    say("muxing")
    final = mix.mux(video, mastered, out_dir / ("preview.mp4" if preview else "final.mp4"))

    ledger.verify(out_dir / "stems")
    prov_json, prov_md = ledger.save(out_dir)
    raw.unlink(missing_ok=True)

    _write_report(out_dir, quality, lufs, peak, report, preview)

    return RenderResult(
        final_video=final, mix_wav=mastered, provenance=prov_json, licence_report=prov_md,
        measured_lufs=lufs, measured_peak_db=peak, stems=stems, balance=report,
        cached_count=sum(1 for r in ledger.records if r.cached),
        generated_count=sum(1 for r in ledger.records if not r.cached),
        is_preview=preview, quality=quality, verdict=quality.verdict,
    )


def _write_report(out_dir: Path, quality: qa.RenderQuality, lufs: float, peak: float,
                  report: "balance.Balance | None", preview: bool) -> Path:
    """The render's verdict, on disk, so the GUI can badge a project truthfully."""
    from dataclasses import asdict

    path = out_dir / "render_report.json"
    path.write_text(json.dumps({
        "verdict": quality.verdict,
        "preview": preview,
        "problems": quality.problems(),
        "warnings": quality.warnings(),
        "lufs": None if not np.isfinite(lufs) else round(float(lufs), 2),
        "peak_dbtp": None if not np.isfinite(peak) else round(float(peak), 2),
        "balance": {
            "music_offset_db": report.music_offset_db, "sfx_offset_db": report.sfx_offset_db,
            "sfx_events": report.sfx_events, "sfx_per_minute": round(report.sfx_per_minute, 1),
        } if report else None,
        "stems": [asdict(q) for q in quality.stems],
    }, indent=2) + "\n", encoding="utf-8")
    return path


def _run_worker(engine_id: str, cues: list[dict], device: str | None,
                say: Progress) -> None:
    """Generate this engine's cues in a dedicated process.

    Isolation is the point: a model that crashes the interpreter - which happens here when
    a second model is loaded after the first is released - must not take the caller down
    with it. In the GUI the caller is the web server, so an unisolated crash would kill
    the whole app mid-render.
    """
    payload = json.dumps({"engine": engine_id, "device": device, "cues": cues})
    by_id = {c["id"]: c for c in cues}

    proc = subprocess.Popen(
        [sys.executable, "-m", "hudka._worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdin and proc.stdout and proc.stderr

    # Drain stderr on its own thread while stdout is read for progress. Reading stdout to
    # EOF first and stderr afterwards is the textbook pipe deadlock, and it happened: each
    # cue's tqdm bar goes to stderr, a 7-cue render never filled the pipe buffer, and a
    # 36-cue render blocked the worker on a full pipe while the parent waited on stdout -
    # 3.4 GB resident, 2% GPU, 15 seconds of CPU across half an hour.
    captured: list[str] = []
    drain = threading.Thread(target=lambda: captured.append(proc.stderr.read()), daemon=True)
    drain.start()

    proc.stdin.write(payload)
    proc.stdin.close()

    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            cue_id = json.loads(line).get("done")
        except json.JSONDecodeError:
            continue
        if cue_id in by_id:
            say(f"  render  {cue_id}  {by_id[cue_id]['prompt'][:52]}")

    drain.join()
    stderr = "".join(captured)
    # Engines print `warning:` lines for things like saturated output. Those used to be
    # read only when the exit code was non-zero - which is to say, never for a warning.
    for line in stderr.splitlines():
        if line.lower().startswith("warning:"):
            say(f"  {line.strip()}")
    if proc.wait() != 0:
        raise RuntimeError(_worker_error(engine_id, proc.returncode, stderr))

    missing = [c["id"] for c in cues if not Path(c["dest"]).exists()]
    if missing:
        raise RuntimeError(
            f"{engine_id} finished without producing: {', '.join(missing)}\n"
            f"{_tail(stderr)}"
        )


#: What a silent worker death usually means, by exit status. Blaming VRAM for all of
#: them - as this used to - sent a user with weights on a USB drive to shrink models.
_EXIT_MEANINGS = {
    0xC0000006: ("STATUS_IN_PAGE_ERROR: a memory-mapped file could not be read. The model "
                 "weights are almost certainly on an external, USB or exFAT drive. Move "
                 "them to an internal SSD (set HUDKA_MODEL_DIR) and retry."),
    0xC0000005: ("access violation inside the model process - usually a PyTorch/CUDA "
                 "driver mismatch. Update the NVIDIA driver, or reinstall torch with "
                 "Setup.bat."),
    0xFFFFFFF7: ("killed (SIGKILL) - the machine ran out of system RAM while loading the "
                 "model. Close other applications, or use the small engines."),
    0xFFFFFFF5: ("segmentation fault inside the model process - usually a broken or "
                 "mismatched torch build. Reinstall torch with Setup.bat."),
}


def _explain_exit(code: int, stderr: str) -> str:
    """A cause a person can act on, from the exit status and whatever stderr holds."""
    if "CUDA out of memory" in stderr or "OutOfMemoryError" in stderr:
        return ("CUDA ran out of VRAM. Use the small engines - stable-audio-3-small-sfx "
                "for effects, stable-audio-3-small-music for beds - which peak near 2 GB.")
    normalised = code & 0xFFFFFFFF
    if normalised in _EXIT_MEANINGS:
        return _EXIT_MEANINGS[normalised]
    if code in (137, 9):
        return _EXIT_MEANINGS[0xFFFFFFF7]
    if code in (139, 11):
        return _EXIT_MEANINGS[0xFFFFFFF5]
    return ("the model process died without raising. Check `hudka doctor`, and that the "
            "model weights are on an internal drive.")


def _worker_error(engine_id: str, code: int, stderr: str) -> str:
    """Explain a worker failure, including the crash case that prints no traceback."""
    if "Traceback" not in stderr:
        return (
            f"{engine_id} crashed while generating (exit code {code}).\n\n"
            f"{_explain_exit(code, stderr)}\n\n"
            f"{_tail(stderr)}"
        )
    return f"{engine_id} failed (exit code {code}):\n{_tail(stderr)}"


def _tail(text: str, lines: int = 12) -> str:
    kept = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(kept[-lines:])


#: Seeds for alternative takes are derived from the cue's own seed by a fixed stride, so
#: the same cue always offers the same set of alternatives and any one of them can be
#: adopted permanently just by recording its seed.
TAKE_SEED_STRIDE = 7919


def take_seeds(base_seed: int, count: int) -> list[int]:
    return [(base_seed + (i + 1) * TAKE_SEED_STRIDE) % 2_147_483_647 for i in range(count)]


def generate_only(
    sheet: CueSheet,
    out_dir: Path,
    *,
    allow_noncommercial: bool = False,
    opted_in: Iterable[str] = (),
    device: str | None = None,
    progress: Progress | None = None,
) -> dict:
    """Generate every cue's audio and stop. No placement, no mix, no master, no mux.

    Generation is the slow part that actually answers "does this sound right". Placing,
    mixing, mastering to a loudness target and re-muxing a 4K video are not, and on a
    long clip they cost as much again as the sounds themselves - paid every time, before
    you are allowed to hear anything.

    The browser plays these stems directly and applies gain, pan, mute and solo as live
    gain nodes, so this is paid once per cue and every adjustment after it is free.

    No provenance.json is written: nothing is delivered here, and the ledger belongs to
    the render that produces a file you could hand to someone. The licence gate is
    unaffected - it lives inside generate_stems and runs before any weights load.
    """
    say = progress or (lambda _: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = Ledger(video=str(Path(sheet.video.path)))
    stems, qualities = generate_stems(
        sheet, out_dir, ledger,
        allow_noncommercial=allow_noncommercial, opted_in=opted_in,
        device=device, progress=progress,
    )

    warnings = [w for q in qualities for w in q.warnings()]
    for warning in warnings:
        say(f"  {warning}")
    say(f"{len(stems)} sound(s) ready to hear - no mix, no master, no video")
    return {
        "stems": sorted(stems),
        "generated": sum(1 for r in ledger.records if not r.cached),
        "cached": sum(1 for r in ledger.records if r.cached),
        "warnings": warnings,
    }


def generate_variations(
    sheet: CueSheet,
    out_dir: Path,
    cue_id: str,
    *,
    count: int = 4,
    device: str | None = None,
    progress: Progress | None = None,
) -> list[dict]:
    """Render several alternative takes of one cue, for the user to choose between.

    Re-rolling a seed and re-rendering the whole mix to hear the result is a slow way to
    ask "is there a better one?". Generating a handful at once and auditioning them makes
    that a listening decision instead of a guessing one.

    Takes live outside `stems/` deliberately: they are candidates, not part of the render,
    and the provenance check would rightly object to untracked audio in the stem tree.
    """
    say = progress or (lambda _: None)
    cue = next((c for c in sheet.all_cues() if c.id == cue_id), None)
    if cue is None:
        raise ValueError(f"no cue {cue_id!r} in this sheet")

    engine = engines.build(cue.engine, device)
    require_usable(engine, allow_noncommercial=False, opted_in=set())
    preflight = getattr(engine, "preflight", None)
    if callable(preflight):
        preflight()

    takes_dir = out_dir / "takes" / cue_id
    takes_dir.mkdir(parents=True, exist_ok=True)

    extra = {}
    if cue.steps is not None:
        extra["steps"] = cue.steps
    if cue.cfg_scale is not None:
        extra["cfg_scale"] = cue.cfg_scale
    if cue.negative_prompt:
        extra["negative_prompt"] = cue.negative_prompt

    duration = cue.duration
    pending, results = [], []
    for seed in take_seeds(cue.seed, count):
        dest = takes_dir / f"{seed}.wav"
        results.append({"seed": seed, "file": dest.name, "cached": dest.exists()})
        if not dest.exists():
            pending.append({
                "id": f"{cue_id}@{seed}", "prompt": cue.prompt, "duration": duration,
                "seed": seed, "dest": str(dest), "video": None,
                "window": [0.0, duration], "extra": extra,
            })

    if pending:
        say(f"{cue.engine}: {len(pending)} variation(s) of {cue_id}")
        _run_worker(cue.engine, pending, device, say)

    say(f"{count} take(s) ready for {cue_id}")
    return results
