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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import balance, engines, mix, presets
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
) -> dict[str, Path]:
    """Generate one WAV per cue, reusing cached stems where nothing changed."""
    opted = set(opted_in)
    say = progress or (lambda _: None)

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

            was_cached = dest.exists()
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

    # Stale stems from earlier renders would otherwise trip the provenance check, and
    # would quietly bloat the project directory.
    keep = {p.resolve() for p in stems.values()}
    for old in cache_dir.rglob("*.wav"):
        if old.resolve() not in keep:
            old.unlink()

    say(f"stems: {generated} generated, {cached} reused")
    return stems


def render(
    sheet: CueSheet,
    out_dir: Path,
    *,
    allow_noncommercial: bool = False,
    opted_in: Iterable[str] = (),
    device: str | None = None,
    progress: Progress | None = None,
) -> RenderResult:
    """Full render: generate, place, duck, master, mux, and write the licence report."""
    say = progress or (lambda _: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    video = Path(sheet.video.path)
    total = sheet.video.duration
    preset = presets.get(sheet.preset)

    ledger = Ledger(video=str(video))
    stems = generate_stems(
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

    say("muxing")
    final = mix.mux(video, mastered, out_dir / "final.mp4")

    ledger.verify(out_dir / "stems")
    prov_json, prov_md = ledger.save(out_dir)
    raw.unlink(missing_ok=True)

    return RenderResult(
        final_video=final, mix_wav=mastered, provenance=prov_json, licence_report=prov_md,
        measured_lufs=lufs, measured_peak_db=peak, stems=stems, balance=report,
        cached_count=sum(1 for r in ledger.records if r.cached),
        generated_count=sum(1 for r in ledger.records if not r.cached),
    )


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
    assert proc.stdin and proc.stdout
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

    stderr = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(_worker_error(engine_id, proc.returncode, stderr))

    missing = [c["id"] for c in cues if not Path(c["dest"]).exists()]
    if missing:
        raise RuntimeError(
            f"{engine_id} finished without producing: {', '.join(missing)}\n"
            f"{_tail(stderr)}"
        )


def _worker_error(engine_id: str, code: int, stderr: str) -> str:
    """Explain a worker failure, including the crash case that prints no traceback."""
    if "Traceback" not in stderr:
        return (
            f"{engine_id} crashed while generating (exit code {code}).\n\n"
            "The model process died without raising, which usually means too little VRAM "
            "for this model.\n"
            "Try a smaller engine: stable-audio-3-small-sfx for effects, "
            "stable-audio-3-small-music for beds.\nBoth run comfortably in 12GB.\n\n"
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
