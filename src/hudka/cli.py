"""Command line interface.

Most people should use the app instead - double-click `Hudka.bat`, or:

    hudka gui                        the whole workflow in a browser

    hudka analyze  video.mp4         analyse and build contact sheets
    hudka scaffold out/video         heuristic starting cue sheet
    hudka render   out/video         generate, mix, master, mux
    hudka licences                   what each engine permits
    hudka doctor                     check ffmpeg and engines
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import analyze as analyze_mod
from . import design, engines, presets, qa, render as render_mod
from .schema import CueSheet

app = typer.Typer(add_completion=False, help=__doc__, no_args_is_help=True)
console = Console()


def _project_dir(video: Path, out: Path | None) -> Path:
    return Path(out) if out else Path("out") / Path(video).stem


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Video file to analyse"),
    out: Path = typer.Option(None, "--out", "-o", help="Project directory"),
) -> None:
    """Probe the video, detect shots, and build timecoded contact sheets."""
    project = _project_dir(video, out)
    console.print(f"[dim]analysing[/] {video}")

    try:
        result = analyze_mod.analyze(video, project)
    except analyze_mod.AnalysisError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    info = result.video
    console.print(
        f"[green]OK[/] {info.duration:.2f}s  {info.width}x{info.height}  {info.fps:.2f}fps  "
        f"{len(result.shots)} shot(s)"
    )
    if info.has_dialogue:
        speech = sum(b - a for a, b in result.speech_ranges)
        console.print(f"     speech detected across {speech:.1f}s - music will duck under it")
    elif info.has_audio:
        console.print("     has audio, but little sustained speech")
    else:
        console.print("     silent source")

    console.print(f"     contact sheets: {project / 'contact'} ({len(result.contact_sheets)})")
    console.print(f"     analysis:       {project / 'analysis.json'}")
    console.print("\nNext: write cues.json from the contact sheets, "
                  f"or run [bold]hudka scaffold {project}[/] for a starting point.")


@app.command()
def scaffold(
    project: Path = typer.Argument(..., help="Project directory from `hudka analyze`"),
    preset: str = typer.Option(None, "--preset", "-p", help="Default: chosen from the video"),
    engine: str = typer.Option(None, "--engine", help="Override the engine for every cue"),
) -> None:
    """Write a heuristic cue sheet: a bed, plus accents on cuts and activity peaks.

    A baseline to react to, not a finished design - it knows where things happen but not
    what they are. Reword the prompts, then render.
    """
    project = Path(project)
    analysis_path = project / "analysis.json"
    if not analysis_path.exists():
        console.print(f"[red]no analysis.json in {project}; run `hudka analyze` first[/]")
        raise typer.Exit(1)

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    sheet = design.scaffold(analysis, preset=preset, engine=engine)
    sheet.save(project / "cues.json")

    pre = presets.get(sheet.preset)
    coverage = design.speech_coverage(analysis, sheet.video.duration)
    console.print(f"[green]wrote[/] {project / 'cues.json'}")
    console.print(f"     preset [bold]{sheet.preset}[/]"
                  + ("" if preset else " (chosen from the video)")
                  + f", {len(sheet.music)} bed, {len(sheet.sfx)} effect(s)")
    if coverage > 0.45:
        console.print(f"     speech covers {coverage:.0%} - cues kept sparse and quiet")
    console.print(f"\n[dim]{pre.guidance}[/]")


@app.command()
def render(
    project: Path = typer.Argument(..., help="Project directory containing cues.json"),
    engine: list[str] = typer.Option(
        [], "--engine",
        help="Opt in to a restricted engine, e.g. --engine hunyuan-foley",
    ),
    allow_noncommercial: bool = typer.Option(
        False, "--allow-noncommercial",
        help="Permit non-commercial-only models. Output cannot be monetized.",
    ),
    device: str = typer.Option(None, "--device", help="torch device, e.g. cuda or cpu"),
    preview: bool = typer.Option(
        False, "--preview",
        help="Placeholder tones instead of real audio, to check timing. Writes to preview/.",
    ),
) -> None:
    """Generate every cue, mix, master to the preset target, and mux back onto the video."""
    project = Path(project)
    cues_path = project / "cues.json"
    if not cues_path.exists():
        console.print(f"[red]no cues.json in {project}[/]")
        console.print(f"Run [bold]hudka scaffold {project}[/] to create one.")
        raise typer.Exit(1)

    try:
        sheet = CueSheet.load(cues_path)
    except Exception as exc:
        console.print(f"[red]cues.json is not valid:[/] {exc}")
        raise typer.Exit(1)

    if allow_noncommercial:
        console.print("[yellow]--allow-noncommercial: this render may not be monetized "
                      "or used in client work.[/]")

    try:
        result = render_mod.render(
            sheet, project, allow_noncommercial=allow_noncommercial,
            opted_in=set(engine), device=device, preview=preview,
            progress=lambda msg: console.print(f"[dim]{msg}[/]"),
        )
    except engines.LicenceError as exc:
        console.print(f"\n[red]licence check failed[/]\n{exc}")
        raise typer.Exit(2)
    except qa.QualityError as exc:
        console.print(f"\n[red]quality checks failed[/]\n{exc}")
        raise typer.Exit(3)
    except Exception as exc:
        console.print(f"[red]render failed:[/] {exc}")
        raise typer.Exit(1)

    from .timing import summary as timing_summary

    console.print(f"[dim]{timing_summary(result.timings)}[/]")
    if result.is_preview:
        console.print(
            f"\n[yellow]PREVIEW[/] {result.final_video}\n"
            "     placeholder tones only - no model ran, this is not real audio.\n"
            "     Drop --preview to render for real."
        )
        return

    if result.quality:
        for line in result.quality.warnings():
            console.print(f"[yellow]warning[/] {line}")

    off = abs(result.measured_lufs - sheet.target_lufs)
    verdict = "[green]on target[/]" if off <= 0.5 else f"[yellow]{off:.2f} LU off target[/]"
    if result.verdict == "warn":
        verdict += "  [yellow](with warnings above)[/]"
    console.print(
        f"\n[green]done[/] {result.final_video}\n"
        f"     {result.measured_lufs:.2f} LUFS  peak {result.measured_peak_db:.2f} dBTP  {verdict}\n"
        f"     {result.generated_count} generated, {result.cached_count} reused\n"
        f"     licence report: {result.licence_report}"
    )


@app.command()
def gui(
    workspace: Path = typer.Option(Path("out"), "--workspace", "-w",
                                   help="Directory holding your projects"),
    port: int = typer.Option(7860, "--port"),
    no_browser: bool = typer.Option(False, "--no-browser"),
) -> None:
    """Open the app: import a video, analyse, design cues, render, audition, download."""
    import threading
    import webbrowser

    import uvicorn

    from .ui.server import create_app

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}"

    console.print(f"[green]Hudka[/] running at [bold]{url}[/]")
    console.print(f"[dim]projects in {workspace.resolve()}  -  press Ctrl+C to stop[/]")

    if not no_browser:
        # Give uvicorn a moment to bind before the browser asks for the page.
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(workspace), host="127.0.0.1", port=port, log_level="warning")


@app.command()
def ui(
    project: Path = typer.Argument(..., help="Project directory"),
    port: int = typer.Option(7860, "--port"),
) -> None:
    """Open the app on the workspace containing this project."""
    project = Path(project)
    if not project.is_dir():
        console.print(f"[red]no such project directory: {project}[/]")
        raise typer.Exit(1)
    gui(workspace=project.parent, port=port, no_browser=False)


@app.command(name="licences")
def licences_cmd() -> None:
    """Show what every engine permits, including the one deliberately excluded."""
    table = Table(title="Engine licences", header_style="bold")
    # Fold rather than truncate: Rich's ellipsis is not representable in a cp1252 console.
    table.add_column("Engine", overflow="fold")
    table.add_column("Licence")
    table.add_column("Commercial")
    table.add_column("Conditions")

    for engine_id, lic in engines.LICENCE_TABLE.items():
        if not lic.commercial:
            mark, conditions = "[red]no[/]", "excluded from renders"
        elif lic.requires_optin:
            mark = "[yellow]opt-in[/]"
            conditions = "excludes " + ", ".join(lic.territory_exclusions)
        else:
            mark = "[green]yes[/]"
            conditions = (f"free under ${lic.revenue_cap_usd/1_000_000:.0f}M revenue"
                          if lic.revenue_cap_usd else "unrestricted")
        table.add_row(engine_id, lic.name, mark, conditions)

    console.print(table)
    console.print("\n[dim]Default stack is Stable Audio 3: the small models everywhere, "
                  "medium for beds where\nthe GPU has room for it (`hudka doctor` shows "
                  "which). Safe for worldwide distribution\nand for offering the capability "
                  "to others. ACE-Step is optional and has not been run here.\n"
                  "mmaudio is listed only to document why it is not used.[/]")


@app.command(name="presets")
def presets_cmd() -> None:
    """List delivery presets and what each one is aiming for."""
    table = Table(title="Presets", header_style="bold")
    table.add_column("Name")
    table.add_column("For")
    table.add_column("LUFS")
    table.add_column("SFX/min")
    table.add_column("Duck")

    for pre in presets.PRESETS.values():
        table.add_row(pre.name, pre.description, f"{pre.target_lufs:.0f}",
                      f"{pre.sfx_per_minute[0]}-{pre.sfx_per_minute[1]}",
                      f"{pre.duck_depth_db:.0f} dB")
    console.print(table)


@app.command()
def doctor() -> None:
    """Check that ffmpeg and the optional engines are actually available."""
    import shutil

    ok = True
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        console.print(f"{'[green]OK  [/]' if found else '[red]MISS[/]'} {tool}"
                      + (f"  [dim]{found}[/]" if found else "  [dim]required[/]"))
        ok = ok and bool(found)

    engine_hint = {
        "stable_audio_3": "git clone https://github.com/Stability-AI/stable-audio-3 (see README)",
        "acestep": "optional: git clone https://github.com/ACE-Step/ACE-Step-1.5",
    }
    for mod in ("scenedetect", "PIL", "stable_audio_3", "acestep"):
        try:
            __import__(mod)
            console.print(f"[green]OK  [/] {mod}")
        except ImportError:
            hint = engine_hint.get(mod, "required")
            console.print(f"[yellow]MISS[/] {mod}  [dim]{hint}[/]")
            ok = ok and mod in engine_hint

    console.print(f"\nmodel cache: [dim]{engines.model_dir()}[/]")

    import os
    from pathlib import Path

    from .engines import hardware
    from .engines.stable_audio3 import REQUIRED_FILES, StableAudio3Engine

    hw = hardware.detect(refresh=True)
    if hw.device == "cuda":
        console.print(f"[green]OK  [/] CUDA: {hw.gpu_name}, {hw.total_vram_gb:.1f} GB total, "
                      f"[bold]{hw.free_vram_gb:.1f} GB free[/] ({hw.free_source})  "
                      f"[dim]torch {hw.torch_build}[/]")
    elif hw.torch_build and "+cpu" in hw.torch_build:
        console.print(
            f"[red]WARN[/] torch {hw.torch_build} is the CPU-only build - generation will be "
            "very slow.\n       Installing stable-audio-3 pulls this in by mistake. Fix it with:\n"
            "       [bold]uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 "
            "--torch-backend=cu128[/]"
        )
    elif not hw.torch_build:
        console.print("[yellow]MISS[/] torch  [dim]installed with an engine; see README[/]")
    else:
        console.print("[yellow]MISS[/] CUDA not available - generation will use CPU")
    console.print(f"     RAM {hw.ram_gb:.0f} GB, {hw.cores} threads")

    # The decision the machine drives, in the words the GUI uses.
    console.print(f"\ntier [bold]{hw.tier.value}[/]: {hardware.reason(hw)}")
    for kind in ("music", "ambience", "sfx"):
        console.print(f"     {kind:9s} {hardware.engine_for(kind, hw.tier):28s} "
                      f"{hardware.steps_for(kind, hw.tier):>2} steps  "
                      f"{'fp16' if hw.device == 'cuda' else 'fp32'}")

    # Weights on disk, adopting any that are already here under a different cache.
    console.print("")
    for variant in ("small-sfx", "small-music", "medium"):
        engine = StableAudio3Engine(f"stable-audio-3-{variant}")
        try:
            engine.preflight(progress=lambda m: console.print(f"     [dim]{m}[/]"))
        except RuntimeError:
            pass                                   # the package line above already said so
        folder = Path(os.environ["HF_HUB_CACHE"]) / engines._repo_folder(engine.repo_id)
        present = engines.snapshot_complete(folder, REQUIRED_FILES) is not None
        console.print(f"{'[green]OK  [/]' if present else '[yellow]MISS[/]'} weights {engine.repo_id}"
                      + ("" if present else "  [dim]downloads on first use[/]"))

    if not ok:
        raise typer.Exit(1)


def main() -> None:  # pragma: no cover
    sys.exit(app())


if __name__ == "__main__":  # pragma: no cover
    main()
