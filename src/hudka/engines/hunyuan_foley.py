"""HunyuanVideo-Foley — opt-in engine for true frame-synced foley.

Unlike every other engine here, this one conditions on the picture itself, so hits land
on the actual motion rather than on a timestamp a human chose. That is a real quality
jump for footage with lots of physical action.

It is opt-in rather than default because its licence Territory excludes the EU, UK and
South Korea, which sits awkwardly with a globally distributed video. See
`licences.TENCENT_HUNYUAN_COMMUNITY`. The gate lives in `base.require_usable`.

VRAM on a 12GB card: the XXL model needs `--enable_offload` (20GB without it); XL needs
8GB offloaded. Offload is therefore forced on by default here.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from ..audio import SAMPLE_RATE, fit_length, read_wav, write_wav
from .base import Engine, GenerateRequest, Licence
from .licences import TENCENT_HUNYUAN_COMMUNITY


class HunyuanFoleyEngine(Engine):
    id = "hunyuan-foley"
    licence: Licence = TENCENT_HUNYUAN_COMMUNITY
    kinds = ("sfx", "ambience")

    def __init__(self, repo_dir: Path, model_dir: Path, variant: str = "xxl",
                 offload: bool = True):
        self.repo_dir = Path(repo_dir)
        self.model_dir = Path(model_dir)
        self.variant = variant
        self.offload = offload

    def generate(self, req: GenerateRequest, out_path: Path) -> Path:
        if req.video is None:
            raise ValueError(
                "hunyuan-foley conditions on picture and needs a video; it cannot generate "
                "from a prompt alone. Use a Stable Audio 3 engine for prompt-only cues."
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            clip = tmp_dir / "clip.mp4"
            self._extract_window(req, clip)

            cmd = [
                "python", "infer.py",
                "--model_path", str(self.model_dir),
                "--single_video", str(clip),
                "--single_prompt", req.prompt,
                "--output_dir", str(tmp_dir / "out"),
            ]
            if self.offload:
                cmd.append("--enable_offload")

            proc = subprocess.run(cmd, cwd=self.repo_dir, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"HunyuanVideo-Foley failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}"
                )

            produced = sorted((tmp_dir / "out").rglob("*.wav"))
            if not produced:
                raise RuntimeError("HunyuanVideo-Foley produced no audio")

            samples, _ = read_wav(produced[0])
            return write_wav(out_path, fit_length(samples, req.duration), SAMPLE_RATE)

    def _extract_window(self, req: GenerateRequest, dest: Path) -> None:
        """Cut just the window the cue covers — the model works on short segments."""
        start = req.window[0] if req.window else 0.0
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", f"{start:.3f}", "-t", f"{req.duration:.3f}",
             "-i", str(req.video), "-an", "-c:v", "libx264", "-preset", "veryfast", str(dest)],
            check=True, capture_output=True,
        )

    def unload(self) -> None:
        return  # runs out-of-process; nothing is held in this process
