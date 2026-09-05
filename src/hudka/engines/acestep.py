"""ACE-Step 1.5 — optional music engine, MIT licensed and therefore unrestricted.

Worth reaching for over Stable Audio 3 when a piece needs to run past 380s (ACE-Step
handles up to 600s) or wants explicit song structure. On Windows the vLLM backend needs
Triton, which usually isn't present; ACE-Step falls back to PyTorch on its own, and this
wrapper forces that explicitly so the fallback isn't a surprise mid-render.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..audio import SAMPLE_RATE, fit_length, write_wav
from .base import Engine, GenerateRequest, Licence
from .licences import MIT


class AceStepEngine(Engine):
    id = "acestep-1.5"
    licence: Licence = MIT
    kinds = ("music",)
    max_duration = 600.0

    def __init__(self, device: str | None = None, model_dir: Path | None = None):
        self.device = device
        self.model_dir = model_dir
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        # Force the PyTorch LM backend: on Windows the vLLM path needs Triton, which the
        # standard install doesn't provide.
        os.environ.setdefault("ACESTEP_LM_BACKEND", "pt")
        try:
            from acestep.pipeline import ACEStepPipeline
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "ACE-Step is not installed.\n"
                '  uv pip install "acestep @ '
                'git+https://github.com/ACE-Step/ACE-Step-1.5"'
            ) from exc

        kwargs = {"dtype": "bfloat16"}
        if self.model_dir:
            kwargs["checkpoint_dir"] = str(self.model_dir)
        if self.device:
            kwargs["device"] = self.device
        self._pipeline = ACEStepPipeline(**kwargs)
        return self._pipeline

    def preflight(self) -> None:
        try:
            import acestep  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the optional install
            raise RuntimeError(
                "ACE-Step is not installed.\n"
                '  uv pip install "acestep @ '
                'git+https://github.com/ACE-Step/ACE-Step-1.5"'
            ) from exc

    def generate(self, req: GenerateRequest, out_path: Path) -> Path:
        duration = min(req.duration, self.max_duration)
        pipeline = self._load()
        audio = pipeline(
            prompt=req.prompt,
            lyrics=req.extra.get("lyrics", "[inst]"),  # instrumental unless asked otherwise
            audio_duration=duration,
            manual_seeds=[req.seed],
        )
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        return write_wav(out_path, fit_length(audio, duration), SAMPLE_RATE)

    def unload(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        try:  # pragma: no cover
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
