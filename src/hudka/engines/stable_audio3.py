"""Stable Audio 3 - the primary engine, used for both sound effects and music.

Chosen as the default because it is the only current option that clears every constraint
at once: trained entirely on licensed data, outputs are owned by the user and freely
commercialisable under $1M revenue, no territorial carve-out, and it fits a 12GB card -
IF it is loaded in the right order. The library moves the 9.2 GB float32 checkpoint onto
the GPU and only then halves it; `_load` below casts first, so medium settles at 4.6 GB of
weights and Stability's published 5.07-6.52GB peak for 5s-380s of generation.

Variants:
    medium       2.3B (1.45B DiT + 0.85B VAE), up to 380s, CUDA - beds of any length
    small-sfx    433M, up to 120s        - one-shot effects, fast iteration
    small-music  433M, up to 120s        - short music beds
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..audio import SAMPLE_RATE, fit_length, write_wav
from .base import Engine, GenerateRequest, Licence
from .licences import STABILITY_COMMUNITY

#: variant -> (model name, max duration, cue kinds it suits)
_VARIANTS = {
    "stable-audio-3-medium": ("medium", 380.0, ("music", "ambience", "sfx")),
    "stable-audio-3-small-sfx": ("small-sfx", 120.0, ("sfx", "ambience")),
    "stable-audio-3-small-music": ("small-music", 120.0, ("music",)),
}


def _is_gated(exc: Exception) -> bool:
    """A gated-repo refusal, however huggingface_hub happens to signal it."""
    try:
        from huggingface_hub.errors import GatedRepoError

        if isinstance(exc, GatedRepoError):
            return True
    except ImportError:  # pragma: no cover
        pass
    text = str(exc)
    return "gated repo" in text.lower() or "401 Client Error" in text


#: These checkpoints are adversarially post-trained for few-step sampling, so 8 is the
#: library default. Measured here, though, raising it markedly enriches sustained
#: material: on a 20s bed the spectral centroid goes 1116 Hz -> 2050 Hz between 8 and 50
#: steps, for about one extra second. A one-shot has far less spectrum to enrich and a
#: render contains many of them, so only beds pay for the extra steps.
DEFAULT_STEPS = 8
BED_STEPS = 50

#: Below about two seconds this model returns saturated garbage roughly half the time -
#: measured at 0.8s, 1.2s and 1.8s, all pinned to +-1.0 with a crest factor near 1.5 dB,
#: while 1.0s, 1.5s and everything from 2.0s up came back clean. It is not a threshold,
#: it is unreliable, so short cues are generated at a safe length and trimmed afterwards.
#: Costs nothing: generation time is flat in duration (~0.35s either way).
MIN_RELIABLE_SECONDS = 2.0

#: A generation whose peak-to-RMS is below this is saturated, not audio.
SATURATION_CREST_DB = 6.0

_STEPS_BY_VARIANT = {
    "medium": BED_STEPS,
    "small-music": BED_STEPS,
    "small-sfx": DEFAULT_STEPS,
}


class StableAudio3Engine(Engine):
    licence: Licence = STABILITY_COMMUNITY

    def __init__(self, engine_id: str, device: str | None = None,
                 model_dir: Path | None = None, steps: int | None = None):
        if engine_id not in _VARIANTS:
            raise ValueError(f"unknown Stable Audio 3 variant: {engine_id}")
        self.id = engine_id
        self.variant, self.max_duration, self.kinds = _VARIANTS[engine_id]
        self.device = device
        self.model_dir = model_dir
        #: None means the variant's own default. It used to be `steps: int = 8`, which
        #: made an explicit 8 indistinguishable from "unset" and silently promoted it to 50.
        self.steps = _STEPS_BY_VARIANT[self.variant] if steps is None else steps
        self._model = None

    @staticmethod
    def _require_package():
        try:
            from stable_audio_3 import StableAudioModel
        except ImportError as exc:  # pragma: no cover - depends on the optional install
            raise RuntimeError(
                "Stable Audio 3 is not installed.\n\n"
                '  uv pip install "stable-audio-3 @ '
                'git+https://github.com/Stability-AI/stable-audio-3"\n'
                "  uv pip install --reinstall torch==2.7.1 torchaudio==2.7.1 "
                "--torch-backend=cu128\n\n"
                "The weights are gated: accept the licence at\n"
                "  https://huggingface.co/stabilityai/stable-audio-3-medium\n"
                "then run `uv run hf auth login`.\n\n"
                "Until then, tick 'preview' in the app to render placeholder audio."
            ) from exc
        return StableAudioModel

    def preflight(self) -> None:
        """Fail before any cue is attempted if the package is not importable."""
        self._require_package()

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"

    def _load(self):
        if self._model is not None:
            return self._model
        StableAudioModel = self._require_package()
        # Weight location is controlled by the Hugging Face cache (HF_HOME), set in
        # `engines.__init__` - `from_pretrained` takes no cache argument of its own.
        try:
            self._model = StableAudioModel.from_pretrained(
                self.variant, device=self._resolve_device()
            )
        except Exception as exc:
            if _is_gated(exc):
                raise RuntimeError(self._gated_message()) from exc
            raise
        return self._model

    def _gated_message(self) -> str:
        return (
            "Stable Audio 3's weights are gated, and this machine is not authorised yet.\n\n"
            "Two one-off steps, both needing your own Hugging Face account:\n\n"
            "  1. Accept the licence (free, instant):\n"
            f"     https://huggingface.co/stabilityai/stable-audio-3-{self.variant}\n\n"
            "  2. Log this machine in:\n"
            "     uv run hf auth login\n\n"
            "Then render again. Meanwhile, tick 'preview' to render placeholder audio."
        )

    @property
    def repo_id(self) -> str:
        return f"stabilityai/stable-audio-3-{self.variant}"

    def is_authorised(self) -> bool:
        """Whether the gated weights can actually be fetched.

        `model_info` is not enough: it returns public metadata for a gated repo even when
        the file downloads are refused, so it reports success right up until the render
        fails. `auth_check` asks the question that actually matters.
        """
        try:
            from huggingface_hub import HfApi

            HfApi().auth_check(self.repo_id)
            return True
        except Exception:
            return False

    def _effective_steps(self) -> int:
        """Diffusion steps, capped on CPU where each one costs seconds rather than ms.

        A 50-step bed is 2.2s on this GPU and 228s on the CPU. The extra steps buy real
        spectral richness, but not at four minutes for twenty seconds of audio - so a
        machine without CUDA gets the fast setting instead of an unusable one.
        """
        if self._resolve_device() == "cpu":
            return min(self.steps, DEFAULT_STEPS)
        return self.steps

    def generate(self, req: GenerateRequest, out_path: Path) -> Path:
        duration = min(req.duration, self.max_duration)
        # Ask for a length the model handles reliably, then trim to what was requested.
        generate_for = max(duration, MIN_RELIABLE_SECONDS)

        model = self._load()
        options = {
            "prompt": req.prompt,
            "duration": generate_for,
            "steps": req.extra.get("steps") or self._effective_steps(),
            "seed": req.seed,
        }
        if req.extra.get("cfg_scale") is not None:
            options["cfg_scale"] = float(req.extra["cfg_scale"])
        if req.extra.get("negative_prompt"):
            options["negative_prompt"] = req.extra["negative_prompt"]

        audio = model.generate(**options)
        if hasattr(audio, "detach"):
            # Comes back as a batch; take the first item and hand numpy a plain array.
            audio = audio.detach().float().cpu()
            if audio.ndim == 3:
                audio = audio[0]
            audio = audio.numpy()

        crest = _crest_db(audio)
        if crest < SATURATION_CREST_DB:
            # Report rather than silently re-roll: provenance records a seed, and that
            # seed has to reproduce the file it is recorded against.
            print(
                f"warning: {self.id} returned saturated audio for {req.prompt[:40]!r} "
                f"(crest {crest:.1f} dB) - the cue will sound like distortion",
                file=sys.stderr,
            )

        return write_wav(out_path, fit_length(audio, duration), SAMPLE_RATE)

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        try:  # pragma: no cover - only meaningful with a GPU present
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass


def _crest_db(samples) -> float:
    """Peak-to-RMS in dB. Near zero means the signal is saturated, not dynamic."""
    a = np.asarray(samples, dtype=np.float32)
    mono = a.mean(axis=0) if a.ndim == 2 and a.shape[0] <= 2 else a.reshape(-1)
    peak = float(np.abs(mono).max())
    rms = float(np.sqrt((mono**2).mean()))
    if peak <= 1e-9 or rms <= 1e-9:
        return 0.0
    return 20.0 * float(np.log10(peak / rms))
