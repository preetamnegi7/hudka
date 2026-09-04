"""Engine registry.

`build()` is the single place engines are constructed, so licence gating cannot be
bypassed by reaching for an implementation directly.
"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from .base import Engine, GenerateRequest, Licence, LicenceError, require_usable
from .licences import (
    CC_BY_NC,
    MIT,
    PUBLIC_DOMAIN,
    STABILITY_COMMUNITY,
    TENCENT_HUNYUAN_COMMUNITY,
)

__all__ = [
    "Engine", "GenerateRequest", "Licence", "LicenceError", "require_usable",
    "build", "model_dir", "DEFAULT_ENGINES", "LICENCE_TABLE",
]

#: Sensible default engine per cue kind — the worldwide-safe stack.
#:
#: The *small* variants are the defaults rather than `medium`, deliberately. Medium is
#: 2B parameters shipped as an 8.6GB float32 checkpoint, and loading it on a 12GB card
#: kills the process outright - no exception, no traceback, just gone. The small models
#: peak around 2.3GB, cover up to 120s, and are what most short-form and explainer work
#: needs anyway. `pick_bed_engine` promotes to medium only with both a reason and headroom.
DEFAULT_ENGINES = {
    "sfx": "stable-audio-3-small-sfx",
    "music": "stable-audio-3-small-music",
    "ambience": "stable-audio-3-small-sfx",
}

#: Longest bed the small models can produce; beyond this a bigger model would be needed.
SMALL_MAX_SECONDS = 120.0

#: Medium needs real headroom to load its float32 checkpoint without taking the process
#: down. Measured against a 12.9GB card, where it does not survive.
MEDIUM_MIN_VRAM_GB = 16.0


def available_vram_gb() -> float:
    """Total VRAM on the default CUDA device, or 0.0 when there is no GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        return 0.0


def pick_bed_engine(duration: float) -> str:
    """Engine for a music or ambience bed of this length.

    Small by default; medium only for beds it cannot cover, and only on a card with the
    headroom to load it. Otherwise the bed is looped to fill the range.
    """
    if duration <= SMALL_MAX_SECONDS:
        return DEFAULT_ENGINES["music"]
    if available_vram_gb() >= MEDIUM_MIN_VRAM_GB:
        return "stable-audio-3-medium"
    return DEFAULT_ENGINES["music"]


#: Every engine id and its licence, including the one deliberately excluded, so
#: `hudka licences` can show the reasoning rather than silently omitting it.
LICENCE_TABLE: dict[str, Licence] = {
    "stable-audio-3-medium": STABILITY_COMMUNITY,
    "stable-audio-3-small-sfx": STABILITY_COMMUNITY,
    "stable-audio-3-small-music": STABILITY_COMMUNITY,
    "acestep-1.5": MIT,
    "hunyuan-foley": TENCENT_HUNYUAN_COMMUNITY,
    "mmaudio": CC_BY_NC,  # listed to document the exclusion; no implementation exists
    "silence": PUBLIC_DOMAIN,
}


def _filesystem_of(drive: str) -> str:
    """Filesystem name for a Windows drive root, or '' when it cannot be determined."""
    if os.name != "nt":
        return ""
    import ctypes

    buf = ctypes.create_unicode_buffer(64)
    try:
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(drive), None, 0, None, None, None, buf, ctypes.sizeof(buf)
        )
    except Exception:
        return ""
    return buf.value if ok else ""


def _is_suitable(drive: str, need_gb: float) -> bool:
    """Whether model weights can safely live on this drive.

    Weights are read by memory-mapping files of several gigabytes. Doing that from a USB
    external disk, or from exFAT, fails with STATUS_IN_PAGE_ERROR - the process dies with
    no Python traceback at all, which is close to impossible to diagnose from the symptom.
    Free space alone is therefore the wrong thing to select on: an external backup drive
    usually has the most of it and is the worst possible choice.
    """
    if os.name == "nt":
        import ctypes

        DRIVE_FIXED = 3
        try:
            if ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(drive)) != DRIVE_FIXED:
                return False
        except Exception:
            return False
        # exFAT and FAT are the giveaway for a removable/portable volume, and neither is
        # dependable for large memory-mapped reads.
        if _filesystem_of(drive).upper() not in ("NTFS", "REFS"):
            return False
    try:
        return shutil.disk_usage(drive).free / 1e9 >= need_gb
    except OSError:
        return False


def model_dir(need_gb: float = 40.0) -> Path:
    """Where weights are cached.

    Prefers a fixed NTFS volume with room to spare - never an external or exFAT disk,
    whatever its free space. Override with HUDKA_MODEL_DIR.
    """
    configured = os.environ.get("HUDKA_MODEL_DIR")
    if configured:
        return Path(configured)

    if os.name == "nt":
        candidates = [f"{chr(letter)}:\\" for letter in range(ord("C"), ord("Z") + 1)]
        roomiest = [d for d in candidates if _is_suitable(d, need_gb)]
        if roomiest:
            # Most free space among the drives that are actually safe to use.
            best = max(roomiest, key=lambda d: shutil.disk_usage(d).free)
            return Path(best) / "hudka-models"

    return Path.home() / ".cache" / "hudka-models"


def _point_hf_cache_at_model_dir() -> None:
    """Send Hugging Face model downloads to the model directory, off the system drive.

    `HF_HUB_CACHE`, deliberately, and not `HF_HOME`: HF_HOME relocates the *token store*
    as well as the blob cache, so setting it silently hides an existing login and turns
    an authorised machine back into a 401. HF_HUB_CACHE moves only the downloaded models.

    Read when huggingface_hub is first imported, hence setting it here - this module is
    imported before any engine implementation pulls HF in.
    """
    if not os.environ.get("HF_HUB_CACHE") and not os.environ.get("HF_HOME"):
        os.environ["HF_HUB_CACHE"] = str(model_dir() / "huggingface")


_point_hf_cache_at_model_dir()


@lru_cache(maxsize=None)
def build(engine_id: str, device: str | None = None) -> Engine:
    """Construct an engine by id. Cached so weights load once per process."""
    if engine_id == "silence":
        from .stub import SilenceEngine

        return SilenceEngine()

    if engine_id.startswith("stable-audio-3"):
        from .stable_audio3 import StableAudio3Engine

        return StableAudio3Engine(engine_id, device=device, model_dir=model_dir())

    if engine_id == "acestep-1.5":
        from .acestep import AceStepEngine

        return AceStepEngine(device=device, model_dir=model_dir() / "acestep")

    if engine_id == "hunyuan-foley":
        from .hunyuan_foley import HunyuanFoleyEngine

        return HunyuanFoleyEngine(
            repo_dir=Path(os.environ.get("HUNYUAN_FOLEY_REPO", model_dir() / "HunyuanVideo-Foley")),
            model_dir=model_dir() / "hunyuan-foley",
        )

    if engine_id == "mmaudio":
        raise LicenceError(
            "mmaudio is not available in this tool.\n"
            "  Its weights are CC-BY-NC-4.0 (non-commercial) despite the MIT code, which "
            "rules out\n  monetized video and client work. Use hunyuan-foley for synced "
            "foley instead."
        )

    raise ValueError(f"unknown engine: {engine_id}")
