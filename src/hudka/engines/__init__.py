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
from . import hardware
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
    "adopt_from_default_cache", "snapshot_complete",
]

#: Sensible default engine per cue kind — the worldwide-safe stack.
#:
#: The *small* variants are the defaults rather than `medium`, deliberately: this is the
#: row every machine can run, CPU included. Medium is 2.3B parameters (1.45B DiT + 0.85B
#: VAE) shipped as a 9.2 GB float32 checkpoint; the library moves it onto the card BEFORE
#: halving it, which is why loading it used to kill a 12 GB process with no traceback (see
#: StableAudio3Engine._load for the fix). The small models peak around 2.3GB and cover up
#: to 120s. `pick_bed_engine` promotes to medium only when this machine's free VRAM says so.
DEFAULT_ENGINES = {
    "sfx": "stable-audio-3-small-sfx",
    "music": "stable-audio-3-small-music",
    "ambience": "stable-audio-3-small-sfx",
}

#: Longest bed the small models can produce; beyond this a bigger model would be needed.
SMALL_MAX_SECONDS = 120.0


def available_vram_gb() -> float:
    """FREE VRAM in GB, physically, from nvidia-smi - or 0.0 without a usable GPU.

    This used to return the card's TOTAL, which decides nothing: the development machine
    is a 12.9 GB card with 7-8 GB free at rest, because the desktop holds the rest. See
    `hardware` for why the free number must not come from torch on Windows.
    """
    return hardware.detect().free_vram_gb


def max_seconds(engine_id: str) -> float:
    """Longest single-pass generation for an engine id (120 s small, 380 s medium)."""
    from .stable_audio3 import _VARIANTS

    return _VARIANTS[engine_id][1] if engine_id in _VARIANTS else SMALL_MAX_SECONDS


def pick_bed_engine(duration: float, hw: "hardware.Hardware | None" = None,
                    kind: str = "music") -> str:
    """Engine for a music or ambience bed of this length, on this machine.

    Medium whenever the tier allows it AND the VRAM free right now covers a bed this long;
    otherwise the small model, which the scaffold loops past 120 s. The check is against
    FREE memory: a 12.9 GB card with a browser holding 5 GB is a 7.9 GB card.
    """
    hw = hw or hardware.detect()
    if (hardware.engine_for(kind, hw.tier) == hardware.MEDIUM
            and hardware.medium_fits(min(duration, max_seconds(hardware.MEDIUM)), hw)):
        return hardware.MEDIUM
    return DEFAULT_ENGINES[kind]


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


# ------------------------------------------------- weights already on this machine

#: The only repositories adoption will ever copy. Structural, not a comment: the default
#: cache on the development machine also holds facebook/musicgen-small, whose weights are
#: CC-BY-NC, and nothing in this codebase may ever point at it.
ADOPTABLE_PREFIX = "stabilityai/stable-audio-3-"


def _default_hf_cache() -> Path:
    """Where huggingface_hub puts things when nobody redirects it (~/.cache/huggingface/hub).

    Not HF_HUB_CACHE - that is the app cache we redirect TO. The two differ on any machine
    that downloaded weights before Hudka pointed the cache elsewhere.
    """
    from huggingface_hub import constants

    return Path(constants.default_cache_path)


def _repo_folder(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def snapshot_complete(repo_dir: Path, required: tuple[str, ...]) -> Path | None:
    """The snapshot directory if `repo_dir` holds a complete revision, else None."""
    try:
        sha = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    snap = repo_dir / "snapshots" / sha
    blobs = repo_dir / "blobs"
    if blobs.is_dir() and any(blobs.glob("*.incomplete")):
        return None
    for rel in required:                      # Path.exists() follows the symlinks
        f = snap / rel
        try:
            if not f.exists() or f.stat().st_size == 0:
                return None
        except OSError:
            return None
    return snap


def _copy_tree(src: Path, dst: Path, say) -> None:
    """Copy an HF repo folder. Relative symlinks are recreated where the OS allows and
    dereferenced where it does not; `*.incomplete` and lock files are left behind."""
    total = sum(p.stat().st_size for p in src.rglob("*")
                if p.is_file() and not p.is_symlink())
    done, last = 0, -1
    for root, _dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.endswith(".incomplete") or name.endswith(".lock"):
                continue
            s, d = Path(root) / name, dst / rel / name
            if s.is_symlink():
                target = os.readlink(s)
                try:
                    os.symlink(target, d)
                except OSError:
                    shutil.copy2(s.resolve(), d)
                continue
            shutil.copy2(s, d)
            done += s.stat().st_size
            pct = int(done * 10 / total) if total else 10
            if pct != last:
                last = pct
                say(f"  copied {done / 1e9:.1f} / {total / 1e9:.1f} GB")


def adopt_from_default_cache(repo_id: str, progress=None) -> Path | None:
    """Copy a complete snapshot from the default HF cache into the app cache.

    The medium checkpoint was found fully downloaded on the development machine - 9.2 GB,
    authorised - in the cache Hudka never reads, because the app redirects HF_HUB_CACHE
    off the system drive. Downloading it again would be absurd.

    Copies, never moves, so whatever put the weights there keeps working. Reads nothing
    outside `repo_id`'s own folder, and only Stable Audio 3 repositories are eligible.
    Copies into a sibling `.adopting` directory and renames at the end, so the library can
    never see a half-copied folder. Returns the adopted folder, or None when there was
    nothing to do.
    """
    if not repo_id.startswith(ADOPTABLE_PREFIX):
        raise ValueError(f"{repo_id} is not adoptable: only Stable Audio 3 weights are")
    say = progress or (lambda _: None)
    from .stable_audio3 import REQUIRED_FILES

    app_cache = Path(os.environ["HF_HUB_CACHE"])
    folder = _repo_folder(repo_id)
    dst, src = app_cache / folder, _default_hf_cache() / folder
    if dst.exists() and src.resolve() == dst.resolve():
        return None                           # the app cache IS the default cache here
    if snapshot_complete(dst, REQUIRED_FILES) or not snapshot_complete(src, REQUIRED_FILES):
        return None

    size = sum(p.stat().st_size for p in src.rglob("*") if p.is_file() and not p.is_symlink())
    say(f"adopting {repo_id} from {src.parent} ({size / 1e9:.1f} GB) - copying, the original stays")
    staging = app_cache / (folder + ".adopting")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        app_cache.mkdir(parents=True, exist_ok=True)
        _copy_tree(src, staging, say)
        if dst.exists():                      # something raced us; theirs wins
            shutil.rmtree(staging, ignore_errors=True)
            return None
        os.replace(staging, dst)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    say(f"adopted {repo_id}")
    return dst


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
