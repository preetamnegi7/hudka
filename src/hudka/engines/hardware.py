"""What this machine can run, measured rather than assumed.

Two numbers people confuse: TOTAL VRAM decides nothing, FREE VRAM decides everything. The
development machine is a 12.9 GB card with 7-8 GB free at rest, because Edge, Spotify, Teams
and an LLM app hold the rest - and it was permanently gated onto the small models by a check
against the total.

On Windows the free figure has to come from nvidia-smi. `torch.cuda.mem_get_info()` counts
other applications' evictable memory as free (11.6 GB reported against 7.6 GB physically
free, same instant) and creates a ~180 MB CUDA context in whichever process asks - which,
for the GUI, is the server that lives all day. So this module never imports torch: the GPU
comes from nvidia-smi, whether torch can use it comes from the wheel's version tag ("+cu128"
vs "+cpu" - a metadata read, no import), RAM comes from the OS. The one exception is a
worker process that is about to create a CUDA context anyway; it opts in to the torch
fallback with HUDKA_ALLOW_TORCH_PROBE=1, for machines without nvidia-smi on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from enum import Enum

MIB_TO_GB = 1.048576 / 1000  # nvidia-smi reports MiB; everything here is decimal GB


class Tier(str, Enum):
    #: No usable CUDA: today's behaviour, exactly - small models, beds capped at 8 steps.
    CPU = "cpu"
    #: CUDA, but medium does not fit in the VRAM that is free right now.
    GPU_LITE = "gpu-lite"
    #: Medium fits a 120 s bed with headroom.
    GPU_MEDIUM = "gpu-medium"
    #: 16 GB or more free: medium fits anything it can generate, 380 s included.
    GPU_LARGE = "gpu-large"


#: MEASURED on the development machine (RTX 4070 12.9 GB, driver 591.86, torch
#: 2.7.1+cu128, fp16 DiT/VAE, bf16 text encoder), not derived - see docs/measurements.md.
#: Medium's PHYSICAL footprint while generating - nvidia-smi's number, the one that decides
#: whether the desktop stays up - is 7.9 GB and FLAT from a 30 s bed to a 380 s one: the DiT
#: works on a fixed latent length, so duration does not move it (a 5 s cue peaks lower, at
#: 7.2 GB). 4.95 GB of that is resident weights after loading. torch's own allocator reports
#: 8.4 GB reserved, ~3 GB of it reclaimable cache between renders. A first prediction from
#: parameter counts said 7.0 GB rising to 8.1; it was wrong in both shape and level, which is
#: why these are measured.
MEDIUM_RUN_PEAK_GB = 7.9
MEDIUM_RESIDENT_GB = 4.95
SMALL_RUN_PEAK_GB = 2.1          # small-music at 120 s: 2.13 GB physical, 1.14 GB resident
RUN_MARGIN_GB = 0.5              # the desktop keeps moving underneath
#: Without bf16 the T5Gemma text encoder lands in fp32 and costs this much more.
TEXT_ENCODER_EXTRA_FP32_GB = 0.57

#: Free-VRAM thresholds in decimal GB.
MEDIUM_MIN_FREE_GB = MEDIUM_RUN_PEAK_GB + RUN_MARGIN_GB     # 8.4
LARGE_MIN_FREE_GB = 16.0
#: The float32 checkpoint (9.2 GB) and its state dict are both staged in RAM before the cast
#: to fp16, so a 16 GB laptop stays on the small models however big its card.
MEDIUM_MIN_RAM_GB = 20.0

MEDIUM = "stable-audio-3-medium"
SMALL_FOR_KIND = {
    "sfx": "stable-audio-3-small-sfx",
    "ambience": "stable-audio-3-small-sfx",
    "music": "stable-audio-3-small-music",
}
FAST_STEPS = 8
BED_STEPS = 50

_TTL_SECONDS = 5.0
_cache: tuple[float, "Hardware"] | None = None


@dataclass(frozen=True)
class Hardware:
    device: str                     # "cuda" | "cpu"
    gpu_name: str = ""
    total_vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    free_source: str = "none"       # "nvidia-smi" | "torch" | "forced" | "none"
    bf16: bool = False
    ram_gb: float = 0.0
    cores: int = 1
    torch_build: str = ""

    @property
    def tier(self) -> Tier:
        if self.device != "cuda":
            return Tier.CPU
        if self.free_vram_gb >= LARGE_MIN_FREE_GB:
            return Tier.GPU_LARGE
        if self.free_vram_gb >= MEDIUM_MIN_FREE_GB and self.ram_gb >= MEDIUM_MIN_RAM_GB:
            return Tier.GPU_MEDIUM
        return Tier.GPU_LITE

    @property
    def held_by_others_gb(self) -> float:
        """VRAM that something other than Hudka is using right now."""
        return max(0.0, self.total_vram_gb - self.free_vram_gb)


# ------------------------------------------------------------------- the fit rule


def medium_need_gb(duration_s: float, hw: Hardware) -> float:
    """Free VRAM medium needs to generate a bed this long, on this machine.

    Measured flat: a 30 s bed and a 380 s bed peak at the same 7.9 GB, so `duration_s` is
    kept for the callers' sake and does not move the answer. Without bf16 the text encoder
    costs 0.57 GB more."""
    del duration_s
    return MEDIUM_RUN_PEAK_GB + (0.0 if hw.bf16 else TEXT_ENCODER_EXTRA_FP32_GB) + RUN_MARGIN_GB


def medium_fits(duration_s: float, hw: Hardware) -> bool:
    return (hw.device == "cuda" and hw.ram_gb >= MEDIUM_MIN_RAM_GB
            and hw.free_vram_gb >= medium_need_gb(duration_s, hw))


def engine_for(kind: str, tier: Tier) -> str:
    """The engine `auto` quality uses for a cue of this kind on this tier.

    Medium takes the beds on the GPU tiers that fit it. One-shots stay on small-sfx: that
    checkpoint is post-trained for exactly them, costs 0.32 s a cue here, and whether medium
    is better for a 2 s hit is unknown in both directions until it is measured.
    """
    if tier in (Tier.GPU_MEDIUM, Tier.GPU_LARGE) and kind in ("music", "ambience"):
        return MEDIUM
    return SMALL_FOR_KIND[kind]


def steps_for(kind: str, tier: Tier, quality: str = "auto") -> int:
    """Diffusion steps for a cue of this kind. The CPU row is today's behaviour exactly."""
    if quality == "fast" or tier is Tier.CPU:
        return FAST_STEPS
    return BED_STEPS if kind in ("music", "ambience") else FAST_STEPS


# --------------------------------------------------------------------- probes


def _probe_nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None

    def query(fields: str) -> list[str] | None:
        try:
            out = subprocess.run(
                [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]

    core = query("name,memory.total,memory.free")
    if not core or len(core) < 3:
        return None
    try:
        total, free = float(core[1]) * MIB_TO_GB, float(core[2]) * MIB_TO_GB
    except ValueError:
        return None
    # compute_cap is a newer query field; an old driver rejects the whole query, so ask
    # separately and treat "unknown" as "no bf16" - which only makes the fit rule stricter.
    cap = 0.0
    extra = query("compute_cap")
    if extra:
        try:
            cap = float(extra[0])
        except ValueError:
            cap = 0.0
    return {"name": core[0], "total_gb": total, "free_gb": free, "compute_cap": cap}


def _torch_build() -> str:
    """The installed torch wheel's version string, WITHOUT importing torch.

    "2.7.1+cu128" is a CUDA build, "2.7.1+cpu" is not - the tag is how a machine with an
    NVIDIA card and the wrong wheel is still told the truth (it has happened here).
    """
    try:
        from importlib.metadata import version

        return version("torch")
    except Exception:
        return ""


def _torch_probe() -> dict | None:
    """Fallback for machines without nvidia-smi. Imports torch, so only in a worker."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        return {"name": props.name, "total_gb": total / 1e9, "free_gb": free / 1e9,
                "compute_cap": float(f"{props.major}.{props.minor}")}
    except Exception:
        return None


def _ram_gb() -> float:
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 1e9
            return 0.0
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return 0.0


def _force(hw: Hardware, tier_name: str) -> Hardware:
    """HUDKA_FORCE_TIER=cpu|gpu-lite|gpu-medium|gpu-large, for support and for tests.

    Sets the fields that DRIVE the tier to values that produce it, so everything that reads
    free VRAM or RAM sees a consistent picture rather than a label that disagrees with them.
    """
    try:
        tier = Tier(tier_name)
    except ValueError:
        return hw
    if tier is Tier.CPU:
        return replace(hw, device="cpu", free_source="forced")
    # Derived from the thresholds, so a re-measured constant cannot strand this.
    free = {Tier.GPU_LITE: MEDIUM_MIN_FREE_GB - 3.0, Tier.GPU_MEDIUM: MEDIUM_MIN_FREE_GB + 1.0,
            Tier.GPU_LARGE: LARGE_MIN_FREE_GB + 4.0}[tier]
    return replace(hw, device="cuda", gpu_name=hw.gpu_name or "forced GPU",
                   total_vram_gb=max(hw.total_vram_gb, free), free_vram_gb=free,
                   free_source="forced", bf16=True, ram_gb=max(hw.ram_gb, MEDIUM_MIN_RAM_GB))


def detect(refresh: bool = False) -> Hardware:
    """This machine, right now. Never raises; cached for a few seconds.

    "Right now" matters: free VRAM moves as other applications open and close, so the tier
    can legitimately change between two page loads. The cache only stops a burst of calls
    from spawning a burst of nvidia-smi processes.
    """
    global _cache
    now = time.monotonic()
    if not refresh and _cache and now - _cache[0] < _TTL_SECONDS:
        return _cache[1]

    hw = Hardware(device="cpu", ram_gb=_ram_gb(), cores=os.cpu_count() or 1,
                  torch_build=_torch_build())
    try:
        gpu = _probe_nvidia_smi()
        source = "nvidia-smi"
        if gpu is None and os.environ.get("HUDKA_ALLOW_TORCH_PROBE") == "1":
            gpu = _torch_probe()
            source = "torch"
        cuda_build = bool(hw.torch_build) and "+cpu" not in hw.torch_build
        if gpu and cuda_build:
            hw = replace(hw, device="cuda", gpu_name=gpu["name"],
                         total_vram_gb=gpu["total_gb"], free_vram_gb=gpu["free_gb"],
                         free_source=source, bf16=gpu.get("compute_cap", 0.0) >= 8.0)
    except Exception:
        pass
    forced = os.environ.get("HUDKA_FORCE_TIER")
    if forced:
        hw = _force(hw, forced)
    _cache = (now, hw)
    return hw


# --------------------------------------------------------------------- words


def summary(hw: Hardware) -> str:
    """One line for the header: what the card is, what is free, what that buys."""
    if hw.device != "cuda":
        return "No NVIDIA GPU in use · small models, fast steps"
    quality = {Tier.GPU_LARGE: "best quality (medium, room to spare)",
               Tier.GPU_MEDIUM: "best quality (medium beds)",
               Tier.GPU_LITE: "small models"}[hw.tier]
    return (f"{hw.gpu_name.replace('NVIDIA GeForce ', '')} · {hw.free_vram_gb:.1f} GB free "
            f"of {hw.total_vram_gb:.1f} · {quality}")


def reason(hw: Hardware) -> str:
    """Why this tier, in plain words - including the one thing the user can change."""
    if hw.device != "cuda":
        if hw.torch_build and "+cpu" in hw.torch_build:
            return ("torch is the CPU-only build, so the GPU is not used. Setup.bat "
                    "reinstalls the CUDA build.")
        return "No NVIDIA GPU detected, so the small models run on the CPU."
    if hw.tier is Tier.GPU_LARGE:
        return f"{hw.free_vram_gb:.1f} GB free: the medium model fits at any length."
    if hw.tier is Tier.GPU_MEDIUM:
        return (f"{hw.free_vram_gb:.1f} GB free: the medium model fits (it needs about "
                f"{medium_need_gb(0.0, hw):.1f} GB while generating).")
    need = medium_need_gb(120.0, hw)
    if hw.ram_gb and hw.ram_gb < MEDIUM_MIN_RAM_GB:
        return (f"Only {hw.ram_gb:.0f} GB of RAM: loading the medium model stages a 9 GB "
                f"checkpoint in memory first, so it stays on the small models.")
    if hw.total_vram_gb >= need + 1.0:
        return (f"Medium would fit on this card, but other applications are holding "
                f"{hw.held_by_others_gb:.1f} GB of its memory ({hw.free_vram_gb:.1f} GB free "
                f"of {hw.total_vram_gb:.1f}; it needs about {need:.1f}). Close browsers "
                f"playing video, games and editors, then reload for the best quality.")
    return (f"{hw.total_vram_gb:.1f} GB card: the medium model needs about {need:.1f} GB "
            f"free, so the small models are used.")

