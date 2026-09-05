"""Generation worker: one engine, one process - and now alive across jobs.

Run as `python -m hudka._worker`. Speaks JSON lines: a `hello` on stdin, then `ready` and
`loaded` on stdout; then one `job` per stdin line, answered with a `done` per cue and a
`job_done`; `ping`/`quit` as expected. Engines keep stderr for their own noise (tqdm,
warnings); a `--hudka job-end <id>` marker is written there after each job so the parent
can attribute stderr lines to the job that produced them.

Why a subprocess at all. Loading a second model into a process that has already loaded
and released one is unreliable here: it is pathologically slow at best, and at worst the
process dies outright - no exception, no traceback, just gone. That is survivable in a
CLI, but inside the GUI the generation thread shares the server's process, so a native
crash would take the whole app down with it and leave the user staring at a dead page.
One engine per process gives each model a clean CUDA context, reclaims memory completely
when the process ends, and turns a hard crash into an exit code the parent can explain.

Why it now stays alive. Measured on the development machine: loading the small model is
~11 s, the medium model ~21 s, and the first cue pays ~13 s of torch.compile warm-up on top.
Per render, with a fresh process per engine, that was 25 s or more of overhead before any
sound - and again on the next render. `engines/pool.py` decides how long a worker stays.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

JOB_END_MARKER = "--hudka job-end "


def _emit(**fields) -> None:
    print(json.dumps(fields), flush=True)


def _cuda_stats(reset_peak: bool = False) -> dict:
    """What this process holds on the card. Self-reported, because on Windows (WDDM)
    nvidia-smi cannot see per-process VRAM from outside."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        out = {"reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
               "peak_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        return out
    except Exception:
        return {}


def _watch_parent(pid: int) -> None:
    """Exit the moment the parent is gone, so a resident worker never outlives the GUI.

    On Windows a handle to the parent becomes signalled when it exits - however it exits -
    so this fires mid-cue, immediately. POSIX polls the parent pid once a second.
    """
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            handle = kernel32.OpenProcess(0x00100000, False, pid)        # SYNCHRONIZE
            if handle:
                kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
                os._exit(0)
        else:
            while os.getppid() == pid:
                time.sleep(1.0)
            os._exit(0)
    except Exception:  # pragma: no cover - the Job Object and atexit are the other layers
        pass


def main() -> int:
    # This process is about to create a CUDA context anyway, so the hardware probe may
    # fall back to torch here when nvidia-smi is not on PATH. The GUI server never may.
    os.environ.setdefault("HUDKA_ALLOW_TORCH_PROBE", "1")

    first = sys.stdin.readline()
    if not first.strip():
        return 0
    hello = json.loads(first)
    engine_id: str = hello["engine"]
    device: str | None = hello.get("device")
    if hello.get("parent_pid"):
        threading.Thread(target=_watch_parent, args=(int(hello["parent_pid"]),),
                         daemon=True).start()

    from .engines import build
    from .engines.base import GenerateRequest

    engine = build(engine_id, device)
    _emit(ready=engine_id, pid=os.getpid())

    # Load eagerly, so the parent sees load time apart from the first cue.
    started = time.perf_counter()
    load = getattr(engine, "load", None)
    if callable(load):
        load()
    describe = getattr(engine, "describe", None)
    info = describe(GenerateRequest(prompt="", duration=1.0, seed=0)) if callable(describe) else {}
    _emit(loaded=engine_id, seconds=round(time.perf_counter() - started, 2),
          device=info.get("device") or device or "cpu", precision=info.get("precision"),
          **_cuda_stats(reset_peak=True))

    for line in sys.stdin:                    # EOF: the parent is gone, or done with us
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "quit" in job:
            break
        if "ping" in job:
            _emit(pong=job["ping"], **_cuda_stats())
            continue
        if "job" not in job:
            continue

        job_id, job_started = job["job"], time.perf_counter()
        try:
            for item in job["cues"]:
                cue_started = time.perf_counter()
                req = GenerateRequest(
                    prompt=item["prompt"], duration=item["duration"], seed=item["seed"],
                    video=Path(item["video"]) if item.get("video") else None,
                    window=tuple(item["window"]) if item.get("window") else None,
                    extra=item.get("extra") or {},
                )
                engine.generate(req, Path(item["dest"]))
                # One line per finished cue, so the parent can report progress live.
                _emit(done=item["id"], seconds=round(time.perf_counter() - cue_started, 2))
        except Exception:
            # A half-failed CUDA context is not worth keeping: report and exit.
            traceback.print_exc()
            print(JOB_END_MARKER + job_id, file=sys.stderr, flush=True)
            _emit(error=job_id, message=traceback.format_exc().strip().splitlines()[-1])
            return 1
        print(JOB_END_MARKER + job_id, file=sys.stderr, flush=True)
        _emit(job_done=job_id, seconds=round(time.perf_counter() - job_started, 2),
              **_cuda_stats(reset_peak=True))

    engine.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
