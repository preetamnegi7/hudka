"""Resident generation workers: one engine, one process, alive across renders.

Every render used to spawn a fresh process per engine, load the weights, pay the first
cue's torch.compile warm-up, generate, and exit - then do it all again for the next engine
and again on the next render. Measured on the development machine that was ~11 s of load
for a small model, ~21 s for medium, plus ~13 s of warm-up each, before any sound. Keeping
the process alive between jobs removes all of it from the second render on.

What does NOT change: one engine per process (loading a second model into a used process
is unreliable here - see _worker.py), a native crash kills only that worker and is
explained by exit code, and the parent never imports torch.

What decides whether a worker stays: FREE VRAM, from the same hardware probe the tiers use.
A worker may sit resident only while at least KEEP_MARGIN_GB stays free for the desktop,
and it is retired after IDLE_SECONDS without work, so Edge and Teams get the card back
when the user has walked away. HUDKA_WORKER_IDLE_MINUTES=0 restores the old behaviour.

Workers never outlive the GUI. Three layers, each sufficient for its case: a Windows Job
Object with KILL_ON_JOB_CLOSE (the OS kills every worker the instant the parent's handle
closes, however the parent died), a parent-pid watchdog inside the worker, and an atexit
/ FastAPI shutdown hook that asks nicely first.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from . import hardware

#: Leave the desktop this much while a model sits idle: compositor, browser and call video
#: stutter under about a gigabyte.
KEEP_MARGIN_GB = 1.0
RUN_MARGIN_GB = hardware.RUN_MARGIN_GB
#: Ten minutes: a tweak-and-listen loop has 30 s to 5 min gaps - listening to a four-minute
#: video IS four minutes - so five would expire mid-session. At ten the user has left.
IDLE_SECONDS = float(os.environ.get("HUDKA_WORKER_IDLE_MINUTES", "10")) * 60.0
JOB_END_MARKER = "--hudka job-end "
STDERR_TAIL_LINES = 400

#: Physical peak while generating, per engine, until a worker has reported its own.
DEFAULT_PEAK_GB = {
    "stable-audio-3-medium": hardware.MEDIUM_RUN_PEAK_GB,
    "stable-audio-3-small-sfx": hardware.SMALL_RUN_PEAK_GB,
    "stable-audio-3-small-music": hardware.SMALL_RUN_PEAK_GB,
    "acestep-1.5": 8.0,
    "hunyuan-foley": 0.0,      # runs out of process
    "silence": 0.0,
}


def peak_of(engine_id: str) -> float:
    return DEFAULT_PEAK_GB.get(engine_id, 2.5)


@dataclass
class JobReport:
    """What one engine's batch cost, for the render's timings."""

    engine_id: str = ""
    start_seconds: float = 0.0      # process spawn to `ready`
    load_seconds: float = 0.0       # weights (0.0 when the worker was already resident)
    generate_seconds: float = 0.0
    cues: dict[str, float | None] = field(default_factory=dict)
    resident: bool = False
    device: str | None = None
    precision: str | None = None
    peak_gb: float | None = None


class WorkerDied(RuntimeError):
    """The worker process ended while we needed it. The message is the diagnosis."""


# ------------------------------------------------------------- exit diagnosis
# Moved here from render.py, unchanged; render.py re-exports them for its tests.

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
        return ("CUDA ran out of VRAM. Close other GPU applications (`hudka doctor` shows "
                "how much is free), or use the small engines - stable-audio-3-small-sfx for "
                "effects, stable-audio-3-small-music for beds - which peak near 2 GB.")
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


# ------------------------------------------------------------ process lifetime


class _JobObject:
    """A Windows Job Object that kills every assigned process when its handle closes.

    The handle lives in this (parent) process, so the OS closes it when the parent dies
    for ANY reason and the workers go with it. No-op elsewhere; a failed assignment is
    ignored because the worker's own parent watchdog covers that case.
    """

    def __init__(self) -> None:
        self.handle = None
        if os.name != "nt":
            return
        try:
            import ctypes
            from ctypes import wintypes

            k = ctypes.windll.kernel32
            k.CreateJobObjectW.restype = ctypes.c_void_p
            k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                                  ctypes.c_void_p, ctypes.c_ulong]
            k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(name, ctypes.c_ulonglong) for name in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class BASIC(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                            ("PerJobUserTimeLimit", ctypes.c_longlong),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class EXTENDED(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            handle = k.CreateJobObjectW(None, None)
            if not handle:
                return
            info = EXTENDED()
            info.BasicLimitInformation.LimitFlags = 0x2000     # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if k.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                self.handle = handle
        except Exception:  # pragma: no cover
            self.handle = None

    def assign(self, proc: subprocess.Popen) -> None:
        if not self.handle:
            return
        try:
            import ctypes

            ctypes.windll.kernel32.AssignProcessToJobObject(self.handle, int(proc._handle))  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass


_JOB = _JobObject()


def pid_alive(pid: int) -> bool:
    """Is a process with this pid still running? Used by tests and status."""
    try:
        if os.name == "nt":
            import ctypes

            k = ctypes.windll.kernel32
            k.OpenProcess.restype = ctypes.c_void_p
            k.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            k.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = k.OpenProcess(0x1000, False, pid)                  # QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            code = ctypes.c_ulong()
            ok = k.GetExitCodeProcess(handle, ctypes.byref(code))
            k.CloseHandle(handle)
            return bool(ok) and code.value == 259                       # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:  # pragma: no cover
        return False


def _free_vram_gb() -> float | None:
    """Physical free VRAM right now, or None when there is no GPU to budget."""
    hw = hardware.detect(refresh=True)
    if hw.device != "cuda" or not hw.free_vram_gb:
        return None
    return hw.free_vram_gb


# ------------------------------------------------------------------ the worker


class EngineWorker:
    """One engine, one process, alive across jobs."""

    def __init__(self, engine_id: str, device: str | None):
        self.engine_id = engine_id
        self.device = device
        self.proc: subprocess.Popen | None = None
        self.info: dict = {}
        self.lock = threading.Lock()
        self.busy = False
        self.last_used = time.monotonic()
        self.start_seconds = 0.0
        self.load_seconds = 0.0
        self.idle_gb: float | None = None
        self.load_peak_gb: float | None = None
        self.run_peak_gb: float | None = None
        self._fresh = True
        self._tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._job_lines: list[str] = []
        self._job_id: str | None = None
        self._job_end = threading.Event()

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        started = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "hudka._worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        _JOB.assign(self.proc)
        threading.Thread(target=self._drain, daemon=True, name=f"hudka-stderr-{self.engine_id}").start()
        self._send({"hello": 1, "engine": self.engine_id, "device": self.device,
                    "parent_pid": os.getpid()})
        self._expect("ready")
        self.start_seconds = round(time.perf_counter() - started, 2)
        self.info = self._expect("loaded")
        self.load_seconds = float(self.info.get("seconds", 0.0) or 0.0)
        self.idle_gb = self.info.get("reserved_gb")
        self.load_peak_gb = self.info.get("peak_gb")
        self._fresh = True
        self.last_used = time.monotonic()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self, grace: float = 5.0) -> None:
        if not self.alive():
            return
        try:
            self._send({"quit": 1})
            assert self.proc and self.proc.stdin
            self.proc.stdin.close()
            self.proc.wait(timeout=grace)
        except Exception:
            try:
                self.proc.kill()  # type: ignore[union-attr]
            except Exception:  # pragma: no cover
                pass

    def stderr_text(self) -> str:
        return "\n".join(self._tail)

    # -- pipes -----------------------------------------------------------------

    def _send(self, message: dict) -> None:
        assert self.proc and self.proc.stdin
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            self.proc.wait()
            raise WorkerDied(_worker_error(self.engine_id, self.proc.returncode, self.stderr_text())) from exc

    def _drain(self) -> None:
        """Read stderr for the worker's whole life. readline keeps consuming a flood with
        no newline in it, so the pipe never fills - the deadlock a 36-cue render once hit."""
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self._tail.append(line.rstrip("\n"))
            if self._job_id is not None:
                self._job_lines.append(line)
                # `in`, not startswith: the marker may follow a partial line of tqdm output.
                if JOB_END_MARKER + self._job_id in line:
                    self._job_end.set()
        self._job_end.set()                     # EOF: the process is gone; wake the waiter

    def _read(self) -> dict:
        assert self.proc and self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                self.proc.wait()
                raise WorkerDied(_worker_error(self.engine_id, self.proc.returncode, self.stderr_text()))
            if line.lstrip().startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _expect(self, key: str) -> dict:
        while True:
            msg = self._read()
            if key in msg:
                return msg
            if "error" in msg:
                assert self.proc
                self.proc.wait()
                raise WorkerDied(_worker_error(self.engine_id, self.proc.returncode or 1, self.stderr_text()))

    # -- work -----------------------------------------------------------------

    def run(self, cues: list[dict], say, job_id: str | None = None) -> JobReport:
        with self.lock:                          # one job at a time per worker
            self.busy = True
            job_id = job_id or uuid.uuid4().hex[:8]
            report = JobReport(engine_id=self.engine_id, resident=not self._fresh,
                               device=self.info.get("device"), precision=self.info.get("precision"))
            if self._fresh:
                report.start_seconds, report.load_seconds = self.start_seconds, self.load_seconds
            by_id = {c["id"]: c for c in cues}
            self._job_lines, self._job_id = [], job_id
            self._job_end.clear()
            try:
                self._send({"job": job_id, "cues": cues})
                while True:
                    msg = self._read()
                    if "done" in msg:
                        seconds = msg.get("seconds")
                        report.cues[msg["done"]] = seconds
                        report.generate_seconds += float(seconds or 0.0)
                        if msg["done"] in by_id:
                            say(f"  render  {msg['done']}  {by_id[msg['done']]['prompt'][:52]}"
                                + (f"  {seconds:.1f}s" if isinstance(seconds, (int, float)) else ""))
                    elif msg.get("job_done") == job_id:
                        self.idle_gb = msg.get("reserved_gb", self.idle_gb)
                        report.peak_gb = msg.get("peak_gb")
                        if report.peak_gb:
                            self.run_peak_gb = max(self.run_peak_gb or 0.0, report.peak_gb)
                        break
                    # An "error" message is followed by the worker exiting; the next _read
                    # sees EOF and raises with the exit-code diagnosis.
                # Two pipes, no ordering guarantee between them: wait for the worker's marker
                # so this job's stderr is attributed to this job, not the next one.
                self._job_end.wait(2.0)
                for line in self._job_lines:
                    if line.lower().startswith("warning:"):
                        say(f"  {line.strip()}")
            finally:
                self._job_id = None
                self._fresh = False
                self.last_used = time.monotonic()
                self.busy = False
            missing = [c["id"] for c in cues if not os.path.exists(c["dest"])]
            if missing:
                raise RuntimeError(
                    f"{self.engine_id} finished without producing: {', '.join(missing)}\n"
                    f"{_tail(self.stderr_text())}"
                )
            report.generate_seconds = round(report.generate_seconds, 2)
            return report


# -------------------------------------------------------------------- the pool


class WorkerPool:
    def __init__(self, idle_seconds: float = IDLE_SECONDS, free_vram=None, reap_every: float = 30.0):
        self.idle_seconds = idle_seconds
        self._free_vram = free_vram if free_vram is not None else _free_vram_gb
        self._reap_every = reap_every
        self._workers: dict[tuple[str, str | None], EngineWorker] = {}
        self._lock = threading.RLock()
        self._reaper: threading.Thread | None = None
        self._closed = False

    def run(self, engine_id: str, cues: list[dict], device: str | None, say) -> JobReport:
        worker = self.get(engine_id, device, say)
        try:
            return worker.run(cues, say)
        except WorkerDied as exc:
            self._forget(worker)
            raise RuntimeError(str(exc)) from exc
        finally:
            if not self._may_stay(worker):
                self._retire(worker)

    def get(self, engine_id: str, device: str | None, say=None) -> EngineWorker:
        with self._lock:
            worker = self._workers.get((engine_id, device))
            if worker and worker.alive():
                if say:
                    say(f"  resident  {engine_id}")
                return worker
            if worker:                            # died between jobs: Task Manager, a driver reset...
                if say:
                    say(f"  {engine_id} worker had exited "
                        f"({_explain_exit(worker.proc.returncode if worker.proc else -1, worker.stderr_text())}); restarting")
                self._forget(worker)
            self._make_room(engine_id)
            worker = EngineWorker(engine_id, device)
            worker.start()                       # WorkerDied here carries the install/gating text
            self._workers[(engine_id, device)] = worker
            self._ensure_reaper()
            if say:
                holds = f"  holds {worker.idle_gb:.1f} GB" if worker.idle_gb else ""
                say(f"  loaded  {engine_id}  {worker.load_seconds:.1f}s  "
                    f"{worker.info.get('device', '?')} {worker.info.get('precision') or ''}".rstrip() + holds)
            return worker

    # -- rules ------------------------------------------------------------------

    def _may_stay(self, worker: EngineWorker) -> bool:
        if self.idle_seconds <= 0 or self._closed:
            return False
        free = self._free_vram()
        if free is None:                          # CPU, or no way to know: RAM is the limit
            return True
        return free >= KEEP_MARGIN_GB

    def _make_room(self, engine_id: str) -> None:
        """Retire idle workers, least recently used first, until the new one fits."""
        free = self._free_vram()
        if free is None:
            return
        need = peak_of(engine_id) + RUN_MARGIN_GB
        idle = sorted((w for w in self._workers.values() if not w.busy), key=lambda w: w.last_used)
        for worker in idle:
            if free >= need:
                break
            free += float(worker.idle_gb or peak_of(worker.engine_id))
            self._retire(worker)

    def can_run_concurrently(self, engine_ids: list[str], device: str | None = None) -> bool:
        if device == "cpu" or len(engine_ids) < 2:
            return False
        if os.environ.get("HUDKA_CONCURRENT_ENGINES", "1") == "0":
            return False
        free = self._free_vram()
        if free is None:
            return False
        credit = sum(float(w.idle_gb or 0.0) for (eid, _d), w in self._workers.items()
                     if eid in engine_ids and w.alive())
        return free + credit >= sum(peak_of(e) for e in engine_ids) + RUN_MARGIN_GB

    # -- housekeeping -------------------------------------------------------------

    def _retire(self, worker: EngineWorker) -> None:
        with self._lock:
            self._forget(worker)
        worker.stop()

    def _forget(self, worker: EngineWorker) -> None:
        with self._lock:
            for key, value in list(self._workers.items()):
                if value is worker:
                    del self._workers[key]

    def _ensure_reaper(self) -> None:
        if self._reaper and self._reaper.is_alive():
            return
        if self.idle_seconds <= 0:
            return

        def reap() -> None:
            while not self._closed:
                time.sleep(self._reap_every)
                now = time.monotonic()
                with self._lock:
                    stale = [w for w in self._workers.values()
                             if not w.busy and now - w.last_used > self.idle_seconds]
                for worker in stale:
                    self._retire(worker)

        self._reaper = threading.Thread(target=reap, daemon=True, name="hudka-worker-reaper")
        self._reaper.start()

    def release(self) -> int:
        """Unload every resident model now. Returns how many workers were stopped."""
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()
        return len(workers)

    def shutdown(self) -> None:
        self._closed = True
        self.release()

    def status(self) -> list[dict]:
        if not self._lock.acquire(timeout=0.2):   # never block a page behind a 20 s load
            return [{"engine": "(loading)", "busy": True}]
        try:
            now = time.monotonic()
            return [{
                "engine": w.engine_id, "pid": w.proc.pid if w.proc else None,
                "alive": w.alive(), "busy": w.busy,
                "resident_gb": w.idle_gb, "peak_gb": w.run_peak_gb or w.load_peak_gb,
                "idle_s": round(now - w.last_used, 1), "device": w.info.get("device"),
                "precision": w.info.get("precision"),
            } for w in self._workers.values()]
        finally:
            self._lock.release()


_POOL: WorkerPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> WorkerPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = WorkerPool()
            atexit.register(_POOL.shutdown)
        return _POOL
