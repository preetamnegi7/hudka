"""Resident engine workers, through real pipes and real processes on the silence stub.

The pool's whole job is process lifetime, so nothing here is faked at the boundary: a real
child speaks the protocol, a real parent dies, real pipes fill up.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hudka.engines import pool as pool_mod

SRC = str(Path(__file__).resolve().parents[1] / "src")


def cue(dest: Path, cue_id: str = "x", duration: float = 1.0) -> dict:
    return {"id": cue_id, "prompt": "p", "duration": duration, "seed": 1, "dest": str(dest),
            "video": None, "window": [0, duration], "extra": {}}


@pytest.fixture
def fresh_pool():
    """Never the process-wide singleton: these tests kill workers on purpose."""
    pool = pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: None, reap_every=0.2)
    yield pool
    pool.shutdown()


def wait_until(pred, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


class TestResidency:
    def test_a_worker_is_reused_across_jobs(self, fresh_pool, tmp_path):
        log: list[str] = []
        first = fresh_pool.run("silence", [cue(tmp_path / "a.wav", "a")], None, log.append)
        pid = fresh_pool.get("silence", None).proc.pid
        second = fresh_pool.run("silence", [cue(tmp_path / "b.wav", "b")], None, log.append)

        assert fresh_pool.get("silence", None).proc.pid == pid, "the same process served both"
        assert first.resident is False and second.resident is True
        assert second.load_seconds == 0.0, "a resident worker has nothing to load"
        assert sum("loaded  silence" in line for line in log) == 1
        assert any("resident  silence" in line for line in log)
        assert (tmp_path / "a.wav").exists() and (tmp_path / "b.wav").exists()
        assert set(first.cues) == {"a"} and set(second.cues) == {"b"}

    def test_an_idle_worker_unloads_and_exits_cleanly(self, tmp_path):
        pool = pool_mod.WorkerPool(idle_seconds=0.5, free_vram=lambda: None, reap_every=0.1)
        try:
            pool.run("silence", [cue(tmp_path / "a.wav")], None, lambda _: None)
            worker = pool.get("silence", None)
            proc = worker.proc
            assert wait_until(lambda: proc.poll() is not None, timeout=10.0), "the reaper never fired"
            assert proc.returncode == 0, "a `quit` is a clean exit, not a kill"
            assert pool.status() == []
        finally:
            pool.shutdown()

    def test_idle_zero_restores_one_process_per_job(self, tmp_path):
        pool = pool_mod.WorkerPool(idle_seconds=0.0, free_vram=lambda: None)
        try:
            log: list[str] = []
            pool.run("silence", [cue(tmp_path / "a.wav", "a")], None, log.append)
            pool.run("silence", [cue(tmp_path / "b.wav", "b")], None, log.append)
            assert sum("loaded  silence" in line for line in log) == 2
            assert not any("resident" in line for line in log)
        finally:
            pool.shutdown()

    def test_release_stops_everything(self, fresh_pool, tmp_path):
        fresh_pool.run("silence", [cue(tmp_path / "a.wav")], None, lambda _: None)
        proc = fresh_pool.get("silence", None).proc
        assert fresh_pool.release() == 1
        assert wait_until(lambda: proc.poll() is not None, timeout=10.0)
        assert fresh_pool.status() == []


class TestPressure:
    def test_an_idle_worker_is_retired_when_the_desktop_needs_the_card(self, tmp_path):
        """The keep rule runs when a job ends; the desktop keeps moving afterwards. Seen on
        the development machine: 2.4 GB free after a render, 0.3 GB minutes later with a
        worker still resident. The reaper must hand the card back under pressure."""
        free = {"gb": 5.0}
        pool = pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: free["gb"], reap_every=0.1)
        try:
            pool.run("silence", [cue(tmp_path / "a.wav")], None, lambda _: None)
            proc = pool.get("silence", None).proc
            time.sleep(0.4)
            assert proc.poll() is None, "plenty free: it stays"
            free["gb"] = 0.4                     # a game or an editor just took the card
            assert wait_until(lambda: proc.poll() is not None, timeout=10.0), \
                "the reaper did not give the card back"
            assert pool.status() == []
        finally:
            pool.shutdown()


class TestCrashes:
    def test_a_worker_that_died_between_jobs_is_diagnosed_and_replaced(self, fresh_pool, tmp_path):
        log: list[str] = []
        fresh_pool.run("silence", [cue(tmp_path / "a.wav", "a")], None, log.append)
        old = fresh_pool.get("silence", None).proc
        old.kill()
        old.wait()

        report = fresh_pool.run("silence", [cue(tmp_path / "b.wav", "b")], None, log.append)
        assert any("worker had exited" in line for line in log)
        assert fresh_pool.get("silence", None).proc.pid != old.pid
        assert report.resident is False and (tmp_path / "b.wav").exists()

    def test_a_worker_dying_mid_job_raises_the_diagnosis(self, monkeypatch, tmp_path):
        """A child that answers the handshake, takes the job, complains about VRAM to stderr
        and dies: the render must fail with the VRAM explanation, and the pool must forget it."""
        script = textwrap.dedent("""
            import sys, json, os
            hello = json.loads(sys.stdin.readline())
            print(json.dumps({"ready": hello["engine"], "pid": os.getpid()}), flush=True)
            print(json.dumps({"loaded": hello["engine"], "seconds": 0.0, "device": "cpu"}), flush=True)
            sys.stdin.readline()
            sys.stderr.write("RuntimeError: CUDA out of memory. Tried to allocate 2 GB\\n")
            sys.stderr.flush()
            os._exit(1)
        """)
        real_popen = subprocess.Popen
        monkeypatch.setattr(pool_mod.subprocess, "Popen",
                            lambda args, **kw: real_popen([sys.executable, "-c", script], **kw))
        pool = pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: None)
        try:
            with pytest.raises(RuntimeError) as err:
                pool.run("silence", [cue(tmp_path / "a.wav")], None, lambda _: None)
            assert "VRAM" in str(err.value)
            assert pool.status() == [], "a dead worker must not be offered again"
        finally:
            pool.shutdown()

    def test_children_die_with_the_parent(self, tmp_path):
        """The parent exits without cleanup (os._exit, no atexit): the Job Object or the
        worker's own watchdog must still take the worker down."""
        script = textwrap.dedent(f"""
            import os, sys
            sys.path.insert(0, {SRC!r})
            from hudka.engines import pool as pool_mod
            pool = pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: None)
            worker = pool.get("silence", None)
            print(worker.proc.pid, flush=True)
            os._exit(3)
        """)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
        assert out.returncode == 3, out.stderr[-800:]
        child_pid = int(out.stdout.strip().splitlines()[-1])
        assert wait_until(lambda: not pool_mod.pid_alive(child_pid), timeout=15.0), \
            f"worker {child_pid} outlived its parent"


class TestPipes:
    def test_warning_lines_are_attributed_to_the_job(self, monkeypatch, tmp_path):
        """An engine's `warning:` on stderr reaches the log - after the marker, so it is
        this job's warning and not the next one's."""
        dest = tmp_path / "x.wav"
        script = textwrap.dedent("""
            import sys, json, wave, os
            hello = json.loads(sys.stdin.readline())
            print(json.dumps({"ready": hello["engine"], "pid": os.getpid()}), flush=True)
            print(json.dumps({"loaded": hello["engine"], "seconds": 0.0, "device": "cpu"}), flush=True)
            for line in sys.stdin:
                job = json.loads(line)
                if "quit" in job: break
                for c in job["cues"]:
                    w = wave.open(c["dest"], "wb"); w.setnchannels(2); w.setsampwidth(2)
                    w.setframerate(44100); w.writeframes(bytes([0, 16]) * 88200); w.close()
                    print(json.dumps({"done": c["id"], "seconds": 0.01}), flush=True)
                sys.stderr.write("warning: test engine complained\\n")
                sys.stderr.write("--hudka job-end " + job["job"] + "\\n"); sys.stderr.flush()
                print(json.dumps({"job_done": job["job"], "seconds": 0.02}), flush=True)
        """)
        real_popen = subprocess.Popen
        monkeypatch.setattr(pool_mod.subprocess, "Popen",
                            lambda args, **kw: real_popen([sys.executable, "-c", script], **kw))
        pool = pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: None)
        try:
            log: list[str] = []
            report = pool.run("silence", [cue(dest)], None, log.append)
            assert any("test engine complained" in line for line in log)
            assert report.cues == {"x": 0.01} and dest.exists()
        finally:
            pool.shutdown()

    def test_a_chatty_worker_does_not_deadlock(self, monkeypatch, tmp_path):
        """2 MB to stderr before anything else: without a concurrent drain the child blocks
        on the write, the parent on the read, and the render never ends. It happened on a
        36-cue project."""
        dest = tmp_path / "x.wav"
        script = textwrap.dedent("""
            import sys, json, wave, os
            hello = json.loads(sys.stdin.readline())
            print(json.dumps({"ready": hello["engine"], "pid": os.getpid()}), flush=True)
            print(json.dumps({"loaded": hello["engine"], "seconds": 0.0, "device": "cpu"}), flush=True)
            for line in sys.stdin:
                job = json.loads(line)
                if "quit" in job: break
                sys.stderr.write("x" * 2_000_000); sys.stderr.flush()
                for c in job["cues"]:
                    w = wave.open(c["dest"], "wb"); w.setnchannels(2); w.setsampwidth(2)
                    w.setframerate(44100); w.writeframes(bytes([0, 16]) * 88200); w.close()
                    print(json.dumps({"done": c["id"], "seconds": 0.01}), flush=True)
                sys.stderr.write("\\n--hudka job-end " + job["job"] + "\\n"); sys.stderr.flush()
                print(json.dumps({"job_done": job["job"], "seconds": 0.02}), flush=True)
        """)
        real_popen = subprocess.Popen
        monkeypatch.setattr(pool_mod.subprocess, "Popen",
                            lambda args, **kw: real_popen([sys.executable, "-c", script], **kw))
        pool = pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: None)
        try:
            import threading

            log: list[str] = []
            outcome: dict = {}

            def run():
                try:
                    pool.run("silence", [cue(dest)], None, log.append)
                    outcome["ok"] = True
                except Exception as exc:  # pragma: no cover - surfaces in the assertion
                    outcome["error"] = exc

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            worker.join(timeout=60)
            assert not worker.is_alive(), "worker deadlocked on a full stderr pipe"
            assert outcome.get("ok"), outcome.get("error")
            assert any("render  x" in line for line in log)
        finally:
            pool.shutdown()


class TestVramRules:
    def test_two_smalls_may_run_together_medium_plus_small_may_not(self):
        pool = pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: 7.3)
        assert pool.can_run_concurrently(["stable-audio-3-small-music", "stable-audio-3-small-sfx"])
        assert not pool.can_run_concurrently(["stable-audio-3-medium", "stable-audio-3-small-sfx"])
        assert not pool.can_run_concurrently(["stable-audio-3-small-music"]), "one engine is not concurrency"
        assert not pool.can_run_concurrently(["stable-audio-3-small-music", "stable-audio-3-small-sfx"], device="cpu")

    def test_no_gpu_means_no_concurrency_and_unlimited_residency(self):
        pool = pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: None)
        assert not pool.can_run_concurrently(["stable-audio-3-small-music", "stable-audio-3-small-sfx"])
        worker = pool_mod.EngineWorker("silence", None)
        assert pool._may_stay(worker), "RAM, not VRAM, is the limit on a CPU machine"

    def test_a_worker_may_stay_only_while_the_desktop_keeps_a_gigabyte(self):
        worker = pool_mod.EngineWorker("silence", None)
        assert pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: 1.5)._may_stay(worker)
        assert not pool_mod.WorkerPool(idle_seconds=600.0, free_vram=lambda: 0.6)._may_stay(worker)

    def test_exit_codes_are_still_explained(self):
        assert "external" in pool_mod._explain_exit(0xC0000006, "").lower()
        assert "VRAM" in pool_mod._explain_exit(1, "CUDA out of memory")
        assert "crashed while generating" in pool_mod._worker_error("e", 5, "")
        assert "failed (exit code 1)" in pool_mod._worker_error("e", 1, "Traceback (most recent call last)\nboom")
