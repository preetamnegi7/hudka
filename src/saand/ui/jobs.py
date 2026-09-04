"""Background jobs for the GUI.

Analysis takes seconds and rendering can take minutes, so neither can run inside a
request. Jobs run on a single worker thread and stream their progress lines back to the
page, which polls for them.

The worker pool is deliberately **one thread**: generation loads models into 12GB of
VRAM, and two renders at once would run the card out of memory. Queueing is the correct
behaviour here, not a limitation.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable


@dataclass
class Job:
    id: str
    kind: str
    project: str
    status: str = "queued"  # queued | running | done | error
    log: list[str] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started: float = field(default_factory=time)
    finished: float | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "project": self.project,
            "status": self.status,
            "log": self.log,
            "result": self.result,
            "error": self.error,
            "elapsed": round((self.finished or time()) - self.started, 1),
        }


class JobRunner:
    """Runs one job at a time and keeps their logs for the page to poll."""

    def __init__(self, history: int = 40):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="saand-job")
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._history = history

    def submit(self, kind: str, project: str,
               work: Callable[[Callable[[str], None]], dict | None]) -> Job:
        """Queue `work`, handing it a `say(line)` callback for progress."""
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, project=project)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            # Keep memory bounded across a long session.
            while len(self._order) > self._history:
                self._jobs.pop(self._order.pop(0), None)

        def run() -> None:
            job.status = "running"
            try:
                job.result = work(lambda line: job.log.append(str(line))) or {}
                job.status = "done"
            except Exception as exc:
                job.status = "error"
                # First line for the page, full trace to the log for debugging.
                job.error = str(exc) or exc.__class__.__name__
                job.log.append(traceback.format_exc())
            finally:
                job.finished = time()

        self._pool.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def active_for(self, project: str) -> Job | None:
        """The running or queued job for a project, if any."""
        with self._lock:
            for job_id in reversed(self._order):
                job = self._jobs.get(job_id)
                if job and job.project == project and job.status in ("queued", "running"):
                    return job
        return None

    def recent(self, limit: int = 10) -> list[dict]:
        with self._lock:
            ids = self._order[-limit:]
        return [self._jobs[i].as_dict() for i in reversed(ids) if i in self._jobs]
