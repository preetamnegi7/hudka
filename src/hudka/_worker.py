"""Generation worker: one engine, one process, then exit.

Run as `python -m hudka._worker`, reading a JSON job from stdin.

Why a subprocess at all. Loading a second model into a process that has already loaded
and released one is unreliable here: it is pathologically slow at best, and at worst the
process dies outright - no exception, no traceback, just gone. That is survivable in a
CLI, but inside the GUI the generation thread shares the server's process, so a native
crash would take the whole app down with it and leave the user staring at a dead page.

Running each engine in its own short-lived process gives a clean CUDA context per model,
reclaims memory completely between engines, and turns a hard crash into a non-zero exit
code the parent can report properly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    job = json.loads(sys.stdin.read())
    engine_id: str = job["engine"]
    device: str | None = job.get("device")

    from .engines import build
    from .engines.base import GenerateRequest

    engine = build(engine_id, device)

    for item in job["cues"]:
        req = GenerateRequest(
            prompt=item["prompt"],
            duration=item["duration"],
            seed=item["seed"],
            video=Path(item["video"]) if item.get("video") else None,
            window=tuple(item["window"]) if item.get("window") else None,
            extra=item.get("extra") or {},
        )
        engine.generate(req, Path(item["dest"]))
        # One line per finished cue, so the parent can report progress live.
        print(json.dumps({"done": item["id"]}), flush=True)

    engine.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
