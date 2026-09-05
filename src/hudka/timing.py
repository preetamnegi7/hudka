"""Wall-clock per stage, one shape for render and generate-only.

Nothing in the render path measured time before this, so a 20 s model load and a 13 s
torch.compile warm-up were paid on every render with nothing on screen to say so. The dict
drops straight into render_report.json; `summary` is formatted once here and never in the
page - one implementation, the same rule the cache key follows.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

#: The stages whose seconds add up to the whole render. `load`, `generate` and `cues` are
#: parts of `engines`, not additions to it.
STAGES = ("engines", "dialogue", "place", "balance", "mix", "master", "mux", "provenance")


class Timings(dict):
    """stage -> seconds. A plain dict, so it serialises as it is.

    `begin`/`end` bracket a block of existing code without re-indenting it; `timed` wraps a
    single call. Both accumulate, so a stage that runs in two places adds up.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._starts: dict[str, float] = {}

    def timed(self, label: str, fn: Callable[[], T]) -> T:
        started = time.perf_counter()
        try:
            return fn()
        finally:
            self._add(label, time.perf_counter() - started)

    def begin(self, label: str) -> None:
        self._starts[label] = time.perf_counter()

    def end(self, label: str) -> None:
        started = self._starts.pop(label, None)
        if started is not None:
            self._add(label, time.perf_counter() - started)

    def total(self) -> float:
        return round(sum(float(self.get(k, 0.0)) for k in STAGES), 2)

    def _add(self, label: str, seconds: float) -> None:
        self[label] = round(float(self.get(label, 0.0)) + seconds, 2)


def summary(t: dict) -> str:
    """'engines 21.3s (load 13.0s) · place 1.3s · mix 0.3s · master 7.7s · mux 8.3s · total 45s'"""
    bits = []
    if "engines" in t:
        load = float(t.get("load") or 0.0)
        note = f"load {load:.1f}s" + (", resident" if t.get("resident") else "")
        bits.append(f"engines {float(t['engines']):.1f}s ({note})")
    for key in ("place", "mix", "master", "mux"):
        if key in t:
            bits.append(f"{key} {float(t[key]):.1f}s")
    if "total" in t:
        bits.append(f"total {float(t['total']):.0f}s")
    return " · ".join(bits)
