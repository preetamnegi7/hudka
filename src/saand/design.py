"""Building a starting cue sheet from analysis alone.

This is the mechanical fallback for when nobody has looked at the footage. It knows
*where* things happen, never *what* they are, so its prompts are generic and meant to be
reworded. Claude reading the contact sheets does the real job.

The hard part is that "where things happen" means something completely different across
formats. Edited footage has hard cuts and large frame-to-frame differences. A screen
recording or a talking-head has **no cuts at all** and motion two orders of magnitude
lower - a fixed threshold tuned for one produces nothing on the other. So anchors come
from cuts when cuts exist, and from an *adaptive* percentile of the clip's own motion
distribution when they do not.
"""

from __future__ import annotations

import numpy as np

from . import presets
from .presets import Preset
from .schema import CURRENT_VERSION, BedCue, CueSheet, SfxCue, VideoInfo

#: Never place two scaffolded accents closer than this, or the mix turns to mush.
#: At 1.4s hits landed 2.25s apart and read as a stutter rather than as punctuation.
MIN_GAP_SECONDS = 2.0
#: Motion below this is camera noise or compression jitter, not an event.
MOTION_FLOOR = 0.0025


def speech_coverage(analysis: dict, duration: float) -> float:
    """Fraction of the clip carrying probable speech. Drives how busy the design can be."""
    if duration <= 0:
        return 0.0
    talked = sum(b - a for a, b in analysis.get("speech_ranges", []))
    return min(1.0, talked / duration)


def suggest_preset(info: VideoInfo, analysis: dict) -> str:
    """Pick a delivery preset from the video's shape and how much of it is narration.

    Speech coverage is the strongest signal available without watching the footage: a clip
    that is talking most of the time needs the music underneath it and the effects sparse,
    whatever its aspect ratio.
    """
    if info.height > info.width:
        return "short-form"
    if speech_coverage(analysis, info.duration) > 0.45:
        return "explainer"
    if info.duration > 90:
        return "cinematic"
    return "short-form"


def _motion_anchors(analysis: dict, want: int, avoid: list[float]) -> list[tuple[float, float]]:
    """Busiest moments in the clip, as (time, strength).

    The threshold is a percentile of this clip's own motion rather than a constant, so it
    adapts to footage whose absolute motion is tiny.
    """
    curve = analysis.get("motion_curve") or []
    if not curve or want <= 0:
        return []

    times = np.array([t for t, _ in curve])
    energy = np.array([e for _, e in curve])
    if energy.max() <= MOTION_FLOOR:
        return []

    # Take a generous candidate pool, then thin it by spacing rather than by threshold
    # alone - spacing is what actually keeps the result listenable.
    #
    # No max-fraction term here. `energy.max() * 0.25` used to sit alongside these and, on
    # real screen-recording footage, was 27x larger than the p90 value - so it decided
    # every selection and the adaptive percentile written for exactly that footage was
    # dead code. It passed 4 of 388 samples where the floor alone passes 10.
    threshold = max(float(np.percentile(energy, 90)), MOTION_FLOOR)
    candidates = [(float(t), float(e)) for t, e in zip(times, energy) if e >= threshold]
    candidates.sort(key=lambda pair: pair[1], reverse=True)

    picked: list[tuple[float, float]] = []
    for at, strength in candidates:
        if len(picked) >= want:
            break
        near = [p for p, _ in picked] + avoid
        if all(abs(at - other) >= MIN_GAP_SECONDS for other in near):
            picked.append((at, strength))
    return sorted(picked)


#: One-shot prompts by role. Every scaffolded effect used to carry one identical string,
#: so every effect sounded identical - which reads as "there are no effects" just as
#: surely as being too quiet does. Rotating within a class means a poor prompt spoils one
#: cue rather than all of them.
SFX_VOCAB = {
    "transition": [
        "soft airy whoosh transition, quick, dry, clean tail, no reverb wash",
        "short filtered air sweep with a soft low thump underneath, dry",
        "smooth panel transition swipe, soft and quick, no reverb",
    ],
    "accent": [
        "light UI panel slide, soft paper-like movement, quick, dry",
        "gentle soft-synth blip with a short felt decay, close",
        "muted wooden tap with a soft short body, dry",
    ],
    "foley": [
        "single soft interface click, tight transient, minimal tail, dry",
        "quick mechanical keyboard tap, close-miked, dry, no room",
        "faint soft selection tick, very short, dry",
    ],
}

#: Long enough to carry a tail, and safely clear of the sub-2-second range where this
#: model returns saturated audio about half the time. 1.8s - which is what a tail alone
#: would suggest - happens to be one of the lengths that fails.
SFX_DURATION_SECONDS = 2.0


def _classify(strength: float, loudest: float) -> str:
    """Which role an anchor plays, from how strong it is relative to the clip's biggest."""
    rel = strength / loudest if loudest > 0 else 0.0
    if rel >= 0.5:
        return "transition"
    return "accent" if rel >= 0.15 else "foley"


def scaffold(analysis: dict, *, preset: str | None = None,
             engine: str | None = None) -> CueSheet:
    """Build a starting cue sheet from `analysis.json` contents."""
    from . import engines as engine_registry

    info = VideoInfo.model_validate(analysis["video"])
    name = preset or suggest_preset(info, analysis)
    pre: Preset = presets.get(name)

    sfx_engine = engine or engine_registry.DEFAULT_ENGINES["sfx"]
    bed_engine = engine or engine_registry.pick_bed_engine(info.duration)
    # A bed longer than the chosen model can generate in one pass is looped with
    # equal-power crossfades rather than truncated into silence.
    bed_loops = info.duration > engine_registry.SMALL_MAX_SECONDS

    shots = analysis.get("shots", [])
    minutes = max(info.duration / 60.0, 0.05)
    coverage = speech_coverage(analysis, info.duration)

    # Density from the preset, pulled down when most of the clip is someone talking.
    low, high = pre.sfx_per_minute
    target = int(round(((low + high) / 2) * minutes))
    if coverage > 0.6:
        # Thin out under narration, but only somewhat. Level is handled separately by the
        # preset; cutting both density and gain for the same reason double-counts it, and
        # that double-count is what left four near-inaudible clicks in a 49s clip.
        target = int(round(target * 0.8))
    target = max(3, min(target, 60))

    cues: list[SfxCue] = []
    cut_times: list[float] = []

    for i, shot in enumerate(shots):
        if i == 0 or shot["start"] < 0.15:
            continue
        cut_times.append(round(shot["start"], 3))
        cues.append(SfxCue(
            id=f"cut{i:02d}", at=round(shot["start"], 3), duration=SFX_DURATION_SECONDS,
            prompt=SFX_VOCAB["transition"][len(cut_times) % len(SFX_VOCAB["transition"])],
            engine=sfx_engine,
            gain_db=pre.sfx_gain_db + presets.SFX_TRIM_DB["transition"],
            seed=1000 + i, shot=i, note="on a shot boundary",
        ))

    # With no cuts - a screen recording, a locked-off talking head - the only structure
    # available is where the picture actually changes.
    remaining = target - len(cues)
    anchors = _motion_anchors(analysis, remaining, cut_times)
    loudest = max((s for _, s in anchors), default=0.0)

    for n, (at, strength) in enumerate(anchors, start=1):
        if at < 0.3 or at > info.duration - 0.3:
            continue
        kind = _classify(strength, loudest)
        options = SFX_VOCAB[kind]
        cues.append(SfxCue(
            id=f"hit{n:02d}", at=round(at, 3), duration=SFX_DURATION_SECONDS,
            prompt=options[n % len(options)],
            engine=sfx_engine,
            # Gain comes from the preset plus a per-role trim. There is deliberately no
            # extra cut for speech coverage: density is already thinned above, and cutting
            # level for the same reason is what buried these.
            gain_db=pre.sfx_gain_db + presets.SFX_TRIM_DB[kind],
            seed=2000 + n, note=f"{kind}, activity peak {strength:.4f}",
        ))

    cues.sort(key=lambda c: c.at)

    # Naming tempo, key, instrumentation and register gives the model something to build
    # on; "calm pad, unobtrusive" gives it almost nothing, which is what "very basic"
    # sounded like. "wide stereo image, airy top end" keeps the bed clear of centre-panned
    # speech - a darker, more mono bed competes with the voice instead of sitting under it.
    bed_prompt = (
        "warm instrumental underscore at 82 BPM in A minor, soft Rhodes electric piano on "
        "a slow four-chord loop, sustained analog pad, light shaker and soft rim click, "
        "gentle upright bass on the root, wide stereo image, airy top end, no vocals, "
        "no lead melody, steady dynamics, sits under a voiceover"
        if coverage > 0.45 else
        "warm instrumental bed at 95 BPM in A minor, clean electric piano, muted guitar "
        "arpeggio, soft kick and shaker, gentle bass, wide stereo image, airy top end, "
        "no vocals, steady dynamics"
    )

    return CueSheet(
        version=CURRENT_VERSION,
        video=info, preset=pre.name, target_lufs=pre.target_lufs,
        true_peak_db=pre.true_peak_db, keep_original_audio=info.has_audio,
        music=[BedCue(
            id="bed", start=0.0, end=round(info.duration, 3), prompt=bed_prompt,
            engine=bed_engine, gain_db=pre.music_gain_db, fade_in=0.5,
            fade_out=min(2.0, info.duration / 4), duck=True, loop=bed_loops, seed=7,
            note="reword this to suit the footage",
        )],
        sfx=cues,
    )
