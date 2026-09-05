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

import hashlib
from dataclasses import dataclass

import numpy as np

from . import presets
from .presets import Preset
from .schema import CURRENT_VERSION, BedCue, CueSheet, SfxCue, VideoInfo

#: Never place two scaffolded accents closer than this, or the mix turns to mush.
#: At 1.4s hits landed 2.25s apart and read as a stutter rather than as punctuation.
MIN_GAP_SECONDS = 2.0
#: Motion below this is camera noise or compression jitter, not an event.
MOTION_FLOOR = 0.0025
#: Speech coverage above which the design drops to the preset's minimum effect density.
HEAVY_NARRATION = 0.9


#: Speech coverage above which the bed has to sit under a voice rather than carry the
#: clip: no lead melody, steady dynamics, and room left in the centre.
BED_NARRATION = 0.45


@dataclass(frozen=True)
class BedTemplate:
    """One candidate underscore.

    Naming tempo, key, instrumentation and register gives the model something to build on;
    "calm pad, unobtrusive" gives it almost nothing, which is what "very basic" sounded
    like. Every prompt keeps a wide stereo image and an airy top end so the bed stays
    clear of centre-panned speech - a darker, more mono bed competes with the voice
    instead of sitting under it.
    """

    mood: str
    #: Written to sit under a voiceover: no lead melody, steady dynamics.
    narration: bool
    pace: str          # "slow" | "medium" | "fast"
    prompt: str


MUSIC_BEDS: tuple[BedTemplate, ...] = (
    BedTemplate("warm rhodes", True, "slow",
        "warm instrumental underscore at 72 BPM in A minor, soft Rhodes electric piano on "
        "a slow four-chord loop, sustained analog pad, light shaker and soft rim click, "
        "gentle upright bass on the root, wide stereo image, airy top end, no vocals, "
        "no lead melody, steady dynamics, sits under a voiceover"),
    BedTemplate("felt piano", True, "slow",
        "calm instrumental underscore at 68 BPM in D minor, felt piano with soft hammers, "
        "warm tape pad underneath, brushed snare pulse, low sine bass, wide stereo image, "
        "airy top end, no vocals, no lead melody, very steady dynamics, sits under a voiceover"),
    BedTemplate("muted keys", True, "medium",
        "warm instrumental underscore at 84 BPM in F major, muted electric piano on a "
        "four-chord loop, sustained pad, soft shaker and light rim click, round bass on "
        "the root, wide stereo image, airy top end, no vocals, no lead melody, steady "
        "dynamics, sits under a voiceover"),
    BedTemplate("marimba pulse", True, "medium",
        "clean instrumental underscore at 90 BPM in C minor, short marimba pulse, soft "
        "analog pad, closed hi-hat and light kick, simple sub bass, wide stereo image, "
        "airy top end, no vocals, no lead melody, steady dynamics, sits under a voiceover"),
    BedTemplate("driving guitar", True, "fast",
        "driving instrumental underscore at 104 BPM in G minor, muted guitar arpeggio, "
        "tight analog pad, soft kick and closed hi-hat, steady eighth-note bass, wide "
        "stereo image, airy top end, no vocals, no lead melody, steady dynamics, sits "
        "under a voiceover"),
    BedTemplate("plucked pulse", True, "fast",
        "energetic instrumental underscore at 112 BPM in E minor, plucked synth pulse on "
        "a two-bar loop, warm pad, light kick and shaker, steady bass, wide stereo image, "
        "airy top end, no vocals, no lead melody, steady dynamics, sits under a voiceover"),
    BedTemplate("slow strings", False, "slow",
        "cinematic instrumental bed at 64 BPM in C minor, sustained strings, low drone, "
        "sparse piano notes, soft timpani pulse, wide stereo image, airy top end, "
        "no vocals, slow swells"),
    BedTemplate("mallets and pad", False, "slow",
        "warm instrumental bed at 70 BPM in D minor, felt piano, slow string pad swell, "
        "soft mallet accents, deep round bass, wide stereo image, airy top end, no vocals, "
        "gentle dynamics"),
    BedTemplate("clean keys", False, "medium",
        "warm instrumental bed at 95 BPM in A minor, clean electric piano, muted guitar "
        "arpeggio, soft kick and shaker, gentle bass, wide stereo image, airy top end, "
        "no vocals, steady dynamics"),
    BedTemplate("dusty loop", False, "medium",
        "relaxed instrumental bed at 88 BPM in E minor, lo-fi electric piano with light "
        "wow and flutter, dusty drum loop, warm sub bass, soft vinyl noise, wide stereo "
        "image, airy top end, no vocals"),
    BedTemplate("bright plucks", False, "fast",
        "upbeat instrumental bed at 118 BPM in G major, bright plucked synth, muted guitar "
        "chops, punchy kick and clap, driving bass, wide stereo image, airy top end, "
        "no vocals, steady dynamics"),
    BedTemplate("arp drive", False, "fast",
        "energetic instrumental bed at 126 BPM in F minor, arpeggiated synth, filtered pad "
        "sweep, four-on-the-floor kick and hats, rolling bass, wide stereo image, airy top "
        "end, no vocals"),
)


def _motion_energy(analysis: dict) -> float:
    """The clip's 90th-percentile motion, in the same units as MOTION_FLOOR.

    The median is useless here - a screen recording sits at almost zero for most of its
    length - but the p90 separates a still talking-head from a busy edit by two orders of
    magnitude on real footage.
    """
    curve = analysis.get("motion_curve") or []
    values = [float(p[1]) for p in curve if isinstance(p, (list, tuple)) and len(p) > 1]
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 90))


def bed_pace(analysis: dict, duration: float) -> str:
    """How busy the picture is, as one of three words the bed library is keyed on."""
    minutes = max(duration / 60.0, 0.05)
    cuts_per_minute = len(analysis.get("shots") or []) / minutes
    energy = _motion_energy(analysis)
    if cuts_per_minute >= 10.0 or energy >= 8 * MOTION_FLOOR:
        return "fast"
    if cuts_per_minute <= 3.0 and energy < MOTION_FLOOR:
        return "slow"
    return "medium"


def pick_bed(info: VideoInfo, analysis: dict, coverage: float) -> tuple[BedTemplate, int]:
    """Choose an underscore, and a seed, that differ from project to project.

    Both used to be constants: one of two prompt strings picked by a single boolean, and
    `seed=7`. Prompt, seed and engine are the whole cache key, so every video of the same
    length got byte-identical music and every video whatever its length got the same key,
    tempo and instruments. Three real projects on this machine had the same prompt hash
    and the same seed.

    The choice is a hash of the source path and the clip's own shape, so it is stable -
    re-scaffolding the same video gives the same bed - while two projects differ even when
    they come from the same file. The seed lands in cues.json either way, so a render
    stays reproducible from the sheet alone.
    """
    pace = bed_pace(analysis, info.duration)
    narration = coverage > BED_NARRATION
    pool = [b for b in MUSIC_BEDS if b.narration is narration and b.pace == pace]
    if not pool:  # pragma: no cover - every (narration, pace) pair is populated
        pool = [b for b in MUSIC_BEDS if b.narration is narration]

    signature = "\x1f".join([
        str(info.path), f"{info.duration:.2f}",
        str(len(analysis.get("shots") or [])), f"{coverage:.3f}",
    ])
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    chosen = pool[int(digest[:8], 16) % len(pool)]
    # Not `hash()`: it is salted per process, so the same video would pick a different
    # bed on every run and re-scaffolding would silently replace the music.
    seed = 1000 + int(digest[8:16], 16) % 90000
    return chosen, seed


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

    from .engines import hardware

    hw = hardware.detect()
    sfx_engine = engine or engine_registry.DEFAULT_ENGINES["sfx"]
    bed_engine = engine or engine_registry.pick_bed_engine(info.duration, hw)
    # A bed longer than the CHOSEN model can generate in one pass is looped with
    # equal-power crossfades rather than truncated into silence. Medium covers 380 s, so a
    # 200 s bed on it is generated whole - and `loop: false` keeps the "covers" QA check
    # honest about what was asked for.
    bed_loops = info.duration > engine_registry.max_seconds(bed_engine)

    shots = analysis.get("shots", [])
    minutes = max(info.duration / 60.0, 0.05)
    coverage = speech_coverage(analysis, info.duration)

    # Density from the preset, pulled down when most of the clip is someone talking.
    low, high = pre.sfx_per_minute
    if coverage > HEAVY_NARRATION:
        # Someone is talking almost the whole time: aim at the preset's *low* end. The
        # midpoint scaled by 0.8 put 35 effects under a 3.6-minute narration, most of
        # them cursor jitter given a rotating "keyboard tap" prompt.
        target = int(round(low * minutes))
    else:
        target = int(round(((low + high) / 2) * minutes))
        if coverage > 0.6:
            # Thin out under narration, but only somewhat. Level is handled separately by
            # the preset; cutting both density and gain for the same reason double-counts
            # it, and that double-count is what left four near-inaudible clicks in a 49s
            # clip.
            target = int(round(target * 0.8))
    target = max(3, min(target, 60))

    cues: list[SfxCue] = []
    cut_times: list[float] = []

    for i, shot in enumerate(shots):
        if i == 0 or shot["start"] < 0.15:
            continue
        # One transition per *moment*, not per boundary. Two page loads a second apart
        # were each given a whoosh (cut04 at 31.9s, cut05 at 32.9s on a real clip), which
        # reads as a stutter. The sound marks the first change; a flash frame's exit a
        # moment later is suppressed by the same spacing rule.
        if cut_times and shot["start"] - cut_times[-1] < MIN_GAP_SECONDS:
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

    bed, bed_seed = pick_bed(info, analysis, coverage)

    return CueSheet(
        version=CURRENT_VERSION,
        video=info, preset=pre.name, target_lufs=pre.target_lufs,
        true_peak_db=pre.true_peak_db, keep_original_audio=info.has_audio,
        music=[BedCue(
            id="bed", start=0.0, end=round(info.duration, 3), prompt=bed.prompt,
            engine=bed_engine, gain_db=pre.music_gain_db, fade_in=0.5,
            fade_out=min(2.0, info.duration / 4), duck=True, loop=bed_loops, seed=bed_seed,
            note=f"{bed.mood} - reword this to suit the footage, or re-roll for another take",
        )],
        sfx=cues,
    )
