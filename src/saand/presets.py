"""Delivery presets. These set loudness, cue density and how hard music gets out of the way.

Targets follow normal streaming practice: -14 LUFS integrated with a -1 dBTP ceiling is
what YouTube, Spotify and most social platforms normalise toward, so mastering to it
means the platform leaves the mix alone instead of turning it down.

**Gains here are offsets from a normalised reference, not raw attenuation.** Every stem
is measured and brought to a known level before its cue gain applies - beds to
-20 LUFS, one-shots to -12 dBFS peak. Before that existed these numbers were offsets
from whatever the model happened to produce, which is how a bed ended up 27 dB under the
dialogue and inaudible.

So `music_gain_db = -7` means "7 dB below the bed reference", and with the bus anchored to
the measured dialogue the separation from a nominal -16 LUFS voice is a predictable
`music_gain_db - 4` LU.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The dialogue level the preset table is written against. Buses are anchored to the
#: *measured* dialogue, so a source recorded hotter or quieter than this still lands at
#: the intended separation instead of drifting with it.
NOMINAL_DIALOGUE_LUFS = -16.0

#: Per-kind trim on top of a preset's `sfx_gain_db`. A transition covering a cut has to
#: carry; a background foley tick should not compete with speech.
SFX_TRIM_DB = {
    "transition": 2.0,
    "accent": -2.0,
    "foley": -5.0,
}


@dataclass(frozen=True)
class Preset:
    name: str
    target_lufs: float
    true_peak_db: float
    #: Rough guide for how many SFX cues per minute suit this format.
    sfx_per_minute: tuple[int, int]
    #: dB relative to the bed reference (-20 LUFS).
    music_gain_db: float
    ambience_gain_db: float
    #: dB relative to the one-shot reference (-12 dBFS peak).
    sfx_gain_db: float
    #: How much the music bed actually drops while speech is present, in dB. This is a
    #: measured delivered depth, not a knob - see `mix._duck_params`.
    duck_depth_db: float
    description: str
    guidance: str


PRESETS: dict[str, Preset] = {
    "short-form": Preset(
        name="short-form",
        target_lufs=-14.0,
        true_peak_db=-1.0,
        sfx_per_minute=(20, 45),
        music_gain_db=-2.0,
        ambience_gain_db=-12.0,
        sfx_gain_db=5.0,
        duck_depth_db=-4.0,
        description="Reels / Shorts / TikTok",
        guidance=(
            "Dense and punchy. Put a whoosh or impact on essentially every cut, accent "
            "on-screen motion, and keep the music forward - it is carrying the energy, not "
            "sitting underneath. Ducking is light so the bed never disappears."
        ),
    ),
    "cinematic": Preset(
        name="cinematic",
        target_lufs=-16.0,
        true_peak_db=-1.0,
        sfx_per_minute=(4, 12),
        music_gain_db=-9.0,
        ambience_gain_db=-4.0,
        sfx_gain_db=2.0,
        duck_depth_db=-8.0,
        description="Film, trailers, narrative",
        guidance=(
            "Sparse and deliberate. Ambience beds do most of the work - they are the "
            "loudest layer in this preset - so reserve effects for moments that warrant "
            "them. Preserve dynamic range; quiet passages should stay quiet. Music ducks "
            "hardest here."
        ),
    ),
    "gameplay": Preset(
        name="gameplay",
        target_lufs=-14.0,
        true_peak_db=-1.0,
        sfx_per_minute=(25, 60),
        music_gain_db=-5.0,
        ambience_gain_db=-9.0,
        sfx_gain_db=3.0,
        duck_depth_db=-5.0,
        description="Gameplay / screen capture",
        guidance=(
            "UI clicks, hits and notifications on interactions; whooshes on transitions. "
            "Music sits low and continuous under the action rather than following the cuts."
        ),
    ),
    "explainer": Preset(
        name="explainer",
        target_lufs=-14.0,
        true_peak_db=-1.0,
        sfx_per_minute=(6, 18),
        music_gain_db=-7.0,
        ambience_gain_db=-12.0,
        sfx_gain_db=1.0,
        duck_depth_db=-6.0,
        description="Product demos, ads, tutorials",
        guidance=(
            "Subtle and clean. Light UI foley on state changes and reveals, a calm bed "
            "underneath, and steady ducking so the voiceover always wins. Nothing should "
            "pull attention away from what is being explained."
        ),
    ),
}

DEFAULT = "short-form"


def get(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; choose one of: {', '.join(sorted(PRESETS))}"
        ) from None
