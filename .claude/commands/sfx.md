---
description: Generate copyright-clean sound effects and background music for a video
argument-hint: <video path> [preset: short-form|cinematic|gameplay|explainer]
allowed-tools: Bash(*), Read, Write, Edit, Glob
---

Design and generate a soundtrack for: **$ARGUMENTS**

You are the video-understanding stage of this pipeline. The tool handles everything
deterministic; your job is the part that needs judgement — deciding what should be heard,
and exactly when.

## 1. Analyse

```bash
.venv/Scripts/python.exe -m hudka.cli analyze "<video>"
```

Note the project directory it reports (`out/<name>`).

## 2. Look at the footage

Read **every** contact sheet in `out/<name>/contact/` with the Read tool. Each tile is
stamped with its shot index and exact timecode — that is how you turn "the door closes" into
`at: 4.28`.

Then read `out/<name>/analysis.json` for:

- `shots` — boundaries, plus `motion` and `peak_at` (the busiest moment in each shot)
- `speech_ranges` — where the source already has speech; the bed must duck here
- `video.has_dialogue` — if true, keep the original audio and stay out of its way

## 3. Write the cue sheet

Write `out/<name>/cues.json` matching `src/hudka/schema.py`. Base it on what you actually
see, not on a template.

Pick a preset (`hudka presets` lists them; default `short-form`) and follow its guidance for
cue density, gain and ducking. Set `target_lufs`/`true_peak_db` from the preset.

**Engines** — default to these; they are cleared for commercial use worldwide:

- `stable-audio-3-small-sfx` — one-shots: impacts, whooshes, clicks, footsteps
- `stable-audio-3-medium` — music beds and long ambience (up to 380s)

Never use `hunyuan-foley` or `mmaudio` unless the user explicitly asks. The first needs an
opt-in flag (its licence excludes the EU/UK/South Korea); the second is non-commercial and
is refused outright.

**Writing prompts.** These go to an audio model, so describe the *sound*, not the picture.
"heavy wooden door slamming shut, short reverb tail" works; "the door closes" does not. Name
the material, the size and the space. For music, give genre, instrumentation, tempo feel and
mood, and say `instrumental, no vocals` — a vocal line under a talking video is unusable.

**Timing.** Put effects on the frame the event happens, not the shot start. Use `peak_at`
from a shot to find its strongest moment. Leave `align_transient: true` so the hit lands on
the beat rather than the file merely starting there.

**Levels.** Effects around −6 to −12 dB, music at the preset's `music_gain_db`, ambience
−22 to −30 dB. Give every cue a distinct `seed`.

Keep ids short and meaningful (`door01`, `whoosh02`) — they name the cached stem files.

## 4. Render

```bash
.venv/Scripts/python.exe -m hudka.cli render "out/<name>"
```

Report the measured LUFS and peak. If it says the loudness is off target, say so rather than
glossing over it.

## 5. Hand it back

Tell the user:

- where `final.mp4` is
- what you designed and why — the reasoning behind the choices, briefly
- measured loudness, and the licence report at `out/<name>/LICENSE-REPORT.md`
- that `hudka ui out/<name>` opens the audition page to re-roll individual cues

If they want changes, edit `cues.json` and re-render — the content-hash cache regenerates
only the cues that actually changed, so iterating is cheap.
