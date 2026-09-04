<img src="assets/hudka.svg" width="76" align="left" alt="" hspace="14" vspace="4">

# Hudka

**AI sound design from video, running entirely on your own machine.**

<br clear="left">

Point it at a clip. It finds the shots and the moments that matter, generates sound
effects and a music bed to match, mixes them under any existing dialogue, masters to a
streaming loudness target, and puts the audio back on the picture.

No cloud. No API keys. No per-render cost. No subscription. Every sound is generated
locally, and every render ships a licence report saying exactly what you are allowed to
do with it.

---

Named for the *hudka* — the small waisted hand drum carried by Garhwali storytellers,
held under rope tension and played with the fingers.

---

## Why this exists

The model most tutorials recommend for video-to-audio is **MMAudio**. Its *code* is MIT,
so people reasonably assume the whole thing is permissive.

**Its weights are CC-BY-NC.** Non-commercial. If you put its output in a monetized video,
you are outside the licence.

This tool refuses to use it. `engines.build("mmaudio")` raises an error explaining why
rather than quietly generating audio you cannot legally publish. What it uses instead is
**Stable Audio 3**, trained on licensed and Creative Commons data, whose licence grants
you ownership of the outputs and free commercial use below $1M annual revenue.

The licence check is enforced in code, not documented in a footnote — see
[LICENSING.md](LICENSING.md). **It is a summary, not legal advice. Read the licences.**

---

## Quick start (Windows)

Open PowerShell and paste this:

```powershell
irm https://raw.githubusercontent.com/preetamnegi7/hudka/main/install.ps1 | iex
```

It installs whatever you are missing, clones the repo, sets everything up and walks you
through model access. Safe to run twice — it skips what is already there.

Then **double-click `Hudka.bat`** and the app opens in your browser.

<details>
<summary>Prefer to do it by hand?</summary>

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # uv, manages Python
winget install Gyan.FFmpeg                                    # ffmpeg, does the video work
git clone https://github.com/preetamnegi7/hudka && cd hudka
```

Then double-click `Setup.bat`, which handles the rest.
</details>

The first render downloads about **3.3 GB** of model weights. After that it runs offline.

> `Setup.bat` exists for a reason. Installing Stable Audio 3 pulls the **CPU-only** torch
> wheel, and the obvious fix silently does nothing — `uv` sees `torch==2.7.1` as already
> satisfied, because the `+cpu` local tag does not affect version matching. It needs
> `--reinstall`. The script handles it.
>
> For the same reason, **running `uv sync` again removes the engine and the CUDA build** —
> they are installed outside `pyproject.toml` deliberately, since pinning a git URL and a
> GPU-specific wheel would break installation for everyone else. If generation stops
> working, run `Setup.bat` again. `hudka doctor` tells you which piece is missing.

### Model access

The weights are gated. Two one-off steps, both free:

1. Sign in to [Hugging Face](https://huggingface.co) and click **Agree** on both:
   - [stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
   - [stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
2. Run `uv run hf auth login` and paste a token from
   [your tokens page](https://huggingface.co/settings/tokens) (read access is enough)

---

## Requirements

| | minimum | measured on |
|---|---|---|
| **GPU** | optional | RTX 4070 — peaks at **1.7 GB VRAM**, so a 4 GB card is fine |
| **No GPU?** | works on CPU | ~17s per effect, ~3 min for a 48s video |
| **Disk** | ~5 GB | weights cache to the roomiest internal SSD |
| **OS** | Windows | Linux/macOS need a shim for the `.bat` files and drive detection |

A 48-second video renders in **66 seconds** on a 4070, from scratch with nothing cached.

---

## What it does

```
video ─▶ analyse ─▶ contact sheets ─▶ cue sheet ─▶ generate ─▶ mix ─▶ final.mp4
         ffmpeg +                     (you, or                 ffmpeg      + licence report
      PySceneDetect                    Claude)              duck + master
```

1. **Analyse** — shot boundaries, motion peaks, and where speech already exists
2. **Design cues** — what should be heard, and exactly when
3. **Generate** — each cue rendered by a licence-cleared model on your GPU
4. **Mix** — cues placed to the sample, music ducked under speech, mastered to −14 LUFS
   with a −1 dBTP ceiling so platforms leave your audio alone

`cues.json` is a plain file. Edit any cue by hand and re-render — a content-hash cache
regenerates only what changed.

### Designing the cues

Two ways, and you do not need both:

- **Create cue sheet** (in the app) — fully local. Anchors sounds to cuts and motion
  peaks. Knows *where* things happen, not *what* they are, so the prompts are generic.
- **`/sfx video.mp4`** in [Claude Code](https://claude.com/claude-code) — optional. Claude
  reads the contact sheets and writes cues from what is actually on screen. Much better
  results. **This is the only step that leaves your machine**; the video frames are sent
  to Anthropic's API. Skip it and everything stays local.

---

## Presets

| Preset | For | LUFS | Effects/min |
|---|---|---|---|
| `short-form` | Reels / Shorts / TikTok | −14 | 20–45 |
| `cinematic` | Film, trailers, narrative | −16 | 4–12 |
| `gameplay` | Gameplay / screen capture | −14 | 25–60 |
| `explainer` | Demos, ads, tutorials | −14 | 6–18 |

The app picks one from the video's shape and how much of it is narration, and you can
override it.

---

## Some things learned the hard way

**Generated audio has arbitrary loudness.** A cue's gain has to be an offset from a
*normalised* reference, not raw attenuation, or the same cue sheet produces a different
balance every run. Getting this wrong put music 27 dB under the dialogue — mastered
perfectly to target, and completely inaudible. `balance.py` now measures it and the render
reports the problem.

**Sub-2-second generations are unreliable.** Measured on `small-sfx`: 0.8s, 1.2s and 1.8s
all returned saturated audio pinned to ±1.0, while 1.0s, 1.5s and everything from 2.0s up
were clean. Not a threshold — roughly half fail. Effects are generated at a safe length
and trimmed.

**Weights cannot live on an external drive.** Memory-mapping multi-gigabyte files from
USB/exFAT fails with `STATUS_IN_PAGE_ERROR` — the process dies with no Python traceback at
all. The cache location check rejects removable and exFAT volumes regardless of free
space, because the external backup disk usually has the most of it.

**Generation runs in its own process.** Loading a second model into a process that already
released one is unreliable, and in the GUI a native crash would otherwise take the whole
server down mid-render.

---

## Command line

The app covers everything, but if you prefer a terminal:

```bash
uv run hudka gui                 # the app
uv run hudka analyze clip.mp4    # shots, motion, speech, contact sheets
uv run hudka scaffold out/clip   # starting cue sheet
uv run hudka render   out/clip   # generate, mix, master, mux
uv run hudka licences            # what each engine permits
uv run hudka doctor              # check ffmpeg, engines, GPU
```

## Tests

```bash
uv run pytest
```

135 tests, all offline — they run against a stub engine and an ffmpeg-generated fixture,
so no weights or GPU are needed. They assert shot detection against known cuts,
sample-accurate cue placement, loudness within tolerance, mix balance, cache reuse, and
that the licence gate blocks restricted engines before any generation starts.

## Status

Working, and honest about its age. Built and tested against Windows + NVIDIA. The
`hunyuan-foley` and `acestep-1.5` engines are written but have not been run. Issues and
pull requests welcome.

## Licence

Code: [MIT](LICENSE). Models: their own, and they differ — see
[LICENSING.md](LICENSING.md).
