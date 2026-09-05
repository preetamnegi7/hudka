# Measurements

Numbers this code depends on, and where each came from. Constants in the source cite this
file; if a number here changes, the constant changes with it - never the other way round.

## Machine

RTX 4070 (12.878 GB, compute 8.9, bf16), driver 591.86, i7-13700K (24 threads), 68.5 GB
RAM, Windows 11 26200, torch 2.7.1+cu128, `stable-audio-3` 0.1.0. Weights on F:
(`F:\hudka-models\huggingface`).

## Free VRAM is not what torch says it is (2026-09-05)

Same instant, same card:

| source | free |
|---|---|
| `torch.cuda.mem_get_info()` | 11.63 GB |
| `nvidia-smi --query-gpu=memory.free` | 7.63 GB |

Under WDDM, `cudaMemGetInfo` counts other applications' evictable memory as free. Budgeting
on it decides a model fits, then evicts Edge and LM Studio to system RAM or dies. It also
creates a ~180 MB CUDA context in whichever process asks. `engines/hardware.py` therefore
reads nvidia-smi and never imports torch in the GUI server.

At rest on this machine 7.3–8.6 GB is free; the desktop (Edge ×9, Spotify, Teams, LM
Studio, ChatGPT, PowerToys, …) holds the rest. With Adobe Premiere Pro 2025 open the card
went to 0.45 GB free and the tier flipped from `gpu-medium` to `gpu-lite` live - which is
the behaviour, not a bug.

## B1 - the medium model on 12 GB (2026-09-05)

`scratchpad/bench_b1.py`: one engine per process (as the real worker), CPU-first load,
fp16 DiT+VAE, bf16 text encoder, 50 steps, seed 63486, the `warm rhodes` bed prompt.
Physical VRAM sampled by `nvidia-smi -lms 500` alongside. 8.64 GB free at start.

### stable-audio-3-medium

| step | seconds | torch allocated | torch reserved (peak) | physical (ours) |
|---|---|---|---|---|
| load (9.22 GB checkpoint) | **20.7** | 4.65 GB | 4.84 GB | 4.95 GB |
| 5 s bed (includes torch.compile warm-up) | 18.9 | 5.25 | 7.01 | 7.18 |
| 30 s bed | 5.9 | 5.25 | 8.38 | 7.80 |
| 60 s bed | 10.2 | 5.25 | 8.38 | 7.82 |
| 120 s bed | 18.0 | 5.25 | 8.38 | 7.79 |
| 240 s bed | 17.7 | 5.25 | 8.38 | 7.89 |
| **380 s bed** | 16.5 | 5.25 | 8.38 | **7.92** |

Card total during the run: 12.02 GB of 12.88 (3.95 GB was other applications).

What this settles:

- **Medium loads and generates on this card, up to its full 380 s, in one pass.** The
  in-repo claim that it "kills the process" was the library's load order (fp32 onto the
  GPU, then halve), not the model's size.
- **The footprint is flat.** 7.9 GB physical from 30 s to 380 s: the DiT works on a fixed
  latent length. The first prediction (7.0 GB at 120 s rising to 8.1 at 380 s, derived from
  parameter counts) was wrong in both shape and level. `MEDIUM_RUN_PEAK_GB = 7.9`,
  `MEDIUM_MIN_FREE_GB = 8.4`.
- **~3 GB of the 8.4 GB torch reserves is allocator cache** (allocated 5.25). It is
  reclaimable between renders with `empty_cache()`; a resident worker should do that.
- **The first cue pays ~13 s of torch.compile warm-up.** Every render pays it while workers
  are created per render. This, plus the 20.7 s load, is the case for resident workers.
- On this machine medium needs the desktop to hold **under ~4.4 GB**. It ran today with
  0.6 GB to spare. The GUI says which applications to close, and the render falls back to
  the small model with a warning rather than failing.

### stable-audio-3-small-music, same prompt and seed

| step | seconds | torch reserved (peak) | physical (ours) |
|---|---|---|---|
| load | 10.8 | 1.16 GB | 1.30 GB |
| 30 s bed (includes warm-up) | 3.2 | 1.86 | 2.05 |
| 120 s bed | 2.5 | 1.99 | 2.13 |

`SMALL_RUN_PEAK_GB = 2.1`. The 12–16 s "engine swap" measured earlier from stem mtimes is
this load plus the warm-up.

### A/B, 30 s, identical prompt and seed, 50 steps

| | spectral centroid | flatness | crest | peak | RMS |
|---|---|---|---|---|---|
| small-music | 256 Hz | 0.000 | 19.3 dB | −1.4 dBFS | −20.6 dBFS |
| medium | 341 Hz | 0.000 | 21.3 dB | 0.0 dBFS | −21.3 dBFS |
| medium, 380 s | 186 Hz | 0.000 | 21.1 dB | −2.9 dBFS | −24.1 dBFS |

Medium carries more high-frequency content and more dynamic range on the same prompt -
consistent with the "airy top end" the prompt asks for. These are descriptors, not a
verdict; the two 30 s files were handed to the user for a listening comparison. The peak at
0.0 dBFS is the library clamping to ±1; placement normalises level, so it costs nothing.

### Not measured yet

- Medium for one-shots (`quality=best`): time per 2 s cue and whether it beats small-sfx,
  which is post-trained for exactly them. `auto` keeps small-sfx until this is listened to.
- fp16 vs bf16 for the DiT/VAE, and the bf16 text-encoder cast's audibility.
- `medium-base` / `small-*-base` (the non-distilled checkpoints) at 50–100 steps.
