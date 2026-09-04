# Audio licensing

This project exists because the licence on an AI audio model is not the same as
the licence on its code, and the difference decides whether you can monetize
what it produces.

**None of this is legal advice.** It is a summary of what these licences said
when this was written, with links so you can read them yourself. Licences change.
Your situation may differ. If money depends on the answer, check it properly.

## What this tool uses

| Model | Role | Licence | Commercial use |
|---|---|---|---|
| **Stable Audio 3** (small-sfx, small-music, medium) | default, effects and music | [Stability AI Community License](https://stability.ai/license) | Free below $1M annual revenue. You own the outputs and may distribute and commercialise them. Above $1M, an Enterprise licence is required. |
| **ACE-Step 1.5** | optional, music over 120s | [MIT](https://github.com/ace-step/ACE-Step-1.5) | Unrestricted. |

## What this tool deliberately refuses to use

| Model | Licence | Why it is excluded |
|---|---|---|
| **MMAudio** | Code MIT, **weights CC-BY-NC-4.0** | Non-commercial weights. Widely recommended for video-to-audio, and the code being MIT leads people to assume the whole thing is. It is not: output cannot be used in monetized or client work. `engines.build("mmaudio")` raises rather than running. |

## Opt-in, not default

| Model | Licence | Condition |
|---|---|---|
| **HunyuanVideo-Foley** | [Tencent Hunyuan Community](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley/blob/main/LICENSE) | Commercial use permitted, but the licence Territory **excludes the European Union, the United Kingdom and South Korea**, and deployments above 100M monthly active users need a separate licence. Requires `--engine hunyuan-foley` and prints a warning. |

## Provenance

Every render writes `provenance.json` and `LICENSE-REPORT.md` into the project
directory, recording for each generated sound: the model, its licence, the
prompt, the seed and the duration.

The render **fails** if any audio file lacks a provenance entry. An untracked
file is exactly the thing this project exists to prevent.

## Training data

Stable Audio 3 was trained on 1,278,902 recordings: 806,284 licensed from
AudioSparx and 472,618 from Freesound under CC-0, CC-BY or CC-Sampling+, with
copyrighted music screened out before training. That provenance is why it is the
default here.
