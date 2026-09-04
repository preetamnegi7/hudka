"""Licence facts for every engine, verified against the licence texts in September 2026.

Kept in one file so the legal position is auditable at a glance rather than scattered
through implementations. `saand licences` prints this table.
"""

from __future__ import annotations

from .base import Licence

STABILITY_COMMUNITY = Licence(
    name="Stability AI Community License",
    url="https://stability.ai/license",
    commercial=True,
    revenue_cap_usd=1_000_000,
    training_data=(
        "1,278,902 recordings: 806,284 licensed from AudioSparx and 472,618 from Freesound "
        "under CC-0 / CC-BY / CC-Sampling+. Copyrighted music screened out before training."
    ),
    notes=(
        "You own the outputs and may distribute and commercialise them freely — no sync "
        "licence, no royalty split. Above $1M annual revenue an Enterprise licence is "
        "required, which also carries legal indemnification. Hosted services should carry "
        "'Powered by Stability AI' attribution."
    ),
)

MIT = Licence(
    name="MIT",
    url="https://github.com/ace-step/ACE-Step-1.5/blob/main/LICENSE",
    commercial=True,
    training_data="Not disclosed by the authors.",
    notes="Unrestricted commercial use and redistribution.",
)

TENCENT_HUNYUAN_COMMUNITY = Licence(
    name="Tencent Hunyuan Community License",
    url="https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley/blob/main/LICENSE",
    commercial=True,
    requires_optin=True,
    territory_exclusions=("the European Union", "the United Kingdom", "South Korea"),
    training_data="Not disclosed by the authors.",
    notes=(
        "Commercial use and hosted services are permitted, but the licence Territory "
        "excludes the EU, UK and South Korea, and deployments above 100M monthly active "
        "users need a separate licence from Tencent. Opt-in because a globally distributed "
        "video sits awkwardly with a territorial carve-out."
    ),
)

# Kept only so the exclusion is explicit and documented. Never selectable by default:
# the weights are CC-BY-NC, which rules out monetized video and client work, even though
# the surrounding code is MIT. This is the trap most video-to-audio tutorials fall into.
CC_BY_NC = Licence(
    name="CC-BY-NC-4.0",
    url="https://creativecommons.org/licenses/by-nc/4.0/",
    commercial=False,
    notes="Non-commercial only. Excluded from this tool's default stack by design.",
)

PUBLIC_DOMAIN = Licence(
    name="n/a (generates silence)",
    url="",
    commercial=True,
    training_data="No model, no training data.",
    notes="Test stub used by CI so the pipeline can be exercised without downloading weights.",
)
