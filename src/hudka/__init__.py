"""Hudka — AI sound design from video.

Generates sound effects and background music matched to video content, using only
models whose licences permit commercial use and redistribution of outputs.

The pipeline is deliberately split so that only one stage is model-driven:

    analyze  ->  [contact sheets]  ->  cue sheet  ->  render  ->  mix  ->  final.mp4

`analyze`, `render` and `mix` are deterministic and testable. Writing the cue sheet
(deciding *what* should be heard and *when*) is the judgement call, and is done by
Claude reading the contact sheets — see `.claude/commands/sfx.md`.
"""

__version__ = "0.1.0"
