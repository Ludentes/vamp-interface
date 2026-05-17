"""FluxSpace expression axes for the ARKit-ControlNet spike.

`smile` uses the verified prompt pair from the 2026-04-21 FluxSpace smile-axis
experiment (docs/research/2026-04-21-fluxspace-smile-axis.md). `pucker` and
`surprise` are built by analogy on the same template and are UNCHARACTERISED —
their scale bands are provisional and must be eyeball-checked in Task 2.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Axis:
    name: str
    # FluxSpaceEditPair averages two edit-prompt attention caches at mix_b.
    edit_prompt_a: str          # generic edit prompt
    edit_prompt_b: str          # portrait-template edit prompt
    mix_b: float                # pair-averaging weight (0.5 = balanced)
    scale_band: list[float]     # edit scales to sweep
    target_channels: list[str]  # ARKit channels the axis should move


AXES: dict[str, Axis] = {
    "smile": Axis(
        name="smile",
        edit_prompt_a="A person smiling warmly.",
        edit_prompt_b=(
            "A photorealistic portrait photograph of a person smiling warmly, "
            "plain grey background, studio lighting, sharp focus."
        ),
        mix_b=0.5,
        scale_band=[0.5, 1.0, 1.5, 2.0],
        target_channels=["mouthSmileLeft", "mouthSmileRight"],
    ),
    "pucker": Axis(
        name="pucker",
        edit_prompt_a="A person puckering their lips.",
        edit_prompt_b=(
            "A photorealistic portrait photograph of a person puckering their "
            "lips, plain grey background, studio lighting, sharp focus."
        ),
        mix_b=0.5,
        scale_band=[0.5, 1.0, 1.5],
        target_channels=["mouthPucker", "mouthFunnel"],
    ),
    "surprise": Axis(
        name="surprise",
        edit_prompt_a="A person with a surprised expression, mouth open.",
        edit_prompt_b=(
            "A photorealistic portrait photograph of a person with a surprised "
            "expression and open mouth, plain grey background, studio lighting, "
            "sharp focus."
        ),
        mix_b=0.5,
        scale_band=[0.5, 1.0, 1.5],
        target_channels=["jawOpen", "browInnerUp", "eyeWideLeft", "eyeWideRight"],
    ),
}
