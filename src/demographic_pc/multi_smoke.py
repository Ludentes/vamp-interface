"""Smoke test for FluxSpaceEditPairMulti.

Three renders on european_m seed 2026 to validate node mechanics:

  A  single-slot smile s=0.5           — sanity vs iter_08 v_eur_me output
  B  two slots of same smile pair at   — additivity: 0.25+0.25 should ≈ A
     0.25 + 0.25
  C  smile s=0.5 + age-maintain s=0.3  — real composition test

Run:
  uv run python -m src.demographic_pc.multi_smoke
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    pair_measure_workflow,
    pair_multi_measure_workflow,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/demographic_pc/multi_smoke"

BASE_PROMPT = (
    "A photorealistic portrait photograph of an adult European man, "
    "neutral expression, plain grey background, studio lighting, sharp focus."
)

SMILE_POS = "An adult European man smiling warmly."
SMILE_NEG = "An adult Middle Eastern man smiling warmly."

# Age-maintain pair: one half keeps adult, the other pushes young → the
# asymmetry encodes "hold age where it is, even if smile wants to age him."
AGE_POS = "An adult European man, neutral expression."
AGE_NEG = "A young European man, neutral expression."

SEED = 2026


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        # A — single-slot via existing pair node (reference)
        destA = OUT / "A_single_smile_s0.50.png"
        meaA = OUT / "A_single_smile_s0.50.pkl"
        if not destA.exists():
            wf = pair_measure_workflow(
                SEED, str(meaA), "multi_smoke_A",
                base_prompt=BASE_PROMPT, edit_a=SMILE_POS, edit_b=SMILE_NEG,
                scale=0.5, start_percent=0.15, end_percent=1.0,
            )
            await client.generate(wf, destA)
        print(f"[A] → {destA}")

        # B — two slots, same smile pair, 0.25 + 0.25 (additivity check)
        destB = OUT / "B_twoslot_smile_0.25_0.25.png"
        meaB = OUT / "B_twoslot_smile_0.25_0.25.pkl"
        if not destB.exists():
            wf = pair_multi_measure_workflow(
                SEED, str(meaB), "multi_smoke_B",
                base_prompt=BASE_PROMPT,
                pairs=[
                    {"edit_a": SMILE_POS, "edit_b": SMILE_NEG,
                     "scale": 0.25, "label": "smile_halfA"},
                    {"edit_a": SMILE_POS, "edit_b": SMILE_NEG,
                     "scale": 0.25, "label": "smile_halfB"},
                ],
            )
            await client.generate(wf, destB)
        print(f"[B] → {destB}")

        # C — smile s=0.5 + age-maintain s=0.3 (real composition)
        destC = OUT / "C_smile_plus_ageholdr.png"
        meaC = OUT / "C_smile_plus_ageholdr.pkl"
        if not destC.exists():
            wf = pair_multi_measure_workflow(
                SEED, str(meaC), "multi_smoke_C",
                base_prompt=BASE_PROMPT,
                pairs=[
                    {"edit_a": SMILE_POS, "edit_b": SMILE_NEG,
                     "scale": 0.5, "label": "smile_v_eur_me"},
                    {"edit_a": AGE_POS, "edit_b": AGE_NEG,
                     "scale": 0.3, "label": "age_hold_adult"},
                ],
            )
            await client.generate(wf, destC)
        print(f"[C] → {destC}")


if __name__ == "__main__":
    asyncio.run(main())
