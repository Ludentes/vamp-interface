"""Minimal probe: do two chained FluxSpaceEditPair nodes compose?

Render three images on elderly_latin_m at seed 2026:
  (1) smile only  (v6_full_demo at scale 0.5)
  (2) race only   (v1_la_hispanic at scale 0.5)
  (3) chained      (both, scale 0.5 each)

Measurement questions this answers:
  - Does chaining produce a distinct image from either single pair?
  - Does the chained image show a smile AND stay closer to Latino than (1)?
  - If yes: single-render composition is viable (cheapest path to the iterative
    corrective loop, no new ComfyUI node needed).
  - If no (chain == smile-only, or visually broken): we fall back to sequential
    img2img renders.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    pair_compose_workflow, pair_measure_workflow,
)
from src.demographic_pc.promptpair_iterate import (
    BASE_AGE_WORDS, BASE_ETHNICITY_WORDS, BASE_GENDER_WORDS, EVAL_BASES,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/demographic_pc/probe_chain"
SEED = 2026
BASE_NAME = "elderly_latin_m"
BASE_PROMPT = dict(EVAL_BASES)[BASE_NAME]

SMILE_POS = f"A {BASE_AGE_WORDS[BASE_NAME]} {BASE_ETHNICITY_WORDS[BASE_NAME]} {BASE_GENDER_WORDS[BASE_NAME]} smiling warmly."
SMILE_NEG = (f"A photorealistic portrait photograph of a {BASE_AGE_WORDS[BASE_NAME]} "
             f"{BASE_ETHNICITY_WORDS[BASE_NAME]} {BASE_GENDER_WORDS[BASE_NAME]} smiling warmly, "
             "plain grey background, studio lighting, sharp focus.")
RACE_POS = "A Latin American person."
RACE_NEG = "A Hispanic person."

SMILE_SCALE = 0.5
RACE_SCALE = 0.5


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        # 1. smile only (single-pair measure workflow)
        smile_png = OUT / "smile_only.png"
        if not smile_png.exists():
            wf = pair_measure_workflow(
                SEED, str(smile_png.with_suffix(".pkl")),
                "probe_smile_only",
                base_prompt=BASE_PROMPT, edit_a=SMILE_POS, edit_b=SMILE_NEG,
                scale=SMILE_SCALE)
            await client.generate(wf, smile_png)
            print(f"[done] {smile_png}")
        # 2. race only
        race_png = OUT / "race_only.png"
        if not race_png.exists():
            wf = pair_measure_workflow(
                SEED, str(race_png.with_suffix(".pkl")),
                "probe_race_only",
                base_prompt=BASE_PROMPT, edit_a=RACE_POS, edit_b=RACE_NEG,
                scale=RACE_SCALE)
            await client.generate(wf, race_png)
            print(f"[done] {race_png}")
        # 3. chained
        chain_png = OUT / "smile_plus_race.png"
        if not chain_png.exists():
            wf = pair_compose_workflow(
                SEED, "probe_smile_plus_race",
                base_prompt=BASE_PROMPT,
                pairs=[
                    {"edit_a": SMILE_POS, "edit_b": SMILE_NEG, "scale": SMILE_SCALE},
                    {"edit_a": RACE_POS, "edit_b": RACE_NEG, "scale": RACE_SCALE},
                ])
            await client.generate(wf, chain_png)
            print(f"[done] {chain_png}")
        # 4. baseline (scale=0, anchor)
        base_png = OUT / "baseline.png"
        if not base_png.exists():
            wf = pair_measure_workflow(
                SEED, None, "probe_baseline",
                base_prompt=BASE_PROMPT, edit_a=SMILE_POS, edit_b=SMILE_NEG,
                scale=0.0)
            await client.generate(wf, base_png)
            print(f"[done] {base_png}")


if __name__ == "__main__":
    asyncio.run(main())
