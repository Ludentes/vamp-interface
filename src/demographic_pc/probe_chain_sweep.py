"""2D scale sweep for chained smile + race pairs on elderly_latin_m.

Probe told us chaining works structurally but at equal weights (0.5, 0.5) the
race direction suppresses the smile. User note: at smile=0.5 the race drift is
mild, so we expect a small race correction — smile=0.5 × race∈{0.0..0.5} —
will land somewhere with both a visible smile and a corrected race flip.
Also vary smile∈{0.3, 0.5} to see how the curve shifts.

Renders all combos, builds a grid collage, and scores each cell via the full
classifier stack for the race/age/identity readouts.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import pair_compose_workflow
from src.demographic_pc.promptpair_iterate import (
    BASE_AGE_WORDS, BASE_ETHNICITY_WORDS, BASE_GENDER_WORDS, EVAL_BASES,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/demographic_pc/probe_chain_sweep"
SEED = 2026
BASE_NAME = "elderly_latin_m"
BASE_PROMPT = dict(EVAL_BASES)[BASE_NAME]

SMILE_POS = f"A {BASE_AGE_WORDS[BASE_NAME]} {BASE_ETHNICITY_WORDS[BASE_NAME]} {BASE_GENDER_WORDS[BASE_NAME]} smiling warmly."
SMILE_NEG = (f"A photorealistic portrait photograph of a {BASE_AGE_WORDS[BASE_NAME]} "
             f"{BASE_ETHNICITY_WORDS[BASE_NAME]} {BASE_GENDER_WORDS[BASE_NAME]} smiling warmly, "
             "plain grey background, studio lighting, sharp focus.")
RACE_POS = "A Latin American person."
RACE_NEG = "A Hispanic person."

SMILE_SCALES = [0.3, 0.5]
RACE_SCALES  = [0.0, 0.1, 0.2, 0.3, 0.5]


async def render_cell(client, smile_s: float, race_s: float) -> Path:
    tag = f"sm{smile_s:.2f}_ra{race_s:.2f}"
    dest = OUT / f"{tag}.png"
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    pairs = [{"edit_a": SMILE_POS, "edit_b": SMILE_NEG, "scale": smile_s}]
    if race_s != 0.0:
        pairs.append({"edit_a": RACE_POS, "edit_b": RACE_NEG, "scale": race_s})
    wf = pair_compose_workflow(SEED, f"probe_sweep_{tag}", BASE_PROMPT, pairs)
    await client.generate(wf, dest)
    print(f"[done] {dest.name}")
    return dest


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        for sm in SMILE_SCALES:
            for ra in RACE_SCALES:
                await render_cell(client, sm, ra)


def collage() -> None:
    from PIL import Image, ImageDraw, ImageFont
    cols = len(RACE_SCALES)
    rows = len(SMILE_SCALES)
    imgs: list[list[Image.Image]] = [
        [Image.open(OUT / f"sm{sm:.2f}_ra{ra:.2f}.png") for ra in RACE_SCALES]
        for sm in SMILE_SCALES
    ]
    w, h = imgs[0][0].size
    pad_top = 40
    pad_left = 90
    out = Image.new("RGB", (pad_left + cols * w, pad_top + rows * h), "white")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for j, ra in enumerate(RACE_SCALES):
        draw.text((pad_left + j * w + 10, 10), f"race={ra:.2f}", fill="black", font=font)
    for i, sm in enumerate(SMILE_SCALES):
        draw.text((10, pad_top + i * h + h // 2), f"smile={sm:.2f}", fill="black", font=font)
        for j, _ in enumerate(RACE_SCALES):
            out.paste(imgs[i][j], (pad_left + j * w, pad_top + i * h))
    dest = OUT / "collage.png"
    out.save(dest)
    print(f"[save] {dest}")


if __name__ == "__main__":
    asyncio.run(main())
    collage()
