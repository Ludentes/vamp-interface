"""Expand calibration corpus for ridge-in-attention fit.

Generates ~150 diverse demographic base-only renders with FluxSpaceBaseMeasure
attention capture, 2 seeds each = ~300 renders. Covers the (age × gender ×
ethnicity × expression) factor space for a downstream ridge fit whose target
is a factor vector labelled per render by existing classifiers.

Output: `output/demographic_pc/fluxspace_metrics/calibration_expanded/` —
separate from the original 10-prompt `calibration/` to avoid mixing scales.

Each render produces one PNG + one PKL (per-(block, step) attn reductions).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import random

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import calibration_workflow, OUT

EXPANDED_DIR = OUT / "calibration_expanded"

AGES = [
    ("young", "a young"),
    ("adult", "an adult"),
    ("middle-aged", "a middle-aged"),
    ("elderly", "an elderly"),
]

GENDERS = [
    ("m", "man"),
    ("f", "woman"),
]

ETHNICITIES = [
    "East Asian",
    "South Asian",
    "Black",
    "European",
    "Latin American",
    "Middle Eastern",
    "Southeast Asian",
    "Mixed-race",
]

EXPRESSIONS = [
    ("neutral", "neutral expression"),
    ("slight-smile", "a slight smile"),
    ("pensive", "a pensive expression"),
    ("serious", "a serious expression"),
]

SEEDS = [20260, 20261]


def _enumerate_prompts(target_n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Return (short_name, full_prompt) pairs, uniform random sample over
    the combinatorial product, deduplicated."""
    combos = list(itertools.product(AGES, GENDERS, ETHNICITIES, EXPRESSIONS))
    rng.shuffle(combos)
    combos = combos[:target_n]
    out = []
    for (age_short, age_art), (g_short, g_word), ethn, (expr_short, expr_phrase) in combos:
        name = f"{age_short}_{ethn.lower().replace(' ', '_').replace('-', '_')}_{g_short}_{expr_short}"
        prompt = (f"A photorealistic portrait photograph of {age_art} {ethn} {g_word}, "
                  f"{expr_phrase}, plain grey background, studio lighting, sharp focus.")
        out.append((name, prompt))
    return out


async def run(n_prompts: int = 150) -> None:
    EXPANDED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    prompts = _enumerate_prompts(n_prompts, rng)
    total = len(prompts) * len(SEEDS)
    print(f"[expand-cal] {len(prompts)} prompts × {len(SEEDS)} seeds = {total} renders")
    done = skipped = 0
    async with ComfyClient() as client:
        for i, (name, prompt) in enumerate(prompts):
            for seed in SEEDS:
                tag = f"{i:03d}_{name}_s{seed}"
                img_dest = EXPANDED_DIR / f"{tag}.png"
                meas_path = EXPANDED_DIR / f"{tag}.pkl"
                if img_dest.exists() and meas_path.exists():
                    skipped += 1
                    continue
                done += 1
                print(f"[expand-cal {done+skipped}/{total}] {tag}")
                await client.generate(
                    calibration_workflow(prompt, seed, str(meas_path), f"calexp_{tag}"),
                    img_dest,
                )
    print(f"[expand-cal] done={done} skipped={skipped} → {EXPANDED_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=150)
    args = ap.parse_args()
    if args.run:
        asyncio.run(run(args.n_prompts))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
