"""Five-rung smile ladder validation.

Tests whether the B-ladder has headroom past the 4-rung "manic" by adding a
fifth rung ("cackling, tongue visible") — does the pair-averaging recipe
still place the edit cleanly, or does the model refuse the extreme prompt?

6 bases × 5 ladder × 7 scales × 1 start_percent = 210 renders.
Writes to intensity_full/ alongside the 4-rung superset — adds rung `05_cackle`.
"""

from __future__ import annotations

import argparse
import asyncio

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_intensity_sweep import (
    BASES_FULL, ROOT_FULL, SCALES_FULL,
)
from src.demographic_pc.fluxspace_metrics import CROSS_SEED, pair_measure_workflow

NEW_RUNG = ("05_cackle",
            "cackling with head thrown back, mouth wide open, tongue visible")

# Full ladder = existing 4 + new 5th. Existing 4 are already rendered in
# intensity_full; we only render 05_cackle here.
NEW_LADDER = [NEW_RUNG]
SP = 0.15


def _splice(demo_phrase: str, ladder_phrase: str) -> str:
    return (f"A photorealistic portrait photograph of {demo_phrase} {ladder_phrase}, "
            f"plain grey background, studio lighting, sharp focus.")


async def run() -> None:
    edit_a = "A person cackling with head thrown back, mouth wide open, tongue visible."
    root = ROOT_FULL
    root.mkdir(parents=True, exist_ok=True)
    total = len(BASES_FULL) * len(NEW_LADDER) * len(SCALES_FULL)
    done = skipped = 0
    async with ComfyClient() as client:
        for base_name, base_prompt, demo_phrase in BASES_FULL:
            for ladder_id, ladder_phrase in NEW_LADDER:
                splice = _splice(demo_phrase, ladder_phrase)
                out_dir = root / base_name / ladder_id
                out_dir.mkdir(parents=True, exist_ok=True)
                for s in SCALES_FULL:
                    dest = out_dir / f"sp{SP:.2f}_s{s:+.2f}.png"
                    if dest.exists():
                        skipped += 1
                        continue
                    done += 1
                    print(f"[5rung {done+skipped}/{total}] "
                          f"{base_name}/{ladder_id} s={s:+.2f}")
                    await client.generate(
                        pair_measure_workflow(
                            CROSS_SEED, None,
                            f"fs5rung_{base_name}_{ladder_id}_s{s:+.2f}",
                            base_prompt=base_prompt,
                            edit_a=edit_a,
                            edit_b=splice,
                            scale=s,
                            start_percent=SP,
                            end_percent=1.0,
                        ),
                        dest,
                    )
    print(f"[5rung] done={done} skipped={skipped} → {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run())
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
