"""Smile intensity dial — Mona Lisa vs Joker.

Sweeps three axes at once to map out fine-grained control of the smile edit:
  * B-ladder    — 4 splice prompts from faint closed-mouth to manic grin.
  * scale s     — 5 magnitudes from 0.4 (subtle) to 2.1 (near-collapse).
  * start_pct   — 3 onsets: 0.15 default, 0.25 later, 0.40 identity-preserving.

Fixed: mix_b=0.5, seed=2026, 3 bases (2 narrow, 1 wide geometry cluster).

Outputs:
  output/demographic_pc/fluxspace_metrics/crossdemo/smile/intensity/
    {base}/{ladder}/sp{start_pct}_s{scale:+.2f}.png
"""

from __future__ import annotations

import argparse
import asyncio

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    AXES, CALIBRATION_PROMPTS, CROSS_SEED, _axis_dir, pair_measure_workflow,
)

ROOT = _axis_dir("smile") / "intensity"
ROOT_FULL = _axis_dir("smile") / "intensity_full"

# B-ladder: low → high smile intensity. Each is a full-sentence splice
# matching the A=bare pattern, so pair averaging still cancels demographic
# confounds. Shared boilerplate kept identical across ladder rungs.
LADDER = [
    ("01_faint",  "with a faint closed-mouth smile"),
    ("02_warm",   "smiling warmly"),
    ("03_broad",  "grinning broadly with teeth showing"),
    ("04_manic",  "with a manic wide-open grin, teeth bared"),
]

# Demographic phrase per base — matches the splice builder in fluxspace_metrics.
BASES = [
    ("elderly_latin_m", CALIBRATION_PROMPTS[5][1], "an elderly Latin American man"),
    ("asian_m",         CALIBRATION_PROMPTS[1][1], "an adult East Asian man"),
    ("european_m",      CALIBRATION_PROMPTS[3][1], "an adult European man"),
]

# Full 6-demographic superset (matches AXES["smile"]["bases"]).
BASES_FULL = [
    ("asian_m",         CALIBRATION_PROMPTS[1][1], "an adult East Asian man"),
    ("black_f",         CALIBRATION_PROMPTS[2][1], "an adult Black woman"),
    ("european_m",      CALIBRATION_PROMPTS[3][1], "an adult European man"),
    ("elderly_latin_m", CALIBRATION_PROMPTS[5][1], "an elderly Latin American man"),
    ("young_european_f",CALIBRATION_PROMPTS[8][1], "a young European woman"),
    ("southasian_f",    CALIBRATION_PROMPTS[9][1], "an adult South Asian woman"),
]

SCALES = [0.4, 1.0, 1.4, 1.8, 2.1]
# Denser scale grid for the full sweep — adds a Mona-Lisa-low (0.2),
# a between-warm-and-approach (0.7), and a past-Joker (2.3).
SCALES_FULL = [0.2, 0.4, 0.7, 1.0, 1.4, 1.8, 2.3]
START_PCTS = [0.15, 0.25, 0.40]


def _splice(demo_phrase: str, ladder_phrase: str) -> str:
    return (f"A photorealistic portrait photograph of {demo_phrase} {ladder_phrase}, "
            f"plain grey background, studio lighting, sharp focus.")


async def run(full: bool = False) -> None:
    edit_a_by_ladder = {
        # A = bare verb phrase, analogous to existing SMILE_EDIT_A="A person smiling warmly."
        "01_faint":  "A person with a faint closed-mouth smile.",
        "02_warm":   AXES["smile"]["edit_a"],
        "03_broad":  "A person grinning broadly with teeth showing.",
        "04_manic":  "A person with a manic wide-open grin, teeth bared.",
    }
    bases = BASES_FULL if full else BASES
    scales = SCALES_FULL if full else SCALES
    root = ROOT_FULL if full else ROOT
    root.mkdir(parents=True, exist_ok=True)
    total = len(bases) * len(LADDER) * len(START_PCTS) * len(scales)
    done = skipped = 0
    async with ComfyClient() as client:
        for base_name, base_prompt, demo_phrase in bases:
            for ladder_id, ladder_phrase in LADDER:
                splice = _splice(demo_phrase, ladder_phrase)
                edit_a = edit_a_by_ladder[ladder_id]
                out_dir = root / base_name / ladder_id
                out_dir.mkdir(parents=True, exist_ok=True)
                for sp in START_PCTS:
                    for s in scales:
                        dest = out_dir / f"sp{sp:.2f}_s{s:+.2f}.png"
                        if dest.exists():
                            skipped += 1
                            continue
                        done += 1
                        print(f"[intensity {done+skipped}/{total}] "
                              f"{base_name}/{ladder_id} sp={sp:.2f} s={s:+.2f}")
                        await client.generate(
                            pair_measure_workflow(
                                CROSS_SEED, None,
                                f"fsint_{base_name}_{ladder_id}_sp{sp:.2f}_s{s:+.2f}",
                                base_prompt=base_prompt,
                                edit_a=edit_a,
                                edit_b=splice,
                                scale=s,
                                start_percent=sp,
                                end_percent=1.0,
                            ),
                            dest,
                        )
    print(f"[intensity] done={done} skipped={skipped} → {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="6 bases × 7 scales × 3 startpcts = 504; writes to intensity_full/")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run(full=args.full))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
