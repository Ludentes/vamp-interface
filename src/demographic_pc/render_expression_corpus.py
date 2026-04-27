"""Expression corpus rebalance (plan: docs/research/2026-04-23-corpus-rebalance-todo.md).

Renders 5 non-smile expression axes across 6 canonical demographic bases with
multiple seeds and scales so NMF on the combined corpus is no longer
smile-dominated.

Shape: 5 axes × 6 bases × 5 seeds × 5 scales × 2 start_percent = 1500 renders.

Saves both PNG and attention-cache .pkl under
  output/demographic_pc/fluxspace_metrics/crossdemo/{axis}/rebalance/{base}/
    seed{seed}_s{scale:+.2f}.{png,pkl}

Resumable: skips if both PNG and PKL already exist.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    CALIBRATION_PROMPTS, CROSS_DIR, pair_measure_workflow,
)

# Canonical 6-base set (matches alpha_interp_attn / smile intensity_full).
BASES = [
    ("asian_m",         CALIBRATION_PROMPTS[1][1], "an adult East Asian man"),
    ("black_f",         CALIBRATION_PROMPTS[2][1], "an adult Black woman"),
    ("european_m",      CALIBRATION_PROMPTS[3][1], "an adult European man"),
    ("elderly_latin_m", CALIBRATION_PROMPTS[5][1], "an elderly Latin American man"),
    ("young_european_f",CALIBRATION_PROMPTS[8][1], "a young European woman"),
    ("southasian_f",    CALIBRATION_PROMPTS[9][1], "an adult South Asian woman"),
]

# Expression axes: A = bare verb phrase (analogous to PAIR_EDIT_A pattern);
# B = full-sentence splice that replaces "neutral expression" clause on the base.
AXES = {
    "anger": {
        "edit_a": "A person with angry brows lowered and jaw tensed.",
        "splice_clause": "with angry brows lowered and jaw tensed",
    },
    "surprise": {
        "edit_a": "A person with raised brows and mouth agape.",
        "splice_clause": "with raised brows and mouth agape",
    },
    "disgust": {
        "edit_a": "A person with a sneer and wrinkled nose.",
        "splice_clause": "with a sneer and wrinkled nose",
    },
    "pucker": {
        "edit_a": "A person with lips puckered forward.",
        "splice_clause": "with lips puckered forward",
    },
    "lip_press": {
        "edit_a": "A person with lips pressed firmly together.",
        "splice_clause": "with lips pressed firmly together",
    },
}

SEEDS = [2026, 4242, 7331, 9001, 1234]
SCALES = [0.4, 0.8, 1.2, 1.6, 2.0]
START_PCTS = [0.15, 0.30]


def _splice(demo_phrase: str, clause: str) -> str:
    return (f"A photorealistic portrait photograph of {demo_phrase} {clause}, "
            f"plain grey background, studio lighting, sharp focus.")


def _axis_root(axis: str) -> Path:
    return CROSS_DIR / axis / "rebalance"


async def run(axes: list[str]) -> None:
    total = len(axes) * len(BASES) * len(SEEDS) * len(SCALES) * len(START_PCTS)
    done = skipped = 0
    async with ComfyClient() as client:
        for axis in axes:
            cfg = AXES[axis]
            root = _axis_root(axis)
            for base_name, base_prompt, demo_phrase in BASES:
                out_dir = root / base_name
                out_dir.mkdir(parents=True, exist_ok=True)
                splice = _splice(demo_phrase, cfg["splice_clause"])
                for seed in SEEDS:
                    for sp in START_PCTS:
                        for s in SCALES:
                            stem = f"seed{seed}_sp{sp:.2f}_s{s:+.2f}"
                            png = out_dir / f"{stem}.png"
                            pkl = out_dir / f"{stem}.pkl"
                            if png.exists() and pkl.exists():
                                skipped += 1
                                continue
                            done += 1
                            print(f"[rebalance {done+skipped}/{total}] "
                                  f"{axis}/{base_name} seed={seed} sp={sp:.2f} s={s:+.2f}")
                            await client.generate(
                                pair_measure_workflow(
                                    seed, str(pkl),
                                    f"reb_{axis}_{base_name}_{stem}",
                                    base_prompt=base_prompt,
                                    edit_a=cfg["edit_a"],
                                    edit_b=splice,
                                    scale=s,
                                    start_percent=sp,
                                    end_percent=1.0,
                                ),
                                png,
                            )
    print(f"[rebalance] done={done} skipped={skipped}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--axes", nargs="*", default=list(AXES.keys()),
                    help=f"subset of {list(AXES.keys())}")
    args = ap.parse_args()
    if not args.run:
        ap.print_help()
        return
    bad = [a for a in args.axes if a not in AXES]
    if bad:
        raise SystemExit(f"unknown axes: {bad}")
    asyncio.run(run(args.axes))


if __name__ == "__main__":
    main()
