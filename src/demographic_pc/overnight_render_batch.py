"""Overnight (several-hour) render batch — paired edit/control for drift
analysis.

Three axes of new data, PNG-only (no measurement pkls → no disk bloat):

  (1) Smile calibration densification:
      6 bases × 4 rungs (faint/warm/broad/manic) × 4 scales (0.0, 0.4,
      0.7, 1.0) × 8 seeds = 768 renders.  scale=0 of every (base, seed)
      serves as the paired control for drift computation.

  (2) Beard axis (careful with gender):
      Additive beard on 2 male clean-shaven bases (asian_m, european_m):
          scales 0.0, 0.4, 0.7, 1.0
      Subtractive beard on elderly_latin_m (already bearded):
          scales 0.0, 0.4, 0.7, 1.0  (same edit, positive scale = less
          beard, using a clean-shaven edit prompt)
      8 seeds each → 3 × 4 × 8 = 96 renders.

  (3) Rebalance re-seeding for anger/surprise/pucker (the three usable
      rebalance axes from calibration):
      3 axes × 6 bases × 3 scales (0.4, 0.8, 1.2) × 8 seeds = 432 renders.

Total: ~1296 renders at ~2.5s = ~54 min. Well within "several hours".

All renders through FluxSpaceEditPair, measure_path=None → no .pkl
written. Each render produces one ~400 KB PNG. Total disk: ~520 MB.

Post-render: score every PNG with MediaPipe and dump a single
blendshapes.json per axis subdir.

Resumable — skips PNGs that already exist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    CALIBRATION_PROMPTS, pair_measure_workflow,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output/demographic_pc/overnight_drift"

SEEDS = [2026, 4242, 7331, 9001, 1234, 5557, 8080, 3141]


# -----------------------------------------------------------------------------
# bases
# -----------------------------------------------------------------------------

BASES_ALL = [
    ("asian_m",         CALIBRATION_PROMPTS[1][1], "an adult East Asian man"),
    ("black_f",         CALIBRATION_PROMPTS[2][1], "an adult Black woman"),
    ("european_m",      CALIBRATION_PROMPTS[3][1], "an adult European man"),
    ("elderly_latin_m", CALIBRATION_PROMPTS[5][1], "an elderly Latin American man"),
    ("young_european_f",CALIBRATION_PROMPTS[8][1], "a young European woman"),
    ("southasian_f",    CALIBRATION_PROMPTS[9][1], "an adult South Asian woman"),
]
BASES_MALE_CLEAN = ["asian_m", "european_m"]  # add-beard targets
BASES_MALE_BEARDED = ["elderly_latin_m"]      # subtract-beard targets

BASES_BY_NAME = {b[0]: b for b in BASES_ALL}


# -----------------------------------------------------------------------------
# smile ladder
# -----------------------------------------------------------------------------

SMILE_LADDER = [
    ("faint",  "A person with a faint closed-mouth smile.",
               "with a faint closed-mouth smile"),
    ("warm",   "A person smiling warmly.",
               "smiling warmly"),
    ("broad",  "A person grinning broadly with teeth showing.",
               "grinning broadly with teeth showing"),
    ("manic",  "A person with a manic wide-open grin, teeth bared.",
               "with a manic wide-open grin, teeth bared"),
]
SMILE_SCALES = [0.0, 0.4, 0.7, 1.0]


# -----------------------------------------------------------------------------
# beard axis
# -----------------------------------------------------------------------------

BEARD_ADD = ("A man with a thick full beard.", "with a thick full beard")
BEARD_REMOVE = ("A clean-shaven man, no beard.", "clean-shaven, no facial hair")
BEARD_SCALES = [0.0, 0.4, 0.7, 1.0]

# Bearded starting-prompt variants of the 3 male bases. Used by
# render_beard_rebalance to get same-identity bearded→unbearded pairs
# — so the `remove` direction can be fit on 3 bases instead of 1, and we
# can separate "beard" from the specific identity of elderly_latin_m.
BEARD_REBAL_BASES = [
    ("asian_m_bearded",         "A photorealistic portrait photograph of an adult East Asian man with a thick full beard, neutral expression, plain grey background, studio lighting, sharp focus.",
                                 "an adult East Asian man with a thick full beard"),
    ("european_m_bearded",      "A photorealistic portrait photograph of an adult European man with a thick full beard, neutral expression, plain grey background, studio lighting, sharp focus.",
                                 "an adult European man with a thick full beard"),
    ("elderly_latin_m_bearded", "A photorealistic portrait photograph of an elderly Latin American man with a thick full beard, neutral expression, plain grey background, studio lighting, sharp focus.",
                                 "an elderly Latin American man with a thick full beard"),
]


# -----------------------------------------------------------------------------
# rebalance re-seed (anger / surprise / pucker only — confirmed usable)
# -----------------------------------------------------------------------------

REBALANCE_AXES = {
    "anger": {
        "edit_a": "A person with angry brows lowered and jaw tensed.",
        "clause": "with angry brows lowered and jaw tensed",
    },
    "surprise": {
        "edit_a": "A person with raised brows and mouth agape.",
        "clause": "with raised brows and mouth agape",
    },
    "pucker": {
        "edit_a": "A person with lips puckered forward.",
        "clause": "with lips puckered forward",
    },
}
REBALANCE_SCALES = [0.0, 0.4, 0.8, 1.2]


def _splice(demo: str, clause: str) -> str:
    return (f"A photorealistic portrait photograph of {demo} {clause}, "
            f"plain grey background, studio lighting, sharp focus.")


def _workflow(seed, prefix, base_prompt, edit_a, edit_b, scale, measure_path):
    """pair_measure_workflow with slim v2 `.v2.npz` measurement output."""
    return pair_measure_workflow(
        seed, measure_path, prefix,
        base_prompt=base_prompt, edit_a=edit_a, edit_b=edit_b,
        scale=scale, start_percent=0.15, end_percent=1.0,
    )


def _measure_path(png_dest: Path) -> str:
    """Convention: `<stem>.v2.npz` alongside the PNG. Extension-based
    version dispatch in the node handles the slim-format write."""
    return str(png_dest.with_suffix(".v2.npz"))


def _already_done(png_dest: Path, expect_npz: bool) -> bool:
    """Resume-safe: treat a cell as done ONLY when both PNG and its
    paired .v2.npz exist and are non-empty. Handles the case where the
    process was killed mid-render leaving an orphan PNG or npz."""
    if not png_dest.exists() or png_dest.stat().st_size < 1024:
        return False
    if not expect_npz:
        return True
    npz = Path(_measure_path(png_dest))
    # scale=0 renders skip the npz path (edit hook is disabled); those
    # are "done" when the PNG exists.
    if "_s+0.00" in png_dest.name:
        return True
    return npz.exists() and npz.stat().st_size > 10_000


async def render_smile(client: ComfyClient) -> int:
    root = OUT_ROOT / "smile"
    n_new = n_skip = 0
    for rung, edit_a, clause in SMILE_LADDER:
        for base_name, base_prompt, demo in BASES_ALL:
            d = root / rung / base_name
            d.mkdir(parents=True, exist_ok=True)
            splice = _splice(demo, clause)
            for seed in SEEDS:
                for s in SMILE_SCALES:
                    dest = d / f"seed{seed}_s{s:+.2f}.png"
                    if dest.exists():
                        n_skip += 1
                        continue
                    print(f"[smile/{rung}/{base_name} seed={seed} s={s:+.2f}]")
                    await client.generate(
                        _workflow(seed,
                                  f"odr_smile_{rung}_{base_name}_seed{seed}_s{s:+.2f}",
                                  base_prompt, edit_a, splice, s,
                                  _measure_path(dest)),
                        dest)
                    n_new += 1
    return n_new, n_skip  # type: ignore[return-value]


async def render_beard(client: ComfyClient) -> int:
    root = OUT_ROOT / "beard"
    n_new = n_skip = 0
    # add-beard on clean-shaven male bases
    for base_name in BASES_MALE_CLEAN:
        base_prompt, demo = BASES_BY_NAME[base_name][1], BASES_BY_NAME[base_name][2]
        d = root / "add" / base_name
        d.mkdir(parents=True, exist_ok=True)
        edit_a, clause = BEARD_ADD
        splice = _splice(demo, clause)
        for seed in SEEDS:
            for s in BEARD_SCALES:
                dest = d / f"seed{seed}_s{s:+.2f}.png"
                if dest.exists():
                    n_skip += 1
                    continue
                print(f"[beard/add/{base_name} seed={seed} s={s:+.2f}]")
                await client.generate(
                    _workflow(seed, f"odr_beardadd_{base_name}_seed{seed}_s{s:+.2f}",
                              base_prompt, edit_a, splice, s,
                              _measure_path(dest)),
                    dest)
                n_new += 1
    # remove-beard on bearded male base
    for base_name in BASES_MALE_BEARDED:
        base_prompt, demo = BASES_BY_NAME[base_name][1], BASES_BY_NAME[base_name][2]
        d = root / "remove" / base_name
        d.mkdir(parents=True, exist_ok=True)
        edit_a, clause = BEARD_REMOVE
        splice = _splice(demo, clause)
        for seed in SEEDS:
            for s in BEARD_SCALES:
                dest = d / f"seed{seed}_s{s:+.2f}.png"
                if dest.exists():
                    n_skip += 1
                    continue
                print(f"[beard/remove/{base_name} seed={seed} s={s:+.2f}]")
                await client.generate(
                    _workflow(seed, f"odr_beardrem_{base_name}_seed{seed}_s{s:+.2f}",
                              base_prompt, edit_a, splice, s,
                              _measure_path(dest)),
                    dest)
                n_new += 1
    return n_new, n_skip  # type: ignore[return-value]


async def render_beard_rebalance(client: ComfyClient) -> int:
    """Same-identity bearded→unbearded pairs: use a forced-bearded base
    prompt on all 3 male bases, then apply `remove` edit at 4 scales.

    Produces 3 × 4 × 8 = 96 renders under beard_rebalance/remove/<base>_bearded/.
    The scale=0 PNG is the bearded control; scales 0.4/0.7/1.0 test how
    strongly the remove direction lifts the beard off each identity. Fixes
    the 2+1 asymmetry by giving `remove` 3 distinct starting identities.
    """
    root = OUT_ROOT / "beard_rebalance"
    n_new = n_skip = 0
    edit_a, clause = BEARD_REMOVE
    for base_name, base_prompt, demo in BEARD_REBAL_BASES:
        d = root / "remove" / base_name
        d.mkdir(parents=True, exist_ok=True)
        splice = _splice(demo, clause)
        for seed in SEEDS:
            for s in BEARD_SCALES:
                dest = d / f"seed{seed}_s{s:+.2f}.png"
                if dest.exists():
                    n_skip += 1
                    continue
                print(f"[beard_rebal/remove/{base_name} seed={seed} s={s:+.2f}]")
                await client.generate(
                    _workflow(seed, f"odr_beardrebal_{base_name}_seed{seed}_s{s:+.2f}",
                              base_prompt, edit_a, splice, s,
                              _measure_path(dest)),
                    dest)
                n_new += 1
    return n_new, n_skip  # type: ignore[return-value]


async def render_rebalance(client: ComfyClient) -> int:
    root = OUT_ROOT / "rebalance_reseed"
    n_new = n_skip = 0
    for axis, cfg in REBALANCE_AXES.items():
        for base_name, base_prompt, demo in BASES_ALL:
            d = root / axis / base_name
            d.mkdir(parents=True, exist_ok=True)
            splice = _splice(demo, cfg["clause"])
            for seed in SEEDS:
                for s in REBALANCE_SCALES:
                    dest = d / f"seed{seed}_s{s:+.2f}.png"
                    if dest.exists():
                        n_skip += 1
                        continue
                    print(f"[rebal/{axis}/{base_name} seed={seed} s={s:+.2f}]")
                    await client.generate(
                        _workflow(seed, f"odr_rb_{axis}_{base_name}_seed{seed}_s{s:+.2f}",
                                  base_prompt, cfg["edit_a"], splice, s,
                                  _measure_path(dest)),
                        dest)
                    n_new += 1
    return n_new, n_skip  # type: ignore[return-value]


async def run(parts: list[str]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        if "smile" in parts:
            n_new, n_skip = await render_smile(client)
            print(f"[smile] rendered={n_new} skipped={n_skip}")
        if "beard" in parts:
            n_new, n_skip = await render_beard(client)
            print(f"[beard] rendered={n_new} skipped={n_skip}")
        if "beard_rebalance" in parts:
            n_new, n_skip = await render_beard_rebalance(client)
            print(f"[beard_rebalance] rendered={n_new} skipped={n_skip}")
        if "rebalance" in parts:
            n_new, n_skip = await render_rebalance(client)
            print(f"[rebalance] rendered={n_new} skipped={n_skip}")


def score_tree(root: Path) -> None:
    """MediaPipe-score every PNG under root, per axis subdir."""
    from src.demographic_pc.score_blendshapes import make_landmarker, score_png
    with make_landmarker() as lm:
        for axis_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            pngs = sorted(axis_dir.rglob("*.png"))
            if not pngs:
                continue
            out_json = axis_dir / "blendshapes.json"
            scores: dict[str, dict] = {}
            for p in pngs:
                rel = str(p.relative_to(axis_dir))
                s = score_png(lm, p)
                if s is not None:
                    scores[rel] = s
            out_json.write_text(json.dumps(scores, indent=2))
            print(f"[score] {axis_dir.name}: {len(scores)}/{len(pngs)} → {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--parts", nargs="*",
                    choices=["smile", "beard", "beard_rebalance", "rebalance"],
                    default=["smile", "beard", "rebalance"])
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    if args.score_only:
        score_tree(OUT_ROOT)
        return
    if not args.run:
        ap.print_help()
        return
    total = 0
    if "smile" in args.parts:
        total += 6 * len(SMILE_LADDER) * len(SMILE_SCALES) * len(SEEDS)
    if "beard" in args.parts:
        total += ((len(BASES_MALE_CLEAN) + len(BASES_MALE_BEARDED))
                  * len(BEARD_SCALES) * len(SEEDS))
    if "beard_rebalance" in args.parts:
        total += len(BEARD_REBAL_BASES) * len(BEARD_SCALES) * len(SEEDS)
    if "rebalance" in args.parts:
        total += (len(REBALANCE_AXES) * len(BASES_ALL) * len(REBALANCE_SCALES)
                  * len(SEEDS))
    print(f"[overnight] parts={args.parts}  est renders={total}  "
          f"est time={total*2.5/60:.0f} min")
    asyncio.run(run(args.parts))
    if args.score:
        score_tree(OUT_ROOT)


if __name__ == "__main__":
    main()
