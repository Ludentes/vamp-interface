"""Phase 2 bootstrap: paired-contrast sampling for ridge-in-attention refit.

For each of 6 bases × 3 axes (smile, age, glasses) × 8 graded levels × 2 seeds,
render a base-only portrait with FluxSpaceBaseMeasure attention capture. All
prompts within one (base, axis) ladder share seed — enables clean paired
differences between adjacent levels and between the level-0 baseline and
every step up.

Total: 288 renders, ~72 min on local GPU.

Outputs: output/demographic_pc/fluxspace_metrics/bootstrap_v1/{axis}/
  {base}_{level_id}_s{seed}.png
  {base}_{level_id}_s{seed}.pkl

Morning Phase 1 scores these PNGs with MediaPipe blendshapes + CLIP + ArcFace
age → continuous labels for the ridge refit.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import OUT, calibration_workflow

BOOTSTRAP_DIR = OUT / "bootstrap_v1"


@dataclass
class Base:
    name: str
    age_art: str          # "an adult", "a young"
    age_token: str        # "adult", "young" — slot used by age ladder
    ethnicity: str        # "East Asian"
    gender_word: str      # "man" / "woman"


BASES = [
    Base("asian_m",         "an adult",   "adult",   "East Asian",      "man"),
    Base("black_f",         "an adult",   "adult",   "Black",           "woman"),
    Base("european_m",      "an adult",   "adult",   "European",        "man"),
    Base("elderly_latin_m", "an elderly", "elderly", "Latin American",  "man"),
    Base("young_european_f","a young",    "young",   "European",        "woman"),
    Base("southasian_f",    "an adult",   "adult",   "South Asian",     "woman"),
]

SEEDS = [2026, 4242]

# Ladders: (level_id, level_phrase). level_id is zero-padded, suffix describes
# the axis value in short words for debugging.
SMILE_LADDER = [
    ("01_neutral",  "neutral expression"),
    ("02_faint",    "a faint closed-mouth smile"),
    ("03_slight",   "a slight smile"),
    ("04_warm",     "smiling warmly"),
    ("05_broad",    "a broad smile with visible teeth"),
    ("06_grin",     "grinning widely with teeth bared"),
    ("07_laugh",    "laughing heartily with mouth open"),
    ("08_cackle",   "cackling with head thrown back, mouth wide open"),
]

# Age ladder replaces both age_art and the noun after it. We provide the full
# age phrase including article so it drops in cleanly.
AGE_LADDER = [
    ("01_teenage",     "a teenage"),
    ("02_young",       "a young"),
    ("03_young_adult", "a young adult"),
    ("04_adult",       "an adult"),
    ("05_mid_adult",   "a mature adult"),
    ("06_middle_aged", "a middle-aged"),
    ("07_older",       "an older"),
    ("08_elderly",     "an elderly"),
]

# Glasses ladder — appended after the subject noun phrase.
GLASSES_LADDER = [
    ("01_none",         ""),
    ("02_thin_wire",    " wearing thin wire-frame reading glasses"),
    ("03_round_wire",   " wearing delicate round wire-frame eyeglasses"),
    ("04_rectangular",  " wearing rectangular black-framed eyeglasses"),
    ("05_thick_rim",    " wearing thick-rimmed black eyeglasses"),
    ("06_oversized",    " wearing oversized bold-framed eyeglasses"),
    ("07_round_sunglass"," wearing round dark sunglasses"),
    ("08_dark_sunglass"," wearing dark aviator sunglasses"),
]


def prompt_smile(base: Base, phrase: str) -> str:
    return (f"A photorealistic portrait photograph of {base.age_art} "
            f"{base.ethnicity} {base.gender_word}, {phrase}, plain grey "
            f"background, studio lighting, sharp focus.")


def prompt_age(base: Base, age_phrase: str) -> str:
    # Age ladder replaces the age article for the whole subject phrase.
    return (f"A photorealistic portrait photograph of {age_phrase} "
            f"{base.ethnicity} {base.gender_word}, neutral expression, plain "
            f"grey background, studio lighting, sharp focus.")


def prompt_glasses(base: Base, glasses_phrase: str) -> str:
    return (f"A photorealistic portrait photograph of {base.age_art} "
            f"{base.ethnicity} {base.gender_word}{glasses_phrase}, neutral "
            f"expression, plain grey background, studio lighting, sharp focus.")


AXES = {
    "smile":   (SMILE_LADDER,   prompt_smile),
    "age":     (AGE_LADDER,     prompt_age),
    "glasses": (GLASSES_LADDER, prompt_glasses),
}


async def run() -> None:
    total = len(BASES) * sum(len(l) for l, _ in AXES.values()) * len(SEEDS)
    print(f"[phase2] total planned: {total} renders")
    done = skipped = 0
    async with ComfyClient() as client:
        for axis_name, (ladder, prompt_fn) in AXES.items():
            out_dir = BOOTSTRAP_DIR / axis_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for base in BASES:
                for level_id, phrase in ladder:
                    prompt = prompt_fn(base, phrase)
                    for seed in SEEDS:
                        tag = f"{base.name}_{level_id}_s{seed}"
                        img_dest = out_dir / f"{tag}.png"
                        meas_path = out_dir / f"{tag}.pkl"
                        if img_dest.exists() and meas_path.exists():
                            skipped += 1
                            continue
                        done += 1
                        print(f"[phase2 {done+skipped}/{total}] {axis_name}/{tag}")
                        await client.generate(
                            calibration_workflow(prompt, seed, str(meas_path),
                                                 f"bootstrap_{axis_name}_{tag}"),
                            img_dest,
                        )
    print(f"[phase2] done={done} skipped={skipped} → {BOOTSTRAP_DIR}")


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
