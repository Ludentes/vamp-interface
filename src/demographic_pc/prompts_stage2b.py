"""Stage 2b prompt grids — expression-varied (smile) and accessory-varied (glasses).

Stage 2 held expression and glasses constant ("neutral expression, plain grey
background…") so ridge directions fit to those labels picked up noise, not
signal. Stage 2b introduces prompt-level variance on each axis so the ridge
has something to fit.

Demographic axes are kept varied (not held constant) so the learned direction
is orthogonal to demographics where it can be.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.demographic_pc.prompts import ETHNICITIES, GENDERS

# Keep a small demographic grid — axis variation is the main driver here.
S2B_AGES = ["young adult", "adult"]  # 2 levels is enough; Stage 2 already covered 5

SMILE_LEVELS = [
    ("neutral",   "neutral expression, lips closed"),
    ("faint",     "faint closed-mouth smile, corners of the mouth slightly raised"),
    ("warm",      "warm smile, mouth corners clearly lifted, lips parted"),
    ("broad",     "broad smile showing teeth, cheeks raised"),
    ("laughing",  "laughing with mouth wide open, showing teeth, cheeks raised and eyes slightly creased"),
]

GLASSES_LEVELS = [
    ("none",      "no glasses, clear view of the eyes"),
    ("thin",      "wearing thin wire-rim eyeglasses"),
    ("thick",     "wearing thick black plastic-rim eyeglasses"),
    ("round",     "wearing round gold-rim eyeglasses"),
]

SMILE_TEMPLATE = (
    "A photorealistic portrait photograph of a {age} {ethnicity} {gender}, "
    "{expression}, plain grey background, studio lighting, sharp focus."
)

GLASSES_TEMPLATE = (
    "A photorealistic portrait photograph of a {age} {ethnicity} {gender}, "
    "neutral expression, {glasses}, plain grey background, studio lighting, sharp focus."
)

# Race-isolation grid: vary only the explicit race clause; hold everything else fixed
# except age and gender (so the ridge can't latch on to a single-demographic confound).
BLACK_AGES = ["12-year-old", "young adult", "middle-aged", "80-year-old"]
BLACK_RACE_LEVELS = [
    ("white",  "with White European features"),
    ("black",  "with Black African features"),
    ("easian", "with East Asian features"),
    ("sasian", "with South Asian features"),
    ("latino", "with Latino Hispanic features"),
]
BLACK_TEMPLATE = (
    "A photorealistic portrait photograph of a {age} {gender} {race_clause}, "
    "neutral expression, plain grey background, studio lighting, sharp focus, 50mm lens."
)


@dataclass
class S2BRow:
    prompt: str
    seed: int
    axis: str          # "smile" or "glasses"
    level_key: str     # e.g. "warm", "thick"
    age: str
    gender: str
    ethnicity: str
    sample_id: str


def smile_grid(seeds_per_cell: int = 1, start_seed: int = 300_000) -> list[S2BRow]:
    rows: list[S2BRow] = []
    for ai, age in enumerate(S2B_AGES):
        for gi, gender in enumerate(GENDERS):
            for ei, ethnicity in enumerate(ETHNICITIES):
                for li, (lkey, expr) in enumerate(SMILE_LEVELS):
                    for si in range(seeds_per_cell):
                        seed = start_seed + ai * 100000 + gi * 10000 + ei * 1000 + li * 100 + si
                        prompt = SMILE_TEMPLATE.format(age=age, ethnicity=ethnicity, gender=gender, expression=expr)
                        sample_id = f"smile-{ai}-{gi}-{ei}-{lkey}-s{seed}"
                        rows.append(S2BRow(prompt=prompt, seed=seed, axis="smile", level_key=lkey,
                                           age=age, gender=gender, ethnicity=ethnicity, sample_id=sample_id))
    return rows


def glasses_grid(seeds_per_cell: int = 1, start_seed: int = 400_000) -> list[S2BRow]:
    rows: list[S2BRow] = []
    for ai, age in enumerate(S2B_AGES):
        for gi, gender in enumerate(GENDERS):
            for ei, ethnicity in enumerate(ETHNICITIES):
                for li, (lkey, gclause) in enumerate(GLASSES_LEVELS):
                    for si in range(seeds_per_cell):
                        seed = start_seed + ai * 100000 + gi * 10000 + ei * 1000 + li * 100 + si
                        prompt = GLASSES_TEMPLATE.format(age=age, ethnicity=ethnicity, gender=gender, glasses=gclause)
                        sample_id = f"glasses-{ai}-{gi}-{ei}-{lkey}-s{seed}"
                        rows.append(S2BRow(prompt=prompt, seed=seed, axis="glasses", level_key=lkey,
                                           age=age, gender=gender, ethnicity=ethnicity, sample_id=sample_id))
    return rows


def black_grid(seeds_per_cell: int = 3, start_seed: int = 500_000) -> list[S2BRow]:
    rows: list[S2BRow] = []
    for ai, age in enumerate(BLACK_AGES):
        for gi, gender in enumerate(GENDERS):
            for ri, (rkey, rclause) in enumerate(BLACK_RACE_LEVELS):
                for si in range(seeds_per_cell):
                    seed = start_seed + ai * 100000 + gi * 10000 + ri * 1000 + si
                    prompt = BLACK_TEMPLATE.format(age=age, gender=gender, race_clause=rclause)
                    sample_id = f"black-{ai}-{gi}-{rkey}-s{seed}"
                    rows.append(S2BRow(prompt=prompt, seed=seed, axis="black", level_key=rkey,
                                       age=age, gender=gender, ethnicity=rkey, sample_id=sample_id))
    return rows


def all_rows(seeds_per_cell: int = 1) -> list[S2BRow]:
    return smile_grid(seeds_per_cell) + glasses_grid(seeds_per_cell)


if __name__ == "__main__":
    rows = all_rows(seeds_per_cell=1)
    print(f"Stage 2b total: {len(rows)}  smile={sum(1 for r in rows if r.axis=='smile')}  glasses={sum(1 for r in rows if r.axis=='glasses')}")
    print("\nFirst 3 smile:")
    for r in rows[:3]:
        print(f"  [{r.sample_id}] seed={r.seed}")
        print(f"    {r.prompt}")
    print("\nFirst 3 glasses:")
    for r in [r for r in rows if r.axis == "glasses"][:3]:
        print(f"  [{r.sample_id}] seed={r.seed}")
        print(f"    {r.prompt}")
