"""Prompt grid for demographic-PC extraction.

Full grid (Stage 2, N=1800):  5 ages × 3 genders × 7 ethnicities × 17 seeds = 1785 ≈ 1800
Sanity grid (Stage 1, N=50):  25 stratified cells × 2 seeds = 50

Each row is a (prompt, seed, age_label, gender_label, ethnicity_label) record.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

AGES = ["child", "young adult", "adult", "middle-aged", "elderly"]
GENDERS = ["man", "woman", "non-binary person"]
ETHNICITIES = [
    "East Asian", "Southeast Asian", "South Asian",
    "Black", "White", "Hispanic or Latino", "Middle Eastern",
]

PROMPT_TEMPLATE = (
    "A photorealistic portrait photograph of a {age} {ethnicity} {gender}, "
    "neutral expression, plain grey background, studio lighting, sharp focus."
)


@dataclass
class PromptRow:
    prompt: str
    seed: int
    age: str
    gender: str
    ethnicity: str
    cell_id: str  # "{age_idx}-{gender_idx}-{ethnicity_idx}"

    @property
    def sample_id(self) -> str:
        return f"{self.cell_id}-s{self.seed}"


def full_grid(seeds_per_cell: int = 17, start_seed: int = 1000) -> list[PromptRow]:
    """Full 5×3×7×seeds_per_cell grid for Stage 2."""
    rows: list[PromptRow] = []
    for ai, age in enumerate(AGES):
        for gi, gender in enumerate(GENDERS):
            for ei, ethnicity in enumerate(ETHNICITIES):
                for si in range(seeds_per_cell):
                    seed = start_seed + ai * 10000 + gi * 1000 + ei * 100 + si
                    rows.append(PromptRow(
                        prompt=PROMPT_TEMPLATE.format(age=age, ethnicity=ethnicity, gender=gender),
                        seed=seed,
                        age=age,
                        gender=gender,
                        ethnicity=ethnicity,
                        cell_id=f"{ai}-{gi}-{ei}",
                    ))
    return rows


def sanity_grid(n_cells: int = 25, seeds_per_cell: int = 2, seed: int = 42) -> list[PromptRow]:
    """Stratified 25-cell × 2-seed = 50 grid for Stage 1.

    Stratification guarantees:
    - every age level (5) appears in ≥ 5 cells
    - every gender level (3) appears in ≥ 8 cells
    - every ethnicity level (7) appears in ≥ 3 cells
    achieved by Latin-square-like allocation.
    """
    rng = random.Random(seed)
    all_cells = list(itertools.product(range(len(AGES)), range(len(GENDERS)), range(len(ETHNICITIES))))
    rng.shuffle(all_cells)

    picked: list[tuple[int, int, int]] = []
    age_count = [0] * len(AGES)
    gender_count = [0] * len(GENDERS)
    eth_count = [0] * len(ETHNICITIES)
    target_age, target_gender, target_eth = 5, 8, 3

    # greedy: prefer cells whose levels are under-represented
    def score(c: tuple[int, int, int]) -> int:
        a, g, e = c
        return (
            max(0, target_age - age_count[a])
            + max(0, target_gender - gender_count[g])
            + max(0, target_eth - eth_count[e])
        )

    while len(picked) < n_cells and all_cells:
        all_cells.sort(key=score, reverse=True)
        c = all_cells.pop(0)
        picked.append(c)
        age_count[c[0]] += 1
        gender_count[c[1]] += 1
        eth_count[c[2]] += 1

    rows: list[PromptRow] = []
    for ai, gi, ei in picked:
        for si in range(seeds_per_cell):
            seed_val = 100000 + ai * 10000 + gi * 1000 + ei * 100 + si
            rows.append(PromptRow(
                prompt=PROMPT_TEMPLATE.format(age=AGES[ai], ethnicity=ETHNICITIES[ei], gender=GENDERS[gi]),
                seed=seed_val,
                age=AGES[ai],
                gender=GENDERS[gi],
                ethnicity=ETHNICITIES[ei],
                cell_id=f"{ai}-{gi}-{ei}",
            ))
    return rows


if __name__ == "__main__":
    sg = sanity_grid()
    print(f"sanity_grid: {len(sg)} samples across {len({r.cell_id for r in sg})} cells")
    from collections import Counter
    print(f"  ages:       {Counter(r.age for r in sg)}")
    print(f"  genders:    {Counter(r.gender for r in sg)}")
    print(f"  eths:       {Counter(r.ethnicity for r in sg)}")
    print("\nfirst 3 prompts:")
    for r in sg[:3]:
        print(f"  [{r.sample_id}] seed={r.seed}")
        print(f"    {r.prompt}")

    fg = full_grid()
    print(f"\nfull_grid: {len(fg)} samples")
