"""Stage 4.5 — render 20 held-out portraits × 5 λ × 2 methods.

Grid:
  - 20 unique {gender, ethnicity} combinations at age='adult' (midpoint)
  - fresh seeds (2000..2019), unused in Stage 2 training
  - λ ∈ {−1.0, −0.5, 0.0, +0.5, +1.0}
  - methods: "ours", "fluxspace"
  - λ=0 is method-agnostic (no-op edit) — rendered once per portrait under method="baseline"

Output layout:
  output/demographic_pc/stage4_5/renders/{method}/{portrait_id}__lam{+0.50}.png
    where method ∈ {baseline, ours, fluxspace}, lam with sign.

Portrait sampling:
  Stratified so every ethnicity (7) appears ≥ twice and every gender (3) ≥ 5×.
  Prompt: same template as Stage 2 with age="adult".

For 'ours', raw strength = λ × 15  (15 years per λ unit — matches the 1-year-per-unit direction).
For 'fluxspace', raw strength = λ × 1  (a full pair-magnitude per λ unit).
Evaluator will rescale post-hoc to compare at matched target-slope.

Usage:
    uv run python -m src.demographic_pc.stage4_5_render
    uv run python -m src.demographic_pc.stage4_5_render --limit 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient, flux_txt2img_workflow
from src.demographic_pc.prompts import ETHNICITIES, GENDERS, PROMPT_TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc" / "stage4_5"
RENDERS = OUT_DIR / "renders"
EDITS_DIR = ROOT / "output" / "demographic_pc" / "edits"

LAMBDAS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]  # extremes probe off-manifold / uncanny
OURS_SCALE = 45.0       # strength = λ * 45 years  (±45y at λ=±1, spans child→elderly)
FLUXSPACE_SCALE = 2.0   # strength = λ * 2 pair-magnitudes

WIDTH, HEIGHT = 768, 1024


def portrait_grid() -> list[dict]:
    """20 stratified {gender, ethnicity} portraits at age='adult'. Seeds 2000..2019."""
    rng = random.Random(42)
    cells = [(g, e) for g in GENDERS for e in ETHNICITIES]  # 21 cells
    rng.shuffle(cells)
    picked = cells[:20]  # 20 unique (g,e) cells
    portraits = []
    for i, (gender, ethnicity) in enumerate(picked):
        seed = 2000 + i
        portraits.append({
            "portrait_id": f"p{i:02d}_{GENDERS.index(gender)}_{ETHNICITIES.index(ethnicity)}_s{seed}",
            "gender": gender,
            "ethnicity": ethnicity,
            "seed": seed,
            "prompt": PROMPT_TEMPLATE.format(age="adult", ethnicity=ethnicity, gender=gender),
        })
    return portraits


async def run_batch(client: ComfyClient, jobs: list[dict]) -> None:
    t0 = time.time()
    done = 0
    for job in jobs:
        dest = Path(job["dest"])
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        wf = flux_txt2img_workflow(
            positive=job["prompt"], seed=job["seed"],
            width=WIDTH, height=HEIGHT, prefix=f"s45_{job['method']}_{job['portrait_id']}_l{job['lam']:+.2f}",
            edit_npz_path=job["edit_npz_path"], edit_strength=job["strength"],
        )
        await client.generate(wf, dest)
        done += 1
        if done % 5 == 0:
            dt = time.time() - t0
            print(f"  [{done}/{len(jobs)}]  rate={done/dt:.2f}/s  eta={(len(jobs)-done)/(done/dt)/60:.1f}min")


async def run(limit: int | None = None) -> None:
    portraits = portrait_grid()
    if limit:
        portraits = portraits[:limit]

    edits = {
        "ours": str(EDITS_DIR / "age_ours.npz"),
        "fluxspace": str(EDITS_DIR / "age_fluxspace_coarse.npz"),
    }
    scales = {"ours": OURS_SCALE, "fluxspace": FLUXSPACE_SCALE}

    jobs: list[dict] = []
    for p in portraits:
        # Baseline (λ=0) — one render shared across methods
        jobs.append({
            "portrait_id": p["portrait_id"], "prompt": p["prompt"], "seed": p["seed"],
            "method": "baseline", "lam": 0.0, "strength": 0.0,
            "edit_npz_path": edits["ours"],  # file provided so node is present, strength=0 no-op
            "dest": RENDERS / "baseline" / f"{p['portrait_id']}__lam+0.00.png",
        })
        for method, path in edits.items():
            scale = scales[method]
            for lam in LAMBDAS:
                if lam == 0.0:
                    continue  # baseline covers it
                jobs.append({
                    "portrait_id": p["portrait_id"], "prompt": p["prompt"], "seed": p["seed"],
                    "method": method, "lam": lam, "strength": lam * scale,
                    "edit_npz_path": path,
                    "dest": RENDERS / method / f"{p['portrait_id']}__lam{lam:+.2f}.png",
                })

    # Save manifest for evaluator
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = [
        {k: str(v) if isinstance(v, Path) else v for k, v in j.items()} for j in jobs
    ]
    with open(OUT_DIR / "render_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[stage4.5] {len(portraits)} portraits × (baseline + 2 methods × 4 λ) = {len(jobs)} renders")

    async with ComfyClient() as client:
        await run_batch(client, jobs)
    print(f"[stage4.5] done. manifest: {OUT_DIR / 'render_manifest.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(limit=args.limit))


if __name__ == "__main__":
    main()
