"""Stage 2b (black) — race-isolation grid.

4 ages × 5 race clauses × 2 genders × 3 seeds = 120 renders.
Same img2img regime as Stage 2b (denoise=0.9, anchor img2img).

Usage:
    uv run python -m src.demographic_pc.stage2b_generate_black
    uv run python -m src.demographic_pc.stage2b_generate_black --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient, flux_img2img_workflow
from src.demographic_pc.prompts_stage2b import black_grid

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc"
SAMPLES_DIR = OUT_DIR / "samples_2b"  # share directory with smile/glasses
ANCHOR_PATH = OUT_DIR / "stage1" / "anchor_768x1024.png"

DENOISE = 0.90


async def run(limit: int | None = None, seeds_per_cell: int = 3) -> None:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(ANCHOR_PATH)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = black_grid(seeds_per_cell=seeds_per_cell)
    if limit:
        rows = rows[:limit]

    manifest_path = OUT_DIR / "manifest_2b_black.json"
    manifest = {
        r.sample_id: {
            "prompt": r.prompt, "seed": r.seed, "axis": r.axis, "level_key": r.level_key,
            "age": r.age, "gender": r.gender, "ethnicity": r.ethnicity,
        } for r in rows
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    done = sum(1 for r in rows if (SAMPLES_DIR / f"{r.sample_id}.png").exists())
    todo = len(rows) - done
    print(f"[stage2b/black] {len(rows)} rows  already={done}  to generate={todo}")

    async with ComfyClient() as client:
        anchor_name = await client.upload_image(ANCHOR_PATH)
        t0 = time.time()
        generated = 0
        for i, row in enumerate(rows, 1):
            dest = SAMPLES_DIR / f"{row.sample_id}.png"
            if dest.exists():
                continue
            wf = flux_img2img_workflow(
                image_name=anchor_name, positive=row.prompt, seed=row.seed,
                denoise=DENOISE, prefix=f"s2b_{row.sample_id}",
            )
            await client.generate(wf, dest)
            generated += 1
            if generated % 10 == 0 or generated == todo:
                dt = time.time() - t0
                rate = generated / dt
                eta = (todo - generated) / rate if rate > 0 else 0
                print(f"  [{i:4d}/{len(rows)}] gen={generated}/{todo}  rate={rate:.2f}/s  eta={eta/60:.1f}min")
    print(f"[stage2b/black] done. samples in {SAMPLES_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds-per-cell", type=int, default=3)
    args = ap.parse_args()
    asyncio.run(run(limit=args.limit, seeds_per_cell=args.seeds_per_cell))


if __name__ == "__main__":
    main()
