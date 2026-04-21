"""Stage 2 — 1785-sample demographic-PC generation on Flux Krea.

Reuses the Stage 1 anchor (output/demographic_pc/stage1/anchor_768x1024.png).
img2img at denoise=0.9 across the full 5×3×7×17 grid. Resumable — skips any
{sample_id}.png that already exists.

Expected runtime: ~5.5s/face × 1785 ≈ 2.7h on RTX 5090. Extrapolation from
Stage 1's 5.5 min for 50 samples.

Usage:
    uv run python -m src.demographic_pc.stage2_generate
    uv run python -m src.demographic_pc.stage2_generate --limit 100  # subset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient, flux_img2img_workflow
from src.demographic_pc.prompts import full_grid

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc"
SAMPLES_DIR = OUT_DIR / "samples"
ANCHOR_PATH = OUT_DIR / "stage1" / "anchor_768x1024.png"

DENOISE = 0.90


async def run(limit: int | None = None) -> None:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(f"Stage 1 anchor not found: {ANCHOR_PATH}. Run stage1_sanity first.")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = full_grid()
    if limit:
        rows = rows[:limit]

    # manifest: prompt + seed + labels per sample_id, for Stage 3/4 joins
    manifest = {
        r.sample_id: {
            "prompt": r.prompt, "seed": r.seed,
            "age": r.age, "gender": r.gender, "ethnicity": r.ethnicity,
            "cell_id": r.cell_id,
        } for r in rows
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    done = sum(1 for r in rows if (SAMPLES_DIR / f"{r.sample_id}.png").exists())
    todo = len(rows) - done
    print(f"[stage2] full_grid: {len(rows)} rows | already done: {done} | to generate: {todo}")

    async with ComfyClient() as client:
        anchor_name = await client.upload_image(ANCHOR_PATH)
        print(f"[stage2] anchor uploaded as {anchor_name}")
        t0 = time.time()
        generated = 0
        for i, row in enumerate(rows, 1):
            dest = SAMPLES_DIR / f"{row.sample_id}.png"
            if dest.exists():
                continue
            wf = flux_img2img_workflow(
                image_name=anchor_name, positive=row.prompt, seed=row.seed,
                denoise=DENOISE, prefix=f"demo_pc_s2_{row.sample_id}",
            )
            await client.generate(wf, dest)
            generated += 1
            if generated % 10 == 0 or generated == todo:
                dt = time.time() - t0
                rate = generated / dt
                eta = (todo - generated) / rate if rate > 0 else 0
                print(
                    f"  [{i:4d}/{len(rows)}] gen={generated:4d}/{todo}  "
                    f"rate={rate:.2f}/s  dt={dt:.0f}s  eta={eta/60:.1f}min"
                )
    print(f"[stage2] done. samples in {SAMPLES_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(limit=args.limit))


if __name__ == "__main__":
    main()
