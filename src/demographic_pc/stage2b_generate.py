"""Stage 2b — render smile/glasses prompt grids.

Same regime as Stage 2 (img2img from Stage 1 anchor at denoise=0.9) so the
resulting conditioning distribution is commensurable with Stage 2.
Resumable (skips existing PNGs).

Usage:
    uv run python -m src.demographic_pc.stage2b_generate
    uv run python -m src.demographic_pc.stage2b_generate --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.demographic_pc.comfy_flux import ComfyClient, flux_img2img_workflow
from src.demographic_pc.prompts_stage2b import all_rows

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc"
SAMPLES_DIR = OUT_DIR / "samples_2b"
ANCHOR_PATH = OUT_DIR / "stage1" / "anchor_768x1024.png"

DENOISE = 0.90


async def run(limit: int | None = None) -> None:
    if not ANCHOR_PATH.exists():
        raise FileNotFoundError(ANCHOR_PATH)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = all_rows(seeds_per_cell=1)
    if limit:
        rows = rows[:limit]

    manifest = {
        r.sample_id: {
            "prompt": r.prompt, "seed": r.seed, "axis": r.axis, "level_key": r.level_key,
            "age": r.age, "gender": r.gender, "ethnicity": r.ethnicity,
        } for r in rows
    }
    with open(OUT_DIR / "manifest_2b.json", "w") as f:
        json.dump(manifest, f, indent=2)

    done = sum(1 for r in rows if (SAMPLES_DIR / f"{r.sample_id}.png").exists())
    todo = len(rows) - done
    print(f"[stage2b/gen] {len(rows)} rows  already={done}  to generate={todo}")

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
    print(f"[stage2b/gen] done. samples in {SAMPLES_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(limit=args.limit))


if __name__ == "__main__":
    main()
