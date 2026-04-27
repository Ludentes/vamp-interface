"""Extract Flux conditioning vectors for Stage 2b prompts.

Mirrors stage2b_conditioning.py but over Stage 2b's smile+glasses grid.
Output aligned to all_rows() sample_id order:
    output/demographic_pc/conditioning_2b.npy        (N, 4864) float32
    output/demographic_pc/conditioning_2b_ids.json

Usage:
    uv run python -m src.demographic_pc.stage2b_conditioning_2b
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.demographic_pc.prompts_stage2b import all_rows
from src.demographic_pc.stage2b_conditioning import (
    CLIP_L_SAFETENSORS, POOLED_DIM, T5_SAFETENSORS,
    encode_batch, load_clip, load_t5,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc"


def main() -> None:
    for p in (CLIP_L_SAFETENSORS, T5_SAFETENSORS):
        if not p.exists():
            raise FileNotFoundError(p)
    rows = all_rows(seeds_per_cell=1)
    print(f"[stage2b_cond] encoding {len(rows)} prompts")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    clip_tok, clip_model = load_clip(device, dtype)
    t5_tok, t5_model = load_t5(device, dtype)

    vecs = np.empty((len(rows), POOLED_DIM), dtype=np.float32)
    ids = [r.sample_id for r in rows]
    prompts = [r.prompt for r in rows]
    batch = 8
    t0 = time.time()
    for i in range(0, len(rows), batch):
        b = prompts[i : i + batch]
        vecs[i : i + len(b)] = encode_batch(b, clip_tok, clip_model, t5_tok, t5_model, device)
        done = i + len(b)
        if done % (batch * 5) == 0 or done == len(rows):
            dt = time.time() - t0
            rate = done / dt
            print(f"  [{done}/{len(rows)}] rate={rate:.1f}/s  eta={(len(rows)-done)/rate/60:.1f}min")

    np.save(OUT_DIR / "conditioning_2b.npy", vecs)
    with open(OUT_DIR / "conditioning_2b_ids.json", "w") as f:
        json.dump(ids, f)
    print(f"[stage2b_cond] wrote conditioning_2b.npy {vecs.shape}")


if __name__ == "__main__":
    main()
