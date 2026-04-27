"""Stamp max_env = max|attn_base + s·δ_mix| per row with has_attn=True.

s = row.scale where finite, else 1.0 (α-sweep rows where scale isn't the
knob — we evaluate at the render's native parameters, which is s=1 in the
cached δ_mix).

Reads from models/blendshape_nmf/attn_cache/{tag}/ (fp16 arrays pre-built
by cache_attn_features.py). Writes max_env as a new column.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "models/blendshape_nmf/sample_index.parquet"
CACHE_ROOT = ROOT / "models/blendshape_nmf/attn_cache"


def main() -> None:
    idx = pd.read_parquet(INDEX)
    has = idx[idx["has_attn"] == True]  # noqa: E712
    tags = sorted(has["attn_tag"].dropna().unique())
    print(f"[stamp] tags: {tags}")

    max_env = np.full(len(idx), np.nan, dtype=np.float32)

    for tag in tags:
        cache_dir = CACHE_ROOT / tag
        meta = json.loads((cache_dir / "meta.json").read_text())
        dm = np.load(cache_dir / "delta_mix.npy")   # (N, steps, blocks, D) fp16
        ab = np.load(cache_dir / "attn_base.npy")   # same shape
        print(f"  [{tag}] cache shape {dm.shape}  kept={len(meta['rels'])}")

        sub = has[has["attn_tag"] == tag]
        for ix, row in sub.iterrows():
            r = int(row["attn_row"])
            if r < 0 or r >= dm.shape[0]:
                continue
            s = row["scale"]
            s = float(s) if pd.notna(s) else 1.0
            # fp32 for the arithmetic; the arrays are tiny per-row (steps·blocks·D ≤ 3M fp32)
            steered = ab[r].astype(np.float32) + s * dm[r].astype(np.float32)
            max_env[ix] = float(np.abs(steered).max())

    n_filled = np.isfinite(max_env).sum()
    idx["max_env"] = max_env
    idx.to_parquet(INDEX, index=False, compression="zstd")
    print(f"[save] → {INDEX}  max_env filled={n_filled}/{len(idx)}")
    finite = max_env[np.isfinite(max_env)]
    print(f"[stats] max_env: min={finite.min():.3f}  p50={np.median(finite):.3f}  "
          f"p95={np.percentile(finite, 95):.3f}  max={finite.max():.3f}")


if __name__ == "__main__":
    main()
