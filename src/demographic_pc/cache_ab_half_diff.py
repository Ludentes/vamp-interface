"""Extend the compact attention cache with `ab_half_diff.mean_d`.

Pulls the ab_half_diff summaries from the drive-archived pkls and stores them
alongside `delta_mix.npy` / `attn_base.npy` as `ab_half_diff.npy`. Uses the
already-cached meta.json rel ordering so the arrays line up with the existing
cache row-by-row.

ab_half_diff is what the A/B edit halves disagree on (vs delta_mix, which is
what they jointly push). For an axis like anger where the A/B pair is
prompt-asymmetric, this is where the axis-specific signal should live.

Drive:  /media/newub/Seagate Hub/vamp-interface-archive/crossdemo/<axis>/<sub>/
Local:  models/blendshape_nmf/attn_cache/<tag>/  (meta.json already exists)

Usage:
  uv run python -m src.demographic_pc.cache_ab_half_diff
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "models/blendshape_nmf/attn_cache"
DRIVE = Path("/media/newub/Seagate Hub/vamp-interface-archive/crossdemo")

# Map each cached tag to its drive source directory.
TAG_TO_DRIVE_DIR = {
    "smile_inphase":       DRIVE / "smile" / "smile_inphase",
    "jaw_inphase":         DRIVE / "smile" / "jaw_inphase",
    "alpha_interp_attn":   DRIVE / "smile" / "alpha_interp_attn",
    "anger_rebalance":     DRIVE / "anger" / "rebalance",
    "surprise_rebalance":  DRIVE / "surprise" / "rebalance",
    "disgust_rebalance":   DRIVE / "disgust" / "rebalance",
    "pucker_rebalance":    DRIVE / "pucker" / "rebalance",
    "lip_press_rebalance": DRIVE / "lip_press" / "rebalance",
}


def cache_one_tag(tag: str) -> None:
    cache_dir = CACHE_ROOT / tag
    meta_path = cache_dir / "meta.json"
    out_path = cache_dir / "ab_half_diff.npy"
    if out_path.exists():
        print(f"  [skip] {tag} ab_half_diff already cached")
        return
    if not meta_path.exists():
        print(f"  [skip] {tag} no meta.json — run cache_attn_features first")
        return
    meta = json.loads(meta_path.read_text())
    rels = meta["rels"]
    step_keys = meta["step_keys"]
    block_keys = meta["block_keys"]
    D = meta["D"]

    drive_dir = TAG_TO_DRIVE_DIR[tag]
    if not drive_dir.exists():
        print(f"  [skip] {tag} drive dir missing: {drive_dir}")
        return

    print(f"  [cache] {tag}: {len(rels)} samples, reading from {drive_dir}")
    out = np.zeros((len(rels), len(step_keys), len(block_keys), D), dtype=np.float16)
    missing_leaf = 0
    skipped = 0
    for i, rel in enumerate(rels):
        pkl_path = (drive_dir / rel).with_suffix(".pkl")
        if not pkl_path.exists():
            skipped += 1
            continue
        try:
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            for si, sk in enumerate(step_keys):
                sd = d["steps"].get(sk, {})
                for bi, bk in enumerate(block_keys):
                    blk = sd.get(bk, {})
                    ab = blk.get("ab_half_diff", {}).get("mean_d")
                    if ab is None:
                        missing_leaf += 1
                        continue
                    out[i, si, bi] = (ab.numpy().astype(np.float16)
                                      if hasattr(ab, "numpy")
                                      else np.asarray(ab, dtype=np.float16))
        except Exception as e:
            print(f"    error on {rel}: {e}")
            skipped += 1
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(rels)}]")

    np.save(out_path, out)
    meta["has_ab_half_diff"] = True
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [cache] {tag} done — skipped {skipped}, leaf-missing {missing_leaf}")
    print(f"           ab_half_diff: {out.shape} {out.dtype}, {out.nbytes/1e9:.2f} GB")


def main():
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for tag in TAG_TO_DRIVE_DIR:
        cache_one_tag(tag)


if __name__ == "__main__":
    main()
