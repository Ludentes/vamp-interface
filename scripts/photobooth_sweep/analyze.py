"""Phase 1 analysis — per-axis effects on id_cos + robust default selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent
SCORES = ROOT / "exp_output/photobooth_phase1/scores.parquet"


def main():
    t = pq.read_table(SCORES).to_pandas()
    print(f"# Phase 1 — {len(t)} cells, {t.photo_id.nunique()} photos")
    print(f"\n## Per-photo coverage")
    print(t.groupby("photo_id").size().to_string())

    print(f"\n## det_mode")
    print(t.groupby("det_mode").size().to_string())

    print(f"\n## id_cos overall")
    print(f"  mean   = {t.id_cos.mean():.3f}")
    print(f"  median = {t.id_cos.median():.3f}")
    print(f"  p05    = {t.id_cos.quantile(0.05):.3f}")
    print(f"  p25    = {t.id_cos.quantile(0.25):.3f}")
    print(f"  p75    = {t.id_cos.quantile(0.75):.3f}")
    print(f"  p95    = {t.id_cos.quantile(0.95):.3f}")
    print(f"  max    = {t.id_cos.max():.3f}")

    print(f"\n## id_cos by photo")
    print(t.groupby("photo_id").id_cos.agg(["mean", "median", "max"]).round(3).to_string())

    for axis in ["face_pixel_budget", "cn_condition", "canny_preset",
                 "refine_denoise", "demo_inject"]:
        print(f"\n## id_cos by {axis}")
        g = t.groupby(axis).id_cos.agg(["count", "mean", "median", "max"]).round(3)
        print(g.to_string())

    # cn_strength: bin into thirds
    print(f"\n## id_cos by cn_strength bin")
    t2 = t.copy()
    t2["cn_strength_bin"] = np.where(t.cn_strength < 0.867, "0.80-0.87",
                            np.where(t.cn_strength < 0.933, "0.87-0.93", "0.93-1.00"))
    print(t2.groupby("cn_strength_bin").id_cos.agg(["count", "mean", "median", "max"]).round(3).to_string())

    print(f"\n## Top 10 cells (det_mode==default only)")
    default = t[t.det_mode == "default"].copy()
    cols = ["cell_id", "face_pixel_budget", "cn_condition", "canny_preset",
            "cn_strength", "refine_denoise", "demo_inject", "id_cos",
            "face_frac", "det_score"]
    print(default.sort_values("id_cos", ascending=False).head(10)[cols].round(3).to_string(index=False))

    print(f"\n## Bottom 10 cells (any det_mode)")
    print(t.sort_values("id_cos").head(10)[cols + ["det_mode"]].round(3).to_string(index=False))

    print(f"\n## Robust default — config maximizing 5th-percentile id_cos per photo")
    # Group by exact axis combo, compute p5 across photos
    cfg_cols = ["face_pixel_budget", "cn_condition", "canny_preset",
                "refine_denoise", "demo_inject"]
    # cn_strength is continuous — bin first
    t2["cs_bin"] = pd.cut(t2.cn_strength, bins=[0.79, 0.87, 0.93, 1.01],
                          labels=["lo", "mid", "hi"]).astype(str) if False else \
                   np.where(t.cn_strength < 0.867, "lo",
                   np.where(t.cn_strength < 0.933, "mid", "hi"))
    # we don't have enough cells/photo to do full p5 — just rank by per-photo mean
    grp = t2.groupby(cfg_cols + ["cs_bin"]).agg(
        n=("id_cos", "count"), mean_id=("id_cos", "mean"),
        min_id=("id_cos", "min"), photos=("photo_id", "nunique")
    ).sort_values("min_id", ascending=False)
    multi = grp[grp.photos >= 2].head(10)
    print(multi.round(3).to_string())


if __name__ == "__main__":
    import pandas as pd  # noqa: F401 — only imported lazily for the optional path
    main()
