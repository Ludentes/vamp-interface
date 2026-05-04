"""Drop pairs whose pos sha shows full eye closure rather than squint.

Per-source blink thresholds (from blink_vs_squint.png analysis):
  ffhq : bs_eyeBlinkL+R > 0.5  (natural FFHQ ceiling ~0.49)
  grid : bs_eyeBlinkL+R > 0.4  (Flux squint↔blink diagonal contamination)

Modifies pair_manifest.parquet in place by flipping `kept` to False on
matching rows (preserves them for audit, doesn't delete).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "output/squint_path_b/pair_manifest.parquet"
REVERSE_INDEX = REPO / "output/reverse_index/reverse_index.parquet"

THR_FFHQ = 0.5
THR_GRID = 0.4


def main() -> None:
    m = pd.read_parquet(MANIFEST)
    print(f"[manifest] {len(m)} rows ({int(m['kept'].sum())} kept) before filter")

    ri = pd.read_parquet(REVERSE_INDEX,
                         columns=["image_sha256", "bs_eyeBlinkLeft", "bs_eyeBlinkRight"])
    ri["blink"] = ri["bs_eyeBlinkLeft"] + ri["bs_eyeBlinkRight"]
    blink_by_sha = dict(zip(ri["image_sha256"], ri["blink"]))

    m["blink_pos"] = m["sha_pos"].map(blink_by_sha)
    n_missing = int(m["blink_pos"].isna().sum())
    if n_missing:
        print(f"[warn] {n_missing} pos shas missing blink (treated as 0)")
        m["blink_pos"] = m["blink_pos"].fillna(0.0)

    is_ffhq = m["source"] == "ffhq"
    is_grid = m["source"] == "flux_solver_a_grid_squint"
    drop = (
        (is_ffhq & (m["blink_pos"] > THR_FFHQ))
        | (is_grid & (m["blink_pos"] > THR_GRID))
    )
    drop_kept = drop & m["kept"]
    print(f"[filter] thresholds: ffhq>{THR_FFHQ}, grid>{THR_GRID}")
    print(f"  ffhq dropped: {int((is_ffhq & drop_kept).sum())}")
    print(f"  grid dropped: {int((is_grid & drop_kept).sum())}")
    print(f"  total kept→False: {int(drop_kept.sum())}")

    m.loc[drop_kept, "kept"] = False
    m.loc[drop_kept, "drop_reason"] = "closed_eye"
    if "drop_reason" not in m.columns:
        m["drop_reason"] = pd.NA

    by_src = m[m["kept"]]["source"].value_counts().to_dict()
    print(f"[manifest] {len(m)} rows ({int(m['kept'].sum())} kept) after filter")
    print(f"[manifest] kept by source: {by_src}")

    m = m.drop(columns=["blink_pos"])
    m.to_parquet(MANIFEST)
    print(f"[manifest] wrote {MANIFEST}")


if __name__ == "__main__":
    main()
