"""Re-select v2 pairs with a per-identity cluster cap.

The first v2 selection (greedy on J, per-render-cap=4) saturated the
top band with same-identity duplicate photos (e.g. ~18/32 of the top
sheet were John Kerry photos). The within-pair arc_cos < 0.5 cap
doesn't prevent the *same person* appearing across many pairs.

Fix: within each (race, gender, age_bin) cell, single-link cluster
rows by ArcFace cosine ≥ CLUSTER_THRESHOLD. During greedy selection,
each identity cluster contributes at most CLUSTER_CAP pairs.

Outputs a fresh pair_manifest.parquet (replacing the previous one)
with FFHQ rows re-selected, and rebuilds contact sheets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REVERSE_INDEX = REPO / "output/reverse_index/reverse_index.parquet"
V2_PAIRS = REPO / "output/solver_c_ffhq_v2/ffhq_squint_pairs.parquet"
V2_INDEX = REPO / "output/solver_c_ffhq_v2/ffhq_squint_v2_index.parquet"
PAIR_MANIFEST = REPO / "output/squint_path_b/pair_manifest.parquet"

CLUSTER_THRESHOLD = 0.40
CLUSTER_CAP = 1          # max pairs per identity cluster
PER_RENDER_CAP = 4       # original cap retained
N_BUDGET = 2000
DSMILE_MAX = 0.30


def cluster_within_cells(df: pd.DataFrame, arc: np.ndarray, threshold: float) -> np.ndarray:
    """Single-link cluster within each cell by arc_cos ≥ threshold.
    Returns cluster_id (int) per row, unique across the whole DataFrame."""
    cluster_id = np.full(len(df), -1, dtype=np.int64)
    next_cluster = 0
    for _, cell in df.groupby(["ff_race", "ff_gender", "ff_age_bin"], sort=False):
        idx = cell.index.to_numpy()
        if len(idx) == 0:
            continue
        a = arc[idx]
        sim = a @ a.T
        np.fill_diagonal(sim, -np.inf)
        # union-find
        parent = np.arange(len(idx))
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a_: int, b_: int) -> None:
            ra, rb = find(a_), find(b_)
            if ra != rb:
                parent[ra] = rb
        ii, jj = np.where(sim >= threshold)
        for i, j in zip(ii, jj):
            if i < j:
                union(int(i), int(j))
        roots = np.array([find(i) for i in range(len(idx))])
        # remap roots to globally-unique ids
        unique_roots, inv = np.unique(roots, return_inverse=True)
        cluster_id[idx] = next_cluster + inv
        next_cluster += len(unique_roots)
    return cluster_id


def main() -> None:
    print("[load] v2 pairs + sha index + reverse_index")
    pairs = pd.read_parquet(V2_PAIRS)
    sha_idx = pd.read_parquet(V2_INDEX)
    ri_cols = ["image_sha256", "arcface_fp32",
               "ff_race", "ff_gender", "ff_age_bin",
               "bs_eyeSquintLeft", "bs_eyeSquintRight",
               "bs_mouthSmileLeft", "bs_mouthSmileRight"]
    ri = pd.read_parquet(REVERSE_INDEX, columns=ri_cols)

    # rebuild the v2 row pool: sha index has row_id -> sha; merge ri on sha
    df = sha_idx.merge(ri, on="image_sha256", how="left").sort_values("row_id").reset_index(drop=True)
    df["theta"] = df["bs_eyeSquintLeft"] + df["bs_eyeSquintRight"]
    df["smile"] = df["bs_mouthSmileLeft"] + df["bs_mouthSmileRight"]
    print(f"[df] {len(df)} rows in v2 pool")

    # ArcFace matrix (L2-normalized — same as solver_c)
    arc = np.stack([np.asarray(v, dtype=np.float32) for v in df["arcface_fp32"]])
    arc = arc / np.maximum(np.linalg.norm(arc, axis=1, keepdims=True), 1e-8)

    print(f"[cluster] single-link within-cell at arc_cos ≥ {CLUSTER_THRESHOLD}")
    cluster_id = cluster_within_cells(df, arc, CLUSTER_THRESHOLD)
    n_clusters = len(np.unique(cluster_id))
    sizes = np.bincount(cluster_id - cluster_id.min())
    big = (sizes > 1).sum()
    print(f"[cluster] {n_clusters} clusters total, {big} with >1 member, "
          f"max size {sizes.max()}")

    # Greedy re-selection with cluster cap + per-render cap
    order = np.argsort(-pairs["J"].to_numpy())
    i_arr = pairs["i"].to_numpy()
    j_arr = pairs["j"].to_numpy()
    chosen = np.zeros(len(pairs), dtype=bool)
    cluster_count: dict[int, int] = {}
    render_count: dict[int, int] = {}
    n = 0
    for idx in order:
        i, j = int(i_arr[idx]), int(j_arr[idx])
        ci, cj = int(cluster_id[i]), int(cluster_id[j])
        if cluster_count.get(ci, 0) >= CLUSTER_CAP or cluster_count.get(cj, 0) >= CLUSTER_CAP:
            continue
        if render_count.get(i, 0) >= PER_RENDER_CAP or render_count.get(j, 0) >= PER_RENDER_CAP:
            continue
        chosen[idx] = True
        cluster_count[ci] = cluster_count.get(ci, 0) + 1
        if cj != ci:
            cluster_count[cj] = cluster_count.get(cj, 0) + 1
        render_count[i] = render_count.get(i, 0) + 1
        render_count[j] = render_count.get(j, 0) + 1
        n += 1
        if n >= N_BUDGET:
            break
    print(f"[select] {n} pairs after cluster_cap={CLUSTER_CAP}, "
          f"render_cap={PER_RENDER_CAP}, budget={N_BUDGET}")

    sel = pairs[chosen].copy().reset_index(drop=True)

    # Compute pos/neg orientation, abs_dsmile, arc_cos
    i_sel = sel["i"].to_numpy()
    j_sel = sel["j"].to_numpy()
    dtheta = sel["dtheta"].to_numpy()
    pos_idx = np.where(dtheta > 0, i_sel, j_sel)
    neg_idx = np.where(dtheta > 0, j_sel, i_sel)
    sha = df["image_sha256"].to_numpy()
    theta = df["theta"].to_numpy()
    smile = df["smile"].to_numpy()
    abs_dsmile = np.abs(smile[pos_idx] - smile[neg_idx])
    arc_cos = np.einsum("ij,ij->i", arc[pos_idx], arc[neg_idx])

    out = pd.DataFrame({
        "rank": np.arange(len(sel)),
        "sha_pos": sha[pos_idx], "sha_neg": sha[neg_idx],
        "ff_race": sel["ff_race"].to_numpy(),
        "ff_gender": sel["ff_gender"].to_numpy(),
        "ff_age_bin": sel["ff_age_bin"].to_numpy(),
        "theta_pos": theta[pos_idx], "theta_neg": theta[neg_idx],
        "abs_dtheta": np.abs(dtheta),
        "smile_pos": smile[pos_idx], "smile_neg": smile[neg_idx],
        "abs_dsmile": abs_dsmile,
        "deta_sq": sel["deta_sq"].to_numpy(),
        "J": sel["J"].to_numpy(),
        "arc_cos": arc_cos,
        "cluster_pos": cluster_id[pos_idx],
        "cluster_neg": cluster_id[neg_idx],
    })
    keep = out["abs_dsmile"] <= DSMILE_MAX
    out["kept"] = keep
    out["source"] = "ffhq"
    print(f"[filter] |Δsmile| ≤ {DSMILE_MAX} → kept {int(keep.sum())}/{len(out)}")

    # Carry forward grid pairs from existing manifest unchanged
    if PAIR_MANIFEST.exists():
        existing = pd.read_parquet(PAIR_MANIFEST)
        if "source" not in existing.columns:
            existing["source"] = "ffhq"
        grid = existing[existing["source"] == "flux_solver_a_grid_squint"].copy()
        if len(grid):
            for c in ("cluster_pos", "cluster_neg"):
                if c not in grid.columns:
                    grid[c] = -1
            print(f"[manifest] carrying forward {len(grid)} grid pairs")
            for c in out.columns:
                if c not in grid.columns:
                    grid[c] = np.nan
            for c in grid.columns:
                if c not in out.columns:
                    out[c] = np.nan
            out = out[grid.columns]
            merged = pd.concat([out, grid], ignore_index=True)
        else:
            merged = out
    else:
        merged = out

    merged.to_parquet(PAIR_MANIFEST)
    by = merged[merged["kept"]]["source"].value_counts().to_dict()
    print(f"[manifest] wrote {PAIR_MANIFEST} ({len(merged)} rows, kept by source: {by})")

    # Updated needed_shas.txt for any extra fetch (should be subset of existing)
    needed = pd.unique(pd.concat([
        merged.loc[merged["kept"], "sha_pos"],
        merged.loc[merged["kept"], "sha_neg"],
    ], ignore_index=True))
    img_dir = REPO / "output/ffhq_images"
    on_disk = {p.stem for p in img_dir.glob("*.png")}
    missing = sorted(set(needed) - on_disk)
    print(f"[shas] {len(needed)} unique needed, {len(missing)} missing on disk")
    if missing:
        Path(REPO / "output/squint_path_b/needed_shas_post_dedup.txt").write_text(
            "\n".join(missing) + "\n"
        )


if __name__ == "__main__":
    main()
