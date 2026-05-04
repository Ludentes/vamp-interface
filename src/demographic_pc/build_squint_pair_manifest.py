"""Build the v9-squint Path B trainer pair manifest.

Reads selected pairs from `output/solver_c_ffhq/ffhq_squint_pairs.parquet`,
maps integer (i, j) row indices back to image_sha256 by replicating the
exact filter chain from `solver_c_squint_feasibility_ffhq.py`, orients
each pair so `pos` is the high-squint side, attaches diagnostics
(ArcFace cosine, |Δsmile|, cell), applies an optional |Δsmile| hard
filter (FFHQ-rescue doc mitigation #2), and writes:

  output/squint_path_b/pair_manifest.parquet
  output/squint_path_b/needed_shas.txt
  output/squint_path_b/manifest_summary.md

The downstream image fetch needs only `needed_shas.txt`. The trainer
needs only `pair_manifest.parquet` plus the on-disk PNGs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REVERSE_INDEX = ROOT / "output/reverse_index/reverse_index.parquet"
PAIRS_IN = ROOT / "output/solver_c_ffhq_v2/ffhq_squint_pairs.parquet"
SHA_INDEX_IN = ROOT / "output/solver_c_ffhq_v2/ffhq_squint_v2_index.parquet"
OUT_DIR = ROOT / "output/squint_path_b"

# |Δsmile| hard filter: keep pairs whose smile blendshape difference is
# below this. Tuned to retain ≥ 500 pairs after filtering. Smile metric
# is bs_mouthSmileLeft + bs_mouthSmileRight, same definition the
# feasibility script used inside η.
DSMILE_THRESHOLD = 0.30
MIN_KEPT_TARGET = 500


def load_filtered_ffhq() -> pd.DataFrame:
    """Load v2 sha index (row_id -> sha mapping after v2's contamination
    filter chain) and re-merge per-row blendshape scalars from
    reverse_index for downstream pos/neg orientation + |Δsmile| filter.
    """
    sha_idx = pd.read_parquet(SHA_INDEX_IN)
    cols = [
        "image_sha256",
        "bs_eyeSquintLeft", "bs_eyeSquintRight",
        "bs_mouthSmileLeft", "bs_mouthSmileRight",
        "arcface_fp32",
    ]
    ri = pd.read_parquet(REVERSE_INDEX, columns=cols)
    df = sha_idx.merge(ri, on="image_sha256", how="left")
    df = df.sort_values("row_id").reset_index(drop=True)
    df["theta"] = df["bs_eyeSquintLeft"] + df["bs_eyeSquintRight"]
    df["smile"] = df["bs_mouthSmileLeft"] + df["bs_mouthSmileRight"]
    return df


def arc_unit(df: pd.DataFrame) -> np.ndarray:
    arc = np.stack([np.asarray(v, dtype=np.float32) for v in df["arcface_fp32"]])
    arc = arc / np.maximum(np.linalg.norm(arc, axis=1, keepdims=True), 1e-8)
    return arc


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_filtered_ffhq()
    print(f"[ffhq] {len(df)} rows after filter")
    arc = arc_unit(df)

    pairs = pd.read_parquet(PAIRS_IN)
    sel = pairs[pairs["selected"]].copy().reset_index(drop=True)
    print(f"[pairs] {len(sel)} selected from {len(pairs)} total")

    if int(sel["i"].max()) >= len(df) or int(sel["j"].max()) >= len(df):
        raise RuntimeError(
            f"index out of range: max(i,j)={max(int(sel['i'].max()), int(sel['j'].max()))} "
            f"vs ffhq filtered len={len(df)} — filter chain drift"
        )

    i_arr = sel["i"].to_numpy()
    j_arr = sel["j"].to_numpy()
    dtheta = sel["dtheta"].to_numpy()
    # dtheta = theta[i] - theta[j]: dtheta > 0 means i has more squint
    pos_idx = np.where(dtheta > 0, i_arr, j_arr)
    neg_idx = np.where(dtheta > 0, j_arr, i_arr)

    sha = df["image_sha256"].to_numpy()
    theta = df["theta"].to_numpy()
    smile = df["smile"].to_numpy()

    sha_pos = sha[pos_idx]
    sha_neg = sha[neg_idx]
    theta_pos = theta[pos_idx]
    theta_neg = theta[neg_idx]
    smile_pos = smile[pos_idx]
    smile_neg = smile[neg_idx]
    abs_dsmile = np.abs(smile_pos - smile_neg)
    arc_cos = np.einsum("ij,ij->i", arc[pos_idx], arc[neg_idx])

    out = pd.DataFrame({
        "rank": np.arange(len(sel)),
        "sha_pos": sha_pos,
        "sha_neg": sha_neg,
        "ff_race": sel["ff_race"].to_numpy(),
        "ff_gender": sel["ff_gender"].to_numpy(),
        "ff_age_bin": sel["ff_age_bin"].to_numpy(),
        "theta_pos": theta_pos,
        "theta_neg": theta_neg,
        "abs_dtheta": np.abs(dtheta),
        "smile_pos": smile_pos,
        "smile_neg": smile_neg,
        "abs_dsmile": abs_dsmile,
        "deta_sq": sel["deta_sq"].to_numpy(),
        "J": sel["J"].to_numpy(),
        "arc_cos": arc_cos,
    })

    # |Δsmile| filter — drop pairs above threshold, but only if we keep
    # at least MIN_KEPT_TARGET. Otherwise relax threshold to the value
    # that retains exactly MIN_KEPT_TARGET.
    keep_mask = out["abs_dsmile"] <= DSMILE_THRESHOLD
    n_kept = int(keep_mask.sum())
    if n_kept >= MIN_KEPT_TARGET:
        threshold_used = DSMILE_THRESHOLD
    else:
        threshold_used = float(np.sort(out["abs_dsmile"].to_numpy())[MIN_KEPT_TARGET - 1])
        keep_mask = out["abs_dsmile"] <= threshold_used
        n_kept = int(keep_mask.sum())
        print(f"[filter] threshold {DSMILE_THRESHOLD} would keep only {keep_mask.sum()} "
              f"< {MIN_KEPT_TARGET}; relaxed to {threshold_used:.4f}")

    out["kept"] = keep_mask
    print(f"[filter] |Δsmile| ≤ {threshold_used:.4f} → kept {n_kept}/{len(out)}")

    out.to_parquet(OUT_DIR / "pair_manifest.parquet")

    needed = pd.unique(pd.concat([
        out.loc[keep_mask, "sha_pos"],
        out.loc[keep_mask, "sha_neg"],
    ], ignore_index=True))
    (OUT_DIR / "needed_shas.txt").write_text("\n".join(map(str, needed)) + "\n")
    print(f"[shas] {len(needed)} unique sha256s needed")

    kept = out[keep_mask]
    summary: list[str] = [
        "# v9-squint Path B pair manifest",
        "",
        f"- source pairs: `{PAIRS_IN.name}` ({len(pairs):,} candidate, {len(sel):,} selected)",
        f"- |Δsmile| threshold: {threshold_used:.4f} (target: {DSMILE_THRESHOLD}, min retain: {MIN_KEPT_TARGET})",
        f"- pairs kept: **{n_kept}** / {len(out)}",
        f"- unique image_sha256 needed: **{len(needed)}**",
        "",
        "## Distributions (kept subset)",
        "",
        "| metric | min | p25 | median | p75 | max |",
        "|---|---|---|---|---|---|",
    ]
    for col in ("abs_dtheta", "abs_dsmile", "arc_cos", "J"):
        v = kept[col].to_numpy()
        summary.append(
            f"| {col} | {v.min():.4f} | {np.percentile(v, 25):.4f} | "
            f"{np.median(v):.4f} | {np.percentile(v, 75):.4f} | {v.max():.4f} |"
        )

    summary += [
        "",
        "## Per-cell counts (kept)",
        "",
        "| race | gender | age | n_pairs | median |Δθ| | median arc_cos |",
        "|---|---|---|---|---|---|",
    ]
    for (r, g, a), grp in kept.groupby(["ff_race", "ff_gender", "ff_age_bin"]):
        summary.append(
            f"| {r} | {g} | {a} | {len(grp)} | "
            f"{grp['abs_dtheta'].median():.3f} | {grp['arc_cos'].median():.3f} |"
        )

    summary += [
        "",
        "## Outputs",
        "",
        f"- `pair_manifest.parquet` — full {len(out)} rows with `kept` flag (training reads kept==True only)",
        f"- `needed_shas.txt` — {len(needed)} unique image_sha256 values to fetch from PC 25 FFHQ shards",
        "",
        "## Next stage",
        "",
        "1. Ship `needed_shas.txt` to PC 25, run shard-scanner, save `output/ffhq_images/{sha}.png` at 512×512 LANCZOS, rsync back.",
        "2. Build sanity collage of top-J 32 kept pairs.",
        "3. Emit ai-toolkit Path B trainer config + dataset CSV.",
    ]

    (OUT_DIR / "manifest_summary.md").write_text("\n".join(summary))
    print(f"[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
