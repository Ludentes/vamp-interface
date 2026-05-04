"""Append filtered Solver A grid pairs to pair_manifest.parquet and
build a sanity collage spanning both sources.

Filters for grid inclusion (loose, designed to ride alongside FFHQ
pairs as a Flux-domain anchor signal):
  - Δsquint > 0.05  (real engagement, not measurement noise)
  - |Δsmile_siglip| < 0.05  (no smile bundle)
  - both anchor and edit faces detected
  - id_cos > 0.70  (no within-pair scale-collapse)

Grid PNGs land in output/ffhq_images/<sha>.png to share the trainer's
single-dir convention with FFHQ.

Collage:
  - top-J 16 FFHQ kept pairs (high J, post |Δsmile| filter)
  - all included grid pairs

Side-by-side rendering: each row is (pos | neg) with overlay of
source / cell / Δθ / arc_cos / J. Saved to
output/squint_path_b/sanity_collage.png and a per-row index
sanity_collage.csv.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
REVERSE_INDEX = REPO / "output" / "reverse_index" / "reverse_index.parquet"
GRID_SCORES = REPO / "output" / "demographic_pc" / "solver_a_squint_grid" / "scores.parquet"
PAIR_MANIFEST = REPO / "output" / "squint_path_b" / "pair_manifest.parquet"
IMG_DIR = REPO / "output" / "ffhq_images"
OUT_DIR = REPO / "output" / "squint_path_b"

# loose grid filters
GRID_DSQUINT_MIN = 0.05
GRID_DSMILE_SIGLIP_MAX = 0.05
GRID_ID_COS_MIN = 0.70

# collage layout
THUMB_SIZE = 256
N_FFHQ_TOP = 16
PAD = 8
HEADER_H = 36


def append_grid_to_manifest() -> pd.DataFrame:
    scores = pd.read_parquet(GRID_SCORES)
    print(f"[grid] scores: {len(scores)} pairs total")

    keep = (
        (scores["d_squint"] > GRID_DSQUINT_MIN)
        & (scores["d_smile_siglip"].abs() < GRID_DSMILE_SIGLIP_MAX)
        & scores["anchor_face_detected"]
        & scores["edit_face_detected"]
        & (scores["id_cos"] > GRID_ID_COS_MIN)
    )
    grid_kept = scores[keep].copy().reset_index(drop=True)
    print(f"[grid] after filter: {len(grid_kept)}/{len(scores)} pairs")

    # pull sha256s from reverse_index keyed by grid_seed/race/gender/age/kind
    ri_cols = [
        "image_sha256", "source", "grid_race", "grid_gender", "grid_age",
        "grid_seed", "grid_kind",
    ]
    ri = pd.read_parquet(REVERSE_INDEX, columns=ri_cols)
    ri = ri[ri["source"] == "flux_solver_a_grid_squint"].copy()
    print(f"[reverse-index] {len(ri)} grid rows available")

    # join: (race, gender, age, seed, kind=anchor) → sha_neg
    #       (race, gender, age, seed, kind=edit)   → sha_pos
    key_cols = ["grid_race", "grid_gender", "grid_age", "grid_seed"]
    anchor = ri[ri["grid_kind"] == "anchor"].rename(columns={"image_sha256": "sha_neg"})
    edit = ri[ri["grid_kind"] == "edit"].rename(columns={"image_sha256": "sha_pos"})
    grid_kept = grid_kept.rename(columns={
        "race": "grid_race", "gender": "grid_gender",
        "age": "grid_age", "seed": "grid_seed",
    })
    grid_kept = grid_kept.merge(anchor[key_cols + ["sha_neg"]], on=key_cols)
    grid_kept = grid_kept.merge(edit[key_cols + ["sha_pos"]], on=key_cols)
    print(f"[grid] after sha join: {len(grid_kept)}")

    # copy grid PNGs into IMG_DIR by sha256 (idempotent)
    n_copied = 0
    for _, r in grid_kept.iterrows():
        for sha, src_png in (
            (r["sha_neg"], r["anchor_png"]),
            (r["sha_pos"], r["edit_png"]),
        ):
            dst = IMG_DIR / f"{sha}.png"
            if dst.exists():
                continue
            img = Image.open(src_png).convert("RGB")
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            img.save(dst, format="PNG")
            n_copied += 1
    print(f"[img] copied {n_copied} grid PNGs into {IMG_DIR}")

    # build manifest rows matching pair_manifest schema
    grid_rows = pd.DataFrame({
        "rank": np.arange(len(grid_kept)),
        "sha_pos": grid_kept["sha_pos"].to_numpy(),
        "sha_neg": grid_kept["sha_neg"].to_numpy(),
        "ff_race": grid_kept["grid_race"].to_numpy(),
        "ff_gender": grid_kept["grid_gender"].to_numpy(),
        "ff_age_bin": grid_kept["grid_age"].to_numpy(),
        "theta_pos": grid_kept["edit_squint"].to_numpy(),
        "theta_neg": grid_kept["anchor_squint"].to_numpy(),
        "abs_dtheta": grid_kept["d_squint"].abs().to_numpy(),
        "smile_pos": grid_kept["edit_smile_bs"].to_numpy(),
        "smile_neg": grid_kept["anchor_smile_bs"].to_numpy(),
        "abs_dsmile": grid_kept["d_smile_bs"].abs().to_numpy(),
        "deta_sq": np.full(len(grid_kept), np.nan),
        "J": np.full(len(grid_kept), np.nan),
        "arc_cos": grid_kept["id_cos"].to_numpy(),
        "kept": np.ones(len(grid_kept), dtype=bool),
    })
    grid_rows["source"] = "flux_solver_a_grid_squint"

    # load existing manifest, drop prior grid rows for idempotency
    manifest = pd.read_parquet(PAIR_MANIFEST)
    if "source" not in manifest.columns:
        manifest["source"] = "ffhq"
    n_prev = int((manifest["source"] != "ffhq").sum())
    manifest = manifest[manifest["source"] == "ffhq"].reset_index(drop=True)
    if n_prev:
        print(f"[manifest] dropped {n_prev} prior non-ffhq rows for idempotency")

    merged = pd.concat([manifest, grid_rows], ignore_index=True)
    merged.to_parquet(PAIR_MANIFEST)
    by_src = merged[merged["kept"]]["source"].value_counts().to_dict()
    print(f"[manifest] wrote {PAIR_MANIFEST} ({len(merged)} rows total)")
    print(f"[manifest] kept by source: {by_src}")
    return merged


def load_thumb(sha: str) -> Image.Image:
    p = IMG_DIR / f"{sha}.png"
    if not p.exists():
        return Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (40, 0, 0))
    return Image.open(p).convert("RGB").resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)


def get_font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_collage(manifest: pd.DataFrame) -> None:
    kept = manifest[manifest["kept"]].copy()
    ffhq = kept[kept["source"] == "ffhq"].sort_values("J", ascending=False).head(N_FFHQ_TOP)
    grid = kept[kept["source"] == "flux_solver_a_grid_squint"].sort_values("abs_dtheta", ascending=False)
    rows_df = pd.concat([ffhq, grid], ignore_index=True)
    print(f"[collage] {len(ffhq)} FFHQ + {len(grid)} grid = {len(rows_df)} rows")

    n = len(rows_df)
    pair_w = THUMB_SIZE * 2 + PAD * 3
    pair_h = THUMB_SIZE + HEADER_H + PAD
    canvas = Image.new("RGB", (pair_w, pair_h * n), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    font = get_font(13)

    for ri, (_, r) in enumerate(rows_df.iterrows()):
        y0 = ri * pair_h
        pos = load_thumb(r["sha_pos"])
        neg = load_thumb(r["sha_neg"])
        canvas.paste(pos, (PAD, y0 + HEADER_H))
        canvas.paste(neg, (PAD * 2 + THUMB_SIZE, y0 + HEADER_H))
        src_short = "FFHQ" if r["source"] == "ffhq" else "GRID"
        cell = f'{r["ff_race"]}/{r["ff_gender"]}/{r["ff_age_bin"]}'
        if pd.notna(r.get("J")):
            extra = f'J={r["J"]:.4f}'
        else:
            extra = "Solver A"
        header = (f'[{src_short}] {cell}  Δθ={r["abs_dtheta"]:.3f}  '
                  f'arc_cos={r["arc_cos"]:.3f}  Δsmile={r["abs_dsmile"]:.3f}  {extra}')
        draw.text((PAD, y0 + 8), header, fill=(220, 220, 220), font=font)
        # column labels
        draw.text((PAD + 6, y0 + HEADER_H + 4), "POS (high squint)",
                  fill=(255, 200, 80), font=font)
        draw.text((PAD * 2 + THUMB_SIZE + 6, y0 + HEADER_H + 4),
                  "NEG (eyes open)", fill=(120, 220, 255), font=font)

    out = OUT_DIR / "sanity_collage.png"
    canvas.save(out, format="PNG", optimize=True)
    print(f"[collage] wrote {out} ({out.stat().st_size/1e6:.1f} MB, {canvas.size[0]}x{canvas.size[1]})")
    rows_df.to_csv(OUT_DIR / "sanity_collage.csv", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = append_grid_to_manifest()
    build_collage(manifest)


if __name__ == "__main__":
    main()
