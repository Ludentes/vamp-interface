"""Judging-sheet: "collage of collages" for outside-judge column comparison.

Each column is a configuration with a multi-line header (title + subtitle +
key params + score summary). Rows are the 20 source identities. The judge
picks a column overall, not a per-image winner.

Generic over (root, cfg_idx, seed_iter, title, subtitle) so Phase 4 (and
beyond) columns can be added by extending COLUMNS_P3 (or registering a new
set).

Usage:
  uv run --no-project python -m scripts.photobooth_sweep.judging_sheet \\
      --set p3 --out exp_output/photobooth_judging/sheet_p3.png
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent

# ---- Layout constants ---------------------------------------------------

MINI = 200              # mini-tile size per stage (square)
INNER_PAD = 4           # padding between render and refined inside a cell
STAGE_LABEL_H = 18      # height of "render | refined" label strip
COL_W = 2 * MINI + INNER_PAD
TILE = MINI             # source-photo tile in the side strip
SIDE_W = 220            # left strip: source photo + identity label
COL_PAD = 10            # padding between cfg columns
ROW_PAD = 8             # padding between rows
SCORE_H = 28            # height of per-cell score badge
HEADER_H = 220          # height of per-column header band
BANNER_H = 100          # top banner with judging instruction
FOOTER_H = 96           # bottom per-column summary band
PHOTO_IDS = [f"id_{i:02d}" for i in range(20)]


# ---- Column spec --------------------------------------------------------

@dataclass(frozen=True)
class Column:
    root: str           # phase root, relative to repo root
    cfg_idx: int
    seed_iter: int = 0
    title: str = ""     # one-line bold header
    subtitle: str = ""  # 2-3 line explanation (\n-separated)
    params: tuple[tuple[str, str], ...] = field(default_factory=tuple)


COLUMNS_P3 = (
    Column(root="exp_output/photobooth_phase3", cfg_idx=0,
           title="Baseline",
           subtitle="HyperSwap's historical default.\n"
                   "Pure source identity, no embedding\nmix toward the doll.",
           params=(("swap_weight", "0.50"),
                   ("alpha (embed mix)", "0.00"),
                   ("identity pull", "max"))),
    Column(root="exp_output/photobooth_phase3", cfg_idx=1,
           title="Light mix",
           subtitle="Pull source embedding slightly\n"
                   "toward the doll's painted face.\n"
                   "Tiny step away from photoreal.",
           params=(("swap_weight", "0.40"),
                   ("alpha (embed mix)", "+0.07"),
                   ("identity pull", "−7%"))),
    Column(root="exp_output/photobooth_phase3", cfg_idx=2,
           title="Moderate mix",
           subtitle="Source embedding pulled 14%\n"
                   "toward the doll.\n"
                   "Still recognisable, less photoreal.",
           params=(("swap_weight", "0.30"),
                   ("alpha (embed mix)", "+0.14"),
                   ("identity pull", "−14%"))),
    Column(root="exp_output/photobooth_phase3", cfg_idx=3,
           title="Strong mix",
           subtitle="Embedding pulled 21% toward doll.\n"
                   "Identity audibly weaker; doll\n"
                   "character may start emerging.",
           params=(("swap_weight", "0.20"),
                   ("alpha (embed mix)", "+0.21"),
                   ("identity pull", "−21%"))),
    Column(root="exp_output/photobooth_phase3", cfg_idx=4,
           title="Heavy mix",
           subtitle="Embedding pulled 28% toward doll.\n"
                   "Identity faint; painted-face\n"
                   "geometry survives more.",
           params=(("swap_weight", "0.10"),
                   ("alpha (embed mix)", "+0.28"),
                   ("identity pull", "−28%"))),
    Column(root="exp_output/photobooth_phase3", cfg_idx=5,
           title="Max mix",
           subtitle="Embedding pulled 35% toward doll —\n"
                   "the extreme end of the dial.\n"
                   "Identity may be lost entirely.",
           params=(("swap_weight", "0.00"),
                   ("alpha (embed mix)", "+0.35"),
                   ("identity pull", "−35%"))),
)


# All-phases survey — render-axis (Phase 2) + identity-axis (Phase 3) +
# off-grid corners (Phase 1). Subsampled to the 4 identities Phase 1 ran on
# (id_00, id_01, id_11, id_16) so every column is fully populated.
COLUMNS_P4 = (
    # ---- Phase 2: render axis (cn_condition × budget × canny preset) ----
    Column(root="exp_output/photobooth_phase2", cfg_idx=0,
           title="P2 base — nat/canny/soft",
           subtitle="Phase-2 reference. Natural 1024,\n"
                   "canny edges (soft thresholds),\n"
                   "demo prompt on.",
           params=(("budget", "natural_1024"),
                   ("CN", "canny / soft"),
                   ("demo_inject", "on"))),
    Column(root="exp_output/photobooth_phase2", cfg_idx=1,
           title="P2 — aggressive canny",
           subtitle="Same as base, but stronger edge\n"
                   "detection. Tighter outline lock,\n"
                   "less room for style drift.",
           params=(("budget", "natural_1024"),
                   ("CN", "canny / aggressive"),
                   ("demo_inject", "on"))),
    Column(root="exp_output/photobooth_phase2", cfg_idx=2,
           title="P2 — depth CN",
           subtitle="Depth conditioning instead of canny.\n"
                   "Softer geometric lock — only volume,\n"
                   "no edges. Tends to drift more.",
           params=(("budget", "natural_1024"),
                   ("CN", "depth"),
                   ("demo_inject", "on"))),
    Column(root="exp_output/photobooth_phase2", cfg_idx=3,
           title="P2 — tight crop / canny",
           subtitle="Tighter face crop (face fills the\n"
                   "frame more). Less shoulders/background,\n"
                   "more pixels on face features.",
           params=(("budget", "tight_1024"),
                   ("CN", "canny / soft"),
                   ("demo_inject", "on"))),
    Column(root="exp_output/photobooth_phase2", cfg_idx=4,
           title="P2 — tight + aggr canny",
           subtitle="Tight crop AND aggressive canny.\n"
                   "Most constrained config in the\n"
                   "render axis.",
           params=(("budget", "tight_1024"),
                   ("CN", "canny / aggressive"),
                   ("demo_inject", "on"))),
    Column(root="exp_output/photobooth_phase2", cfg_idx=5,
           title="P2 — tight + depth",
           subtitle="Tight crop with depth CN.\n"
                   "Tighter framing, softest geometric\n"
                   "lock — stylization gets more rope.",
           params=(("budget", "tight_1024"),
                   ("CN", "depth"),
                   ("demo_inject", "on"))),
    # ---- Phase 1 corners: regimes Phase 2/3 locked out ----
    Column(root="exp_output/photobooth_phase1", cfg_idx=0,
           title="P1 — heavy refine pass",
           subtitle="Phase-2-like, but with a refine\n"
                   "pass (denoise=0.15) after the swap.\n"
                   "Tests whether refine helps or hurts.",
           params=(("CN", "canny / soft @0.95"),
                   ("refine_denoise", "0.15"),
                   ("demo_inject", "on"))),
    Column(root="exp_output/photobooth_phase1", cfg_idx=21,
           title="P1 — doll prompt only",
           subtitle="Demographic prompt turned OFF.\n"
                   "Render gets the matryoshka prompt\n"
                   "alone — no race/age/gender pull.",
           params=(("CN", "depth @0.92"),
                   ("budget", "natural_1024"),
                   ("demo_inject", "OFF"))),
    Column(root="exp_output/photobooth_phase1", cfg_idx=8,
           title="P1 — doll prompt, 768",
           subtitle="Strongest stylization regime found:\n"
                   "depth CN, no demo prompt, 768 render.\n"
                   "High clip_style, weak identity.",
           params=(("CN", "depth @0.94"),
                   ("budget", "natural_768"),
                   ("demo_inject", "OFF"))),
    # ---- Phase 3: identity axis (current COLUMNS_P3 verbatim) ----
    *COLUMNS_P3,
)


COLUMN_SETS = {
    "p3": (COLUMNS_P3, PHOTO_IDS),
    "p4": (COLUMNS_P4, ["id_00", "id_01", "id_11", "id_16"]),
}


# ---- Drawing helpers ----------------------------------------------------

def putlines(canvas: np.ndarray, lines: list[str], origin: tuple[int, int],
             scale: float = 0.45, color=(230, 230, 230), thickness: int = 1,
             line_h: int = 18) -> None:
    x, y = origin
    for i, ln in enumerate(lines):
        cv2.putText(canvas, ln, (x, y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                    cv2.LINE_AA)


def fit_square(img: np.ndarray | None, px: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.full((px, px, 3), 60, np.uint8)
    h, w = img.shape[:2]
    s = min(px / w, px / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((px, px, 3), 30, np.uint8)
    y0, x0 = (px - nh) // 2, (px - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = rs
    return canvas


def load_scores(col: Column):
    p = ROOT / col.root / "scores.parquet"
    t = pq.read_table(p).to_pandas()
    mask = t.cfg_idx == col.cfg_idx
    if "seed_iter" in t.columns:
        mask &= t.seed_iter == col.seed_iter
    return t[mask]


def cell_dir(col: Column, photo_id: str) -> Path:
    sfx = "" if col.seed_iter == 0 else f"_s{col.seed_iter}"
    return ROOT / col.root / "cells" / f"{photo_id}__cfg{col.cfg_idx:03d}{sfx}"


# ---- Main composition ---------------------------------------------------

def make_sheet(columns: tuple[Column, ...], out_path: Path,
               title: str, instruction: str,
               photo_ids: list[str] | None = None) -> None:
    if photo_ids is None:
        photo_ids = PHOTO_IDS
    n_cols = len(columns)
    col_w = COL_W
    row_h = MINI + SCORE_H

    total_w = SIDE_W + n_cols * (col_w + COL_PAD) + COL_PAD
    total_h = (BANNER_H + HEADER_H + STAGE_LABEL_H
               + len(photo_ids) * (row_h + ROW_PAD) + ROW_PAD + FOOTER_H)
    canvas = np.full((total_h, total_w, 3), 14, np.uint8)

    # ---- Top banner ----
    cv2.rectangle(canvas, (0, 0), (total_w, BANNER_H), (28, 28, 28), -1)
    putlines(canvas, [title], (16, 30), scale=0.85, thickness=2,
             color=(245, 245, 245), line_h=24)
    putlines(canvas, instruction.splitlines(), (16, 56), scale=0.5,
             color=(190, 190, 190), line_h=18)

    # ---- Per-column headers ----
    header_y = BANNER_H
    for j, col in enumerate(columns):
        x0 = SIDE_W + j * (col_w + COL_PAD) + COL_PAD
        cv2.rectangle(canvas, (x0, header_y), (x0 + col_w, header_y + HEADER_H),
                      (34, 34, 34), -1)
        # title
        putlines(canvas, [col.title], (x0 + 10, header_y + 26),
                 scale=0.7, thickness=2, color=(240, 240, 240), line_h=22)
        # subtitle
        putlines(canvas, col.subtitle.splitlines(), (x0 + 10, header_y + 60),
                 scale=0.46, color=(200, 200, 200), line_h=17)
        # params block
        py = header_y + 60 + len(col.subtitle.splitlines()) * 17 + 14
        cv2.line(canvas, (x0 + 10, py - 8), (x0 + col_w - 10, py - 8),
                 (70, 70, 70), 1)
        for k, v in col.params:
            putlines(canvas, [f"{k}: {v}"], (x0 + 10, py), scale=0.42,
                     color=(180, 220, 180), line_h=16)
            py += 16

    # ---- Stage labels strip (under headers, above first row) ----
    label_y = BANNER_H + HEADER_H
    cv2.rectangle(canvas, (0, label_y), (total_w, label_y + STAGE_LABEL_H),
                  (20, 20, 20), -1)
    for j in range(n_cols):
        x0 = SIDE_W + j * (col_w + COL_PAD) + COL_PAD
        putlines(canvas, ["render (pre-swap)"], (x0 + 8, label_y + 14),
                 scale=0.42, color=(170, 170, 200), line_h=14)
        putlines(canvas, ["refined (post-swap)"],
                 (x0 + MINI + INNER_PAD + 8, label_y + 14),
                 scale=0.42, color=(170, 200, 170), line_h=14)

    # ---- Per-photo rows ----
    # Pre-load score tables per column for badges + footer stats.
    # Key by column index (not cfg_idx) since the same cfg_idx can appear
    # in different roots (e.g. phase2 cfg0 and phase1 cfg0).
    score_tables = {j: load_scores(col) for j, col in enumerate(columns)}

    for i, pid in enumerate(photo_ids):
        y0 = (BANNER_H + HEADER_H + STAGE_LABEL_H
              + i * (row_h + ROW_PAD) + ROW_PAD)
        # source photo + label
        src_p = ROOT / f"data/importer/identities/{pid}.png"
        src = cv2.imread(str(src_p)) if src_p.exists() else None
        tile_side = min(TILE, SIDE_W - 20)
        src_tile = fit_square(src, tile_side)
        sx0 = 10
        canvas[y0:y0 + tile_side, sx0:sx0 + tile_side] = src_tile
        putlines(canvas, [pid], (sx0, y0 + tile_side + 18), scale=0.5,
                 color=(220, 220, 220), line_h=16)

        for j, col in enumerate(columns):
            x0 = SIDE_W + j * (col_w + COL_PAD) + COL_PAD
            cdir = cell_dir(col, pid)

            render_img = cv2.imread(str(cdir / "render.png"))
            refined_img = cv2.imread(str(cdir / "refined.png"))
            render_tile = fit_square(render_img, MINI)
            refined_tile = fit_square(refined_img, MINI)

            canvas[y0:y0 + MINI, x0:x0 + MINI] = render_tile
            x1 = x0 + MINI + INNER_PAD
            canvas[y0:y0 + MINI, x1:x1 + MINI] = refined_tile

            # subtle border between the two mini-tiles to anchor the eye
            cv2.line(canvas, (x0 + MINI + INNER_PAD // 2, y0),
                     (x0 + MINI + INNER_PAD // 2, y0 + MINI),
                     (60, 60, 60), 1)

            row = score_tables[j]
            r = row[row.photo_id == pid]
            if not r.empty:
                rr = r.iloc[0]
                id_cos = rr.id_cos
                cs = rr.clip_style
                if np.isnan(id_cos):
                    txt = "id=nan  (no face detected)"
                    color = (140, 140, 240)
                else:
                    txt = f"id_cos={id_cos:.2f}   clip_style={cs:.2f}"
                    color = (180, 220, 180)
            else:
                txt = "—"
                color = (140, 140, 140)
            cv2.rectangle(canvas, (x0, y0 + MINI),
                          (x0 + col_w, y0 + MINI + SCORE_H),
                          (24, 24, 24), -1)
            putlines(canvas, [txt], (x0 + 8, y0 + MINI + 19), scale=0.45,
                     color=color, line_h=14)

    # ---- Footer: per-column summary ----
    fy = total_h - FOOTER_H + 8
    cv2.rectangle(canvas, (0, total_h - FOOTER_H), (total_w, total_h),
                  (24, 24, 24), -1)
    putlines(canvas, [f"Per-column means (over the {len(photo_ids)} identities above)"],
             (16, fy + 12), scale=0.5, color=(220, 220, 220), line_h=16)
    for j, col in enumerate(columns):
        x0 = SIDE_W + j * (col_w + COL_PAD) + COL_PAD
        tbl = score_tables[j]
        n = len(tbl)
        nan_n = int(tbl.id_cos.isna().sum())
        m_id = float(tbl.id_cos.dropna().mean()) if n - nan_n else float("nan")
        m_cs = float(tbl.clip_style.dropna().mean()) if n - nan_n else float("nan")
        lines = [
            f"id_cos mean: {m_id:.3f}",
            f"clip_style mean: {m_cs:.3f}",
            f"nan rate: {nan_n}/{n}",
        ]
        putlines(canvas, lines, (x0 + 8, fy + 32), scale=0.42,
                 color=(200, 220, 200), line_h=15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"wrote {out_path}  ({canvas.shape[1]}×{canvas.shape[0]} px)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="p3", choices=tuple(COLUMN_SETS.keys()),
                   help="which column set to render")
    p.add_argument("--out", default=None,
                   help="output PNG path (default: exp_output/photobooth_judging/sheet_<set>.png)")
    p.add_argument("--title",
                   default="Photobooth — which column do you prefer overall?")
    p.add_argument("--instruction",
                   default=("Each column is one configuration of the matryoshka pipeline.\n"
                           "Read the column header, then scan the 20 portraits down the column.\n"
                           "Pick the column whose results you'd want as the product — not a single image."))
    args = p.parse_args()

    cols, photo_ids = COLUMN_SETS[args.set]
    out = ROOT / (args.out or f"exp_output/photobooth_judging/sheet_{args.set}.png")
    make_sheet(cols, out, title=args.title, instruction=args.instruction,
               photo_ids=photo_ids)


if __name__ == "__main__":
    main()
