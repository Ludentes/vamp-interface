"""Render contact sheets for Phase 2 visual inspection.

Two views:

  --view grid       — rows=photo_id, cols=cfg_idx (one seed picked); shows refined.png.
                      Adds id_cos badge per cell; nan cells get a red X overlay.
  --view stages     — one or many cells side-by-side: ctrl | render | swap | refined.

Defaults to the iter-0 seed.

Usage:
  uv run --no-project python -m scripts.photobooth_sweep.contact_sheet --view grid
  uv run --no-project python -m scripts.photobooth_sweep.contact_sheet --view stages \\
      --cells id_09__cfg002 id_12__cfg001_s1 id_16__cfg000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent
PHASE2 = ROOT / "exp_output/photobooth_phase2"
CELLS = PHASE2 / "cells"
SCORES = PHASE2 / "scores.parquet"


def set_root(root: Path) -> None:
    global PHASE2, CELLS, SCORES
    PHASE2 = root
    CELLS = root / "cells"
    SCORES = root / "scores.parquet"

TILE_PX = 192  # per-image tile size (square)
PAD = 4
LABEL_H = 22


def load_scores():
    return pq.read_table(SCORES).to_pandas()


def fit_square(img: np.ndarray | None, px: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.full((px, px, 3), 80, np.uint8)
    h, w = img.shape[:2]
    s = min(px / w, px / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((px, px, 3), 30, np.uint8)
    y0, x0 = (px - nh) // 2, (px - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = rs
    return canvas


def label_strip(text: str, w: int, h: int = LABEL_H,
                color=(230, 230, 230)) -> np.ndarray:
    strip = np.full((h, w, 3), 20, np.uint8)
    cv2.putText(strip, text, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)
    return strip


def cell_dir(photo_id: str, cfg_idx: int, seed_iter: int) -> Path:
    suffix = "" if seed_iter == 0 else f"_s{seed_iter}"
    return CELLS / f"{photo_id}__cfg{cfg_idx:03d}{suffix}"


def make_grid(seed_iter: int, out_path: Path,
              stages: tuple[str, ...] = ("refined",)) -> None:
    """Per cfg col, show one or more stages side-by-side (render, swap, refined)."""
    t = load_scores()
    t = t[t.seed_iter == seed_iter]
    photos = sorted(t.photo_id.unique())
    cfgs = sorted(t.cfg_idx.unique())

    n_stages = len(stages)
    cols = len(cfgs)
    rows = len(photos)
    cell_w = TILE_PX
    cell_h = TILE_PX + LABEL_H
    # composite-cell width: N stages stacked horizontally with PAD between
    comp_w = n_stages * cell_w + (n_stages - 1) * PAD if n_stages > 1 else cell_w

    head_h = 24
    side_w = 56
    grid_w = side_w + cols * (comp_w + PAD) + PAD
    grid_h = head_h + rows * (cell_h + PAD) + PAD
    canvas = np.full((grid_h, grid_w, 3), 10, np.uint8)

    # column headers: cfg label spans the composite width
    for j, c in enumerate(cfgs):
        x0 = side_w + j * (comp_w + PAD) + PAD
        title = f"cfg{c:03d}  [{' | '.join(stages)}]" if n_stages > 1 else f"cfg{c:03d}"
        canvas[0:head_h, x0:x0 + comp_w] = label_strip(title, comp_w, head_h)

    for i, pid in enumerate(photos):
        y0 = head_h + i * (cell_h + PAD) + PAD
        rs = label_strip(pid, side_w, cell_h)
        canvas[y0:y0 + cell_h, 0:side_w] = rs

        for j, c in enumerate(cfgs):
            row = t[(t.photo_id == pid) & (t.cfg_idx == c)]
            if row.empty:
                continue
            r = row.iloc[0]
            x0_comp = side_w + j * (comp_w + PAD) + PAD
            cdir = cell_dir(pid, c, seed_iter)

            for k, stage in enumerate(stages):
                xk = x0_comp + k * (cell_w + PAD)
                img_path = cdir / f"{stage}.png"
                if img_path.exists():
                    tile = fit_square(cv2.imread(str(img_path)), cell_w)
                else:
                    tile = np.full((cell_w, cell_w, 3), 60, np.uint8)
                # nan overlay only on the swap/refined stages (render is pre-swap)
                if stage in ("swap", "refined") and not np.isfinite(r.id_cos):
                    cv2.line(tile, (0, 0), (cell_w, cell_w), (0, 0, 220), 2)
                    cv2.line(tile, (cell_w, 0), (0, cell_w), (0, 0, 220), 2)
                canvas[y0:y0 + cell_w, xk:xk + cell_w] = tile

            # one badge per composite cell (under all stages)
            txt = "nan" if not np.isfinite(r.id_cos) else f"{r.id_cos:.2f}"
            badge = label_strip(f"{txt} {r.det_mode[:3]}", comp_w, LABEL_H,
                                color=(150, 255, 150) if np.isfinite(r.id_cos)
                                else (150, 150, 255))
            canvas[y0 + cell_w:y0 + cell_w + LABEL_H,
                   x0_comp:x0_comp + comp_w] = badge

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"wrote {out_path}  ({canvas.shape[1]}×{canvas.shape[0]} px)")


def make_stages(cell_ids: list[str], out_path: Path) -> None:
    t = load_scores().set_index("cell_id")
    stages = ["ctrl", "render", "swap", "refined"]
    n = len(cell_ids)
    cols = len(stages)
    head_h = 24
    side_w = 110
    tile = 256
    cell_h = tile + LABEL_H
    grid_w = side_w + cols * (tile + PAD) + PAD
    grid_h = head_h + n * (cell_h + PAD) + PAD
    canvas = np.full((grid_h, grid_w, 3), 10, np.uint8)
    for j, s in enumerate(stages):
        x0 = side_w + j * (tile + PAD) + PAD
        canvas[0:head_h, x0:x0 + tile] = label_strip(s, tile, head_h)

    for i, cid in enumerate(cell_ids):
        y0 = head_h + i * (cell_h + PAD) + PAD
        if cid in t.index:
            r = t.loc[cid]
            id_cos = r.id_cos
            cfg_idx = int(r.cfg_idx)
            label = f"{cid}\nid={id_cos:.2f} cfg{cfg_idx:03d}"
        else:
            label = cid
        # left label (two lines via two rectangles)
        rs = label_strip(label.split("\n")[0], side_w, LABEL_H)
        canvas[y0:y0 + LABEL_H, 0:side_w] = rs
        rs2 = label_strip(label.split("\n")[1] if "\n" in label else "",
                          side_w, LABEL_H)
        canvas[y0 + LABEL_H:y0 + 2 * LABEL_H, 0:side_w] = rs2

        cdir = CELLS / cid
        for j, s in enumerate(stages):
            x0 = side_w + j * (tile + PAD) + PAD
            img_path = cdir / f"{s}.png"
            img = cv2.imread(str(img_path)) if img_path.exists() else None
            t_img = fit_square(img, tile)
            canvas[y0:y0 + tile, x0:x0 + tile] = t_img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"wrote {out_path}  ({canvas.shape[1]}×{canvas.shape[0]} px)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--view", choices=("grid", "stages"), default="grid")
    p.add_argument("--seed-iter", type=int, default=0,
                   help="grid view: which seed iter to show (0/1/2)")
    p.add_argument("--cells", nargs="*", default=None,
                   help="stages view: cell_ids to render")
    p.add_argument("--out", default=None)
    p.add_argument("--root", default=None,
                   help="experiment root (default: photobooth_phase2)")
    p.add_argument("--stages", default="refined",
                   help="comma-list of stages per cfg col, e.g. 'render,refined'")
    args = p.parse_args()

    if args.root:
        set_root(ROOT / args.root)

    if args.view == "grid":
        stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
        suffix = "" if stages == ("refined",) else "_" + "+".join(stages)
        default_out = (PHASE2.relative_to(ROOT)
                       / f"grid_s{args.seed_iter}{suffix}.png")
        out = ROOT / (args.out or str(default_out))
        make_grid(args.seed_iter, out, stages=stages)
    else:
        cells = args.cells or [
            "id_09__cfg002",       # ceiling cell (0.904)
            "id_17__cfg001",       # stable mid
            "id_16__cfg000",       # catastrophic (-0.014, face_frac=0.003)
            "id_12__cfg001",       # collapse mode (0.20)
            "id_11__cfg004_s1",    # tight nan (forced det)
            "id_19__cfg002",       # depth-collapses-on-id_19 (0.28)
        ]
        out = ROOT / (args.out or "exp_output/photobooth_phase2/stages.png")
        make_stages(cells, out)


if __name__ == "__main__":
    main()
