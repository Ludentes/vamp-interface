"""Smoke collage — for each of the 11 NMF atoms, show the 5 corpus images
with the highest atom-k activation.

Not a Phase 4 test (we don't apply fitted directions at inference).
A vocabulary sanity check: does each atom pick out visually coherent
facial changes? Useful as a figure for the blog post.

Output: models/blendshape_nmf/atom_extremes_collage.png
        models/blendshape_nmf/atom_extremes_list.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
NMF_DIR = ROOT / "models/blendshape_nmf"
OUT_PNG = NMF_DIR / "atom_extremes_collage.png"
OUT_JSON = NMF_DIR / "atom_extremes_list.json"

CORPUS_SOURCES = [
    (METRICS / "bootstrap_v1/blendshapes.json", METRICS / "bootstrap_v1"),
    (METRICS / "crossdemo/smile/alpha_interp/blendshapes.json",
     METRICS / "crossdemo/smile/alpha_interp"),
    (METRICS / "crossdemo/smile/smile_inphase/blendshapes.json",
     METRICS / "crossdemo/smile/smile_inphase"),
    (METRICS / "crossdemo/smile/jaw_inphase/blendshapes.json",
     METRICS / "crossdemo/smile/jaw_inphase"),
    (METRICS / "crossdemo/smile/intensity_full/blendshapes.json",
     METRICS / "crossdemo/smile/intensity_full"),
]

# 11 atom labels from nmf-decomposition-result doc
ATOM_LABELS = [
    "AU12+AU10  broad smile",
    "AU1+AU2    brow raise",
    "AU7        lid tighten",
    "AU64+AU45  gaze-down+blink",
    "AU4        brow lower",
    "AU12       pure smile",
    "AU16+AU26  lower lip + jaw",
    "AU26       pure jaw drop",
    "AU24+AU28  lip press (fragile)",
    "AU61/62    horizontal gaze",
    "AU18       pucker",
]

N_COLS = 5
THUMB = 200  # px per cell


def load_corpus() -> tuple[np.ndarray, list[str], list[Path]]:
    W_basis = np.load(NMF_DIR / "W_nmf_k11.npy")
    meta = json.loads((NMF_DIR / "manifest.json").read_text())
    channels = meta["channels"]
    rows = []
    paths: list[Path] = []
    for bs_json, img_dir in CORPUS_SOURCES:
        if not bs_json.exists():
            continue
        scores = json.loads(bs_json.read_text())
        for rel, s in scores.items():
            img_path = img_dir / rel
            if not img_path.exists():
                continue
            rows.append([s.get(c, 0.0) for c in channels])
            paths.append(img_path)
    X = np.array(rows, dtype=np.float64)
    print(f"  [load] {len(rows)} samples, {len(channels)} channels")
    return X, channels, paths


def main() -> None:
    print("[collage] loading corpus + basis")
    X, channels, paths = load_corpus()
    W_basis = np.load(NMF_DIR / "W_nmf_k11.npy")
    atoms = np.clip(X @ np.linalg.pinv(W_basis), 0.0, None)  # (N, k)
    n_atoms = atoms.shape[1]
    print(f"  [load] atoms shape = {atoms.shape}")

    # For each atom, pick top N_COLS images by atom-k activation
    picks_per_atom: list[list[dict]] = []
    for k in range(n_atoms):
        col = atoms[:, k]
        top_idx = np.argsort(-col)[:N_COLS]
        picks_per_atom.append([
            {"path": str(paths[int(i)].relative_to(ROOT)),
             "coeff": float(col[int(i)])}
            for i in top_idx
        ])

    # Assemble collage
    W = N_COLS * THUMB
    H = n_atoms * THUMB + 30 * n_atoms  # label strip per row
    canvas = Image.new("RGB", (W + 240, H), (255, 255, 255))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    y_off = 0
    for k in range(n_atoms):
        # left-side label column
        draw.text((8, y_off + THUMB // 2 - 10), ATOM_LABELS[k], fill=(0, 0, 0), font=font)
        for col, pick in enumerate(picks_per_atom[k]):
            img = Image.open(ROOT / pick["path"]).convert("RGB")
            img.thumbnail((THUMB, THUMB))
            # Centre it on a THUMB x THUMB cell
            cell = Image.new("RGB", (THUMB, THUMB), (240, 240, 240))
            ox = (THUMB - img.width) // 2
            oy = (THUMB - img.height) // 2
            cell.paste(img, (ox, oy))
            canvas.paste(cell, (240 + col * THUMB, y_off))
            # coefficient annotation
            draw.text((240 + col * THUMB + 4, y_off + THUMB - 14),
                      f"{pick['coeff']:.2f}", fill=(200, 40, 40), font=small)
        y_off += THUMB + 10

    canvas.save(OUT_PNG)
    print(f"  [save] {OUT_PNG}  ({canvas.width}×{canvas.height})")

    OUT_JSON.write_text(json.dumps({
        "atom_labels": ATOM_LABELS,
        "n_atoms": n_atoms,
        "n_cols": N_COLS,
        "picks": picks_per_atom,
    }, indent=2))
    print(f"  [save] {OUT_JSON}")


if __name__ == "__main__":
    main()
