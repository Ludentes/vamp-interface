"""Lash-doubling gate for chibi splat-scale variants.

Given three render-frame PNGs (pre-fix / v2-baseline / candidate) for two key
frames (f328 full-blink, f599 half-blink), crop a tight eye-region strip and
lay them out as a 2x3 collage. The artifact we are checking for is a doubled
dark horizontal band across the closed eye (two lash lines instead of one
merged seam).

Also runs an optional intensity-profile probe: for each closed eye column the
vertical luminance profile should show one prominent dark minimum (single
merged lash). Two minima with a brighter gap between = doubled lashes.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Eye-strip crop box on a 512x512 LAM render. Hand-picked so both eyes plus a
# little above and below comfortably fit; covers both wide and closed poses.
# (x0, y0, x1, y1) — adjustable via CLI if needed.
DEFAULT_CROP = (140, 290, 380, 390)


def _load(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def _crop(img: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1].copy()


def _label_strip(img: np.ndarray, text: str) -> np.ndarray:
    h, w = img.shape[:2]
    pad = 22
    out = np.full((h + pad, w, 3), 255, dtype=np.uint8)
    out[pad:, :, :] = img
    pil = Image.fromarray(out)
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()
    d.text((6, 3), text, fill=(0, 0, 0), font=font)
    return np.array(pil)


def lash_profile_score(strip: np.ndarray,
                       eye_cols: tuple[int, int]) -> dict:
    """Per-eye intensity profile probe.

    For each given column (left-eye center, right-eye center) in the strip,
    extract the vertical luminance profile, find dark minima with non-trivial
    prominence, and return a score: number of minima + the brightness gap
    between them. 1 minimum = clean lash; 2 = doubled.
    """
    luma = (strip[..., 0] * 0.299
            + strip[..., 1] * 0.587
            + strip[..., 2] * 0.114).astype(np.float32)
    out = {}
    for tag, col in zip(("L", "R"), eye_cols):
        col = max(2, min(col, luma.shape[1] - 3))
        # Average a 5-pixel-wide column to denoise.
        profile = luma[:, col - 2: col + 3].mean(axis=1)
        # Find local minima with simple 3-point rule.
        mins = []
        for i in range(1, len(profile) - 1):
            if profile[i] < profile[i - 1] and profile[i] < profile[i + 1]:
                mins.append((i, float(profile[i])))
        # Keep minima darker than (mean - 0.3*std) — prominent only.
        thr = profile.mean() - 0.3 * profile.std()
        deep = [m for m in mins if m[1] < thr]
        out[tag] = {
            "col": col,
            "n_deep_minima": len(deep),
            "deep_minima": deep,
            "profile_min": float(profile.min()),
            "profile_mean": float(profile.mean()),
        }
        # Doubling score: if 2+ deep minima, brightness between them.
        if len(deep) >= 2:
            y0 = deep[0][0]
            y1 = deep[-1][0]
            mid_max = float(profile[y0:y1].max())
            depth = float(min(deep[0][1], deep[-1][1]))
            out[tag]["between_brightness"] = mid_max
            out[tag]["depth"] = depth
            out[tag]["doubling_score"] = (mid_max - depth) / max(
                profile.max() - profile.min(), 1.0
            )
        else:
            out[tag]["doubling_score"] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_f328", required=True)
    ap.add_argument("--pre_f599", required=True)
    ap.add_argument("--v2_f328", required=True)
    ap.add_argument("--v2_f599", required=True)
    ap.add_argument("--cand_f328", required=True)
    ap.add_argument("--cand_f599", required=True)
    ap.add_argument("--cand_tag", default="candidate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", nargs=4, type=int, default=list(DEFAULT_CROP),
                    metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--eye_cols_in_crop", nargs=2, type=int, default=(70, 170),
                    metavar=("LX", "RX"),
                    help="Center x of each eye within the cropped strip "
                         "(measured after crop, not in original image).")
    args = ap.parse_args()

    crop = tuple(args.crop)
    paths = {
        ("pre",  "f328"): args.pre_f328,
        ("pre",  "f599"): args.pre_f599,
        ("v2",   "f328"): args.v2_f328,
        ("v2",   "f599"): args.v2_f599,
        ("cand", "f328"): args.cand_f328,
        ("cand", "f599"): args.cand_f599,
    }
    strips: dict[tuple[str, str], np.ndarray] = {}
    for key, p in paths.items():
        if not Path(p).is_file():
            print(f"missing {key}: {p}", file=sys.stderr)
            return 2
        strips[key] = _crop(_load(Path(p)), crop)

    # Probe each strip.
    print(f"\n--- lash gate ({args.cand_tag}) ---")
    print(f"crop box: {crop}   eye cols in crop: {args.eye_cols_in_crop}")
    for (variant, frame), strip in strips.items():
        s = lash_profile_score(strip, tuple(args.eye_cols_in_crop))
        l, r = s["L"], s["R"]
        print(f"{variant:>4s} {frame}: "
              f"L:n={l['n_deep_minima']} dbl={l['doubling_score']:.3f}  "
              f"R:n={r['n_deep_minima']} dbl={r['doubling_score']:.3f}")

    # Build 2x3 collage. Rows = f328, f599. Cols = pre, v2, candidate.
    col_labels = ("pre-fix", "v2 baseline", args.cand_tag)
    row_keys = ("f328", "f599")
    col_keys = ("pre", "v2", "cand")
    h, w = strips[("pre", "f328")].shape[:2]
    labeled = {
        (v, f): _label_strip(strips[(v, f)], f"{lab} · {f}")
        for v, lab in zip(col_keys, col_labels)
        for f in row_keys
    }
    lh = labeled[("pre", "f328")].shape[0]
    gap = 6
    canvas = np.full((lh * 2 + gap, w * 3 + gap * 2, 3), 255, dtype=np.uint8)
    for ri, f in enumerate(row_keys):
        for ci, v in enumerate(col_keys):
            x = ci * (w + gap)
            y = ri * (lh + gap)
            canvas[y:y + lh, x:x + w] = labeled[(v, f)]
    Image.fromarray(canvas).save(args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
