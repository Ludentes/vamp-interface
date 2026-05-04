"""Per-image eye-region heatmaps for the squint Path B trainer.

Replaces the trainer's fixed anisotropic Gaussian (centered at latent
y=0.41, x=0.50) with a per-sha mask that hugs the actual eye landmarks.
For squint the signal is a few-pixel eyelid change, and FFHQ faces shift
~±0.10 in y and have variable yaw — a fixed Gaussian dilutes the
gradient onto cheek/brow.

Inputs : output/squint_path_b/pair_manifest.parquet (kept rows only)
         output/ffhq_images/<sha>.png  (already 512×512)
Output : output/squint_path_b/eye_masks/<sha>.npy   (float32 [H,W])
         output/squint_path_b/eye_masks/_overlays.png  (sanity grid)
         output/squint_path_b/eye_masks/_summary.json

Mask conventions
----------------
* Latent grid at MASK_RES (default 128, downsampled by trainer to actual
  latent shape via bilinear interp).
* Filled eye polygons (MediaPipe canonical eye rings, both eyes) +
  Gaussian blur → ringless soft heatmap.
* Normalised so mean=1 over the spatial grid; trainer applies its own
  `peak` multiplier on top (so peak=1.0 in yaml ≡ uniform).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "output/squint_path_b/pair_manifest.parquet"
IMG_DIR = REPO / "output/ffhq_images"
OUT_DIR = REPO / "output/squint_path_b/eye_masks"
MEDIAPIPE_MODEL = REPO / "models/mediapipe/face_landmarker.task"

MASK_RES = 128            # output H = W
BLUR_PX = 3.5             # gaussian σ in mask-px (~14 px at 512)
DILATE_PX = 2             # binary dilation before blur (lid + crow's-feet halo)
N_OVERLAYS = 16

# MediaPipe canonical 478-mesh eye rings (image-mirrored: A is subject's right).
EYE_RING_A = [33, 7, 163, 144, 145, 153, 154, 155, 133,
              173, 157, 158, 159, 160, 161, 246]
EYE_RING_B = [263, 249, 390, 373, 374, 380, 381, 382, 362,
              398, 384, 385, 386, 387, 388, 466]


def build_landmarker(model_path: Path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts), mp


def mask_for_image(rgb: np.ndarray, landmarker, mp_mod) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    img = mp_mod.Image(image_format=mp_mod.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(img)
    if not res.face_landmarks:
        return None
    lms = res.face_landmarks[0]

    # Rasterize at full image res for clean polygon edges, then downsample.
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for ring in (EYE_RING_A, EYE_RING_B):
        pts = [(lms[i].x * w, lms[i].y * h) for i in ring]
        draw.polygon(pts, fill=255)

    if DILATE_PX > 0:
        canvas = canvas.filter(ImageFilter.MaxFilter(2 * DILATE_PX + 1))

    canvas = canvas.resize((MASK_RES, MASK_RES), Image.Resampling.BILINEAR)
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=BLUR_PX))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr + 1e-3                           # tiny floor so background gets weight ε
    arr = arr / arr.mean()                     # normalise so mean=1 over grid
    return arr.astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(MANIFEST)
    kept = m[m["kept"]]
    shas = pd.unique(pd.concat([kept["sha_pos"], kept["sha_neg"]], ignore_index=True))
    print(f"[manifest] {len(kept)} kept pairs → {len(shas)} unique shas")

    landmarker, mp_mod = build_landmarker(MEDIAPIPE_MODEL)

    t0 = time.time()
    n_ok = n_skip_existing = n_no_face = n_missing_png = 0
    overlay_samples: list[tuple[str, np.ndarray, np.ndarray]] = []
    failed: list[str] = []

    for k, sha in enumerate(shas):
        out_path = OUT_DIR / f"{sha}.npy"
        if out_path.exists():
            n_skip_existing += 1
            continue
        png = IMG_DIR / f"{sha}.png"
        if not png.exists():
            n_missing_png += 1
            failed.append(sha)
            continue
        try:
            rgb = np.asarray(Image.open(png).convert("RGB"))
        except Exception:
            n_missing_png += 1
            failed.append(sha)
            continue

        mask = mask_for_image(rgb, landmarker, mp_mod)
        if mask is None:
            n_no_face += 1
            failed.append(sha)
            continue
        np.save(out_path, mask)
        n_ok += 1

        if len(overlay_samples) < N_OVERLAYS and (k % max(1, len(shas) // N_OVERLAYS)) == 0:
            overlay_samples.append((sha, rgb, mask))

        if (k + 1) % 200 == 0:
            print(f"  [{k+1}/{len(shas)}] ok={n_ok} skip={n_skip_existing} "
                  f"no_face={n_no_face} no_png={n_missing_png} "
                  f"({(time.time()-t0)/(k+1)*1000:.1f} ms/img)")

    print(f"[done] {n_ok} new, {n_skip_existing} cached, "
          f"{n_no_face} no-face, {n_missing_png} no-png, "
          f"{time.time()-t0:.1f}s")

    # Sanity overlay grid
    if overlay_samples:
        cols = 4
        rows = (len(overlay_samples) + cols - 1) // cols
        thumb = 256
        grid = Image.new("RGB", (cols * thumb, rows * thumb), (16, 16, 16))
        for i, (sha, rgb, mask) in enumerate(overlay_samples):
            base = Image.fromarray(rgb).resize((thumb, thumb), Image.Resampling.LANCZOS)
            heat = (np.clip(mask / max(mask.max(), 1e-6), 0, 1) * 255).astype(np.uint8)
            heat_img = Image.fromarray(heat).resize((thumb, thumb), Image.Resampling.BILINEAR)
            heat_rgb = Image.merge("RGB", (heat_img, Image.new("L", heat_img.size, 0),
                                            Image.new("L", heat_img.size, 0)))
            blended = Image.blend(base, heat_rgb, alpha=0.45)
            grid.paste(blended, ((i % cols) * thumb, (i // cols) * thumb))
        grid.save(OUT_DIR / "_overlays.png", optimize=True)
        print(f"[overlay] wrote {OUT_DIR/'_overlays.png'}")

    summary = {
        "n_unique_shas": int(len(shas)),
        "n_masked": n_ok + n_skip_existing,
        "n_no_face": n_no_face,
        "n_missing_png": n_missing_png,
        "mask_res": MASK_RES,
        "blur_px": BLUR_PX,
        "dilate_px": DILATE_PX,
        "rings_used": ["EYE_RING_A (16 pts)", "EYE_RING_B (16 pts)"],
        "normalisation": "mean=1 over grid; trainer multiplies by yaml peak",
        "n_failed_shas": len(failed),
    }
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))
    if failed:
        (OUT_DIR / "_failed.txt").write_text("\n".join(failed) + "\n")
    print(f"[summary] {summary}")


if __name__ == "__main__":
    main()
