"""Objective per-frame artifact scoring for a PersonaLive demo video.

For each frame extract:
  - arcface_cos: ArcFace cosine vs anchor (identity drift; lower = more drift)
  - lap_var: Laplacian variance (sharpness; lower = mush / motion blur)
  - flicker: mean(|frame[t] - frame[t-1]|) (temporal instability)
  - face_det: did mediapipe FaceMesh find a face? (0/1)
  - bg_drift: temporal std on a fixed background mask (anchor-static; should be low)
  - ghost_resid: L1 residual to anchor on a forehead mask, high-pass
                 (catches ghost eyebrows / forehead streaks)

Also emit a contact sheet of the worst-N frames by ghost_resid + lap_var.

The video can be either solo (HxW) or side-by-side (Hx2W where left is the
driver and right is the PersonaLive output). For side-by-side, only the right
half is scored (use --right-half).

Usage:
    uv run python scripts/artifact_score.py \\
        --video /tmp/realtime_demo_fixed.mp4 \\
        --anchor /home/newub/w/vamp-interface/data/llf-phase2/asian_m__06_neutral.midframe.png \\
        --out-csv /tmp/artifact_score.csv \\
        --out-thumbs /tmp/artifact_worst.png \\
        --right-half
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.demographic_pc.classifiers import InsightFaceClassifier  # noqa: E402
from src.demographic_pc.score_blendshapes import make_landmarker  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video",  required=True)
    p.add_argument("--anchor", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-thumbs", default=None,
                   help="Path to write a worst-frames contact sheet (PNG)")
    p.add_argument("--right-half", action="store_true",
                   help="Video is side-by-side (driver|output); score only output")
    p.add_argument("--stride", type=int, default=2,
                   help="Sample every Nth frame (default 2 = ~30 Hz from 60 fps)")
    p.add_argument("--worst-n", type=int, default=8)
    return p.parse_args()


def laplacian_variance(rgb_uint8: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def hp_residual_to_anchor(out_rgb: np.ndarray, anchor_rgb: np.ndarray,
                          forehead_mask: np.ndarray) -> float:
    """High-pass L1 on a forehead mask. Picks up ghost eyebrows & forehead
    streaks that should not be there if the anchor's forehead is clean."""
    diff = cv2.absdiff(out_rgb, anchor_rgb).astype(np.float32).mean(axis=-1)
    blur = cv2.GaussianBlur(diff, (15, 15), 0)
    hp = np.clip(diff - blur, 0, None)
    return float(hp[forehead_mask > 0].mean()) if forehead_mask.any() else float("nan")


def make_forehead_mask(anchor_rgb: np.ndarray, lm) -> np.ndarray:
    """Top half of the FaceMesh-bounded face on the anchor."""
    import mediapipe as mp
    h, w = anchor_rgb.shape[:2]
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=anchor_rgb))
    mask = np.zeros((h, w), dtype=np.uint8)
    if not res.face_landmarks:
        return mask
    pts = np.array([[lm_.x * w, lm_.y * h] for lm_ in res.face_landmarks[0]])
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    # Forehead = top 35% of face bbox (above eyebrows).
    y_top = max(0, y1 - int(0.05 * (y2 - y1)))
    y_split = y1 + int(0.35 * (y2 - y1))
    x_pad = int(0.05 * (x2 - x1))
    cv2.rectangle(mask, (x1 - x_pad, y_top), (x2 + x_pad, y_split), 255, -1)
    return mask


def make_bg_mask(anchor_rgb: np.ndarray, lm) -> np.ndarray:
    """Inverse of face bbox + neck region — should be static across frames."""
    import mediapipe as mp
    h, w = anchor_rgb.shape[:2]
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=anchor_rgb))
    mask = np.full((h, w), 255, dtype=np.uint8)
    if not res.face_landmarks:
        return mask
    pts = np.array([[lm_.x * w, lm_.y * h] for lm_ in res.face_landmarks[0]])
    x1, y1 = pts.min(axis=0).astype(int)
    x2, y2 = pts.max(axis=0).astype(int)
    pad = int(0.20 * max(x2 - x1, y2 - y1))
    cv2.rectangle(mask, (max(0, x1 - pad), max(0, y1 - pad)),
                  (min(w, x2 + pad), h), 0, -1)
    return mask


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    anchor_bgr = cv2.imread(args.anchor)
    if anchor_bgr is None:
        raise FileNotFoundError(args.anchor)
    anchor_rgb = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2RGB)

    print("[init] insightface buffalo_l", flush=True)
    insf = InsightFaceClassifier(ctx_id=0, with_embedding=True)
    print("[init] mediapipe FaceLandmarker", flush=True)
    lm = make_landmarker()

    anchor_emb = insf.predict(anchor_bgr)["embedding"]
    if anchor_emb is None:
        raise RuntimeError("ArcFace failed on anchor")

    cap = cv2.VideoCapture(str(video_path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.right_half:
        out_w = src_w // 2
    else:
        out_w = src_w
    print(f"[init] video {src_w}x{src_h}@{src_fps:.1f}  frames={n_total}  out_w={out_w}", flush=True)

    # Anchor at output resolution for residual masks.
    anchor_at_out = cv2.resize(anchor_rgb, (out_w, src_h), interpolation=cv2.INTER_AREA)
    forehead_mask = make_forehead_mask(anchor_at_out, lm)
    bg_mask       = make_bg_mask(anchor_at_out, lm)

    rows = []
    prev = None
    bg_acc_sq = np.zeros((src_h, out_w, 3), dtype=np.float64)
    bg_acc    = np.zeros((src_h, out_w, 3), dtype=np.float64)
    bg_n = 0

    fi = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if fi % args.stride != 0:
            fi += 1
            continue
        if args.right_half:
            bgr = bgr[:, src_w // 2:]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        ins = insf.predict(bgr)
        det_ok = bool(ins["detected"])
        emb = ins["embedding"]
        cos = float(np.dot(anchor_emb, emb)) if (det_ok and emb is not None) else float("nan")
        lapv = laplacian_variance(rgb)

        flicker = float(cv2.absdiff(rgb, prev).mean()) if prev is not None else float("nan")
        prev = rgb

        ghost = hp_residual_to_anchor(rgb, anchor_at_out, forehead_mask)

        # Background-drift accumulation.
        bg_acc    += rgb.astype(np.float64)
        bg_acc_sq += rgb.astype(np.float64) ** 2
        bg_n += 1

        rows.append({
            "frame_idx": fi,
            "arcface_cos": cos,
            "det_ok":      det_ok,
            "lap_var":     lapv,
            "flicker":     flicker,
            "ghost_resid": ghost,
        })
        fi += 1

    cap.release()

    # Compute per-pixel temporal std on bg mask, take mean over masked area.
    if bg_n >= 2:
        mean = bg_acc / bg_n
        var  = (bg_acc_sq / bg_n) - mean ** 2
        var  = np.clip(var, 0, None)
        std  = np.sqrt(var).mean(axis=-1)
        bg_drift = float(std[bg_mask > 0].mean()) if bg_mask.any() else float("nan")
    else:
        bg_drift = float("nan")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print()
    print(f"[summary] {len(df)} sampled frames")
    print(df[["arcface_cos", "lap_var", "flicker", "ghost_resid"]].describe().round(3).to_string())
    print(f"\n  background-drift (mean per-pixel temporal std on bg mask): {bg_drift:.3f}")
    print(f"  face-detect rate: {df['det_ok'].mean():.3f}")
    print(f"  written: {args.out_csv}")

    if args.out_thumbs:
        # Worst-frame contact sheet by ghost_resid (clipped flicker also useful).
        worst = df.sort_values("ghost_resid", ascending=False).head(args.worst_n)
        cap = cv2.VideoCapture(str(video_path))
        rows_imgs = []
        for _, r in worst.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["frame_idx"]))
            ok, bgr = cap.read()
            if not ok:
                continue
            label = (f"f={int(r['frame_idx'])}  "
                     f"arc={r['arcface_cos']:.2f}  "
                     f"lap={r['lap_var']:.0f}  "
                     f"flk={r['flicker']:.1f}  "
                     f"gh={r['ghost_resid']:.2f}")
            cv2.putText(bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
            rows_imgs.append(bgr)
        cap.release()
        if rows_imgs:
            n_per_row = 4
            rows_2d = [rows_imgs[i:i + n_per_row] for i in range(0, len(rows_imgs), n_per_row)]
            grid = []
            for r in rows_2d:
                while len(r) < n_per_row:
                    r.append(np.zeros_like(r[0]))
                grid.append(np.hstack(r))
            sheet = np.vstack(grid)
            cv2.imwrite(args.out_thumbs, sheet)
            print(f"  written: {args.out_thumbs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
