"""Crop-wobble diagnostic for ARKit extraction pipeline.

For one take's MOV, run mediapipe face_mesh and compute per-frame
bbox (cx, cy, sz). Reports adjacent-frame deltas in pixels for both:
  - raw (per-frame) — raw input jitter
  - ema (alpha=0.2) — what extract_arkit_pairs.py actually feeds to MotEncoder

If the EMA-smoothed center jitter is >5px frame-to-frame on a calm-ish
take, residual input wobble is plausibly contaminating teacher m_f.

Output JSON: per-mode {center_dx_p50, center_dx_p95, side_d_p50,
side_d_p95, mean_side, n_frames}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mov", required=True)
    ap.add_argument("--max_frames", type=int, default=400)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--ema_alpha", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(args.mov)
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1)

    raw_bb = []  # (cx, cy, sz)
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok or kept >= args.max_frames:
            break
        if idx % args.stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                xs = np.array([p.x * w for p in lm])
                ys = np.array([p.y * h for p in lm])
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                length = max(x2 - x1, y2 - y1) * 1.1
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                raw_bb.append((float(cx), float(cy), float(length)))
            kept += 1
        idx += 1
    cap.release()

    if len(raw_bb) < 2:
        print(json.dumps({"error": "no detections"}))
        sys.exit(1)

    raw = np.array(raw_bb)  # (N, 3) — cx, cy, sz
    # EMA pass.
    a = args.ema_alpha
    ema = np.zeros_like(raw)
    ema[0] = raw[0]
    for i in range(1, len(raw)):
        ema[i] = a * raw[i] + (1 - a) * ema[i - 1]

    def stats(bb):
        d_cx = np.abs(np.diff(bb[:, 0]))
        d_cy = np.abs(np.diff(bb[:, 1]))
        d_sz = np.abs(np.diff(bb[:, 2]))
        return {
            "n_frames": int(len(bb)),
            "mean_side": float(np.mean(bb[:, 2])),
            "center_dx_p50": float(np.percentile(d_cx, 50)),
            "center_dx_p95": float(np.percentile(d_cx, 95)),
            "center_dy_p50": float(np.percentile(d_cy, 50)),
            "center_dy_p95": float(np.percentile(d_cy, 95)),
            "side_d_p50": float(np.percentile(d_sz, 50)),
            "side_d_p95": float(np.percentile(d_sz, 95)),
            # As fraction of mean side, since absolute pixels depend on input res.
            "center_dx_p50_pct": float(np.percentile(d_cx, 50)
                                       / np.mean(bb[:, 2]) * 100),
            "center_dx_p95_pct": float(np.percentile(d_cx, 95)
                                       / np.mean(bb[:, 2]) * 100),
        }

    out = {"raw": stats(raw), "ema": stats(ema), "ema_alpha": args.ema_alpha}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
