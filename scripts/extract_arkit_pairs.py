"""Extract (b_expr, m_f) pairs from a Live Link Face take.

For each sampled frame i:
  1. Read MOV frame i (cv2 with CAP_PROP_ORIENTATION_AUTO=1 so iPhone
     Live Link MOVs come out portrait-upright).
  2. Crop face via StabilizedFaceCropper(strategy='ema') to match the
     inference path in apply_bridge_to_personalive.py / render_take.py.
  3. Resize crop to 224x224 -> frozen motion_encoder -> m_f (1,1,32,16).
  4. b_expr <- CSV[i, 0:52] + CSV[i, 55:61].
  5. Save {b_expr, m_f, frame_idx} pkl.

Resumable: skips frames whose pkl already exists.

NOTE 2026-05-05 evening: previous v1 of this script lacked rotation
auto-handling; iPhone MOVs were fed sideways into motion_encoder, so the
recorded m_f targets were a function of sideways faces. v2 student
trained on that corpus is orthogonal (cos≈0.01) to teacher m_f from
correctly-oriented inference. This rewrite re-extracts the corpus.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.llf_csv import load_llf_b61                       # noqa: E402
from arkit_bridge.symmetry import flip_b_expr                        # noqa: E402
from arkit_bridge.teacher_personalive import load_motion_encoder    # noqa: E402


def find_take_files(take_dir: Path) -> tuple[Path, Path]:
    movs = sorted(take_dir.glob("*_iPhone.mov"))
    csvs = sorted(take_dir.glob("*_iPhone.csv"))
    if not movs or not csvs:
        raise FileNotFoundError(f"no *_iPhone.mov/csv in {take_dir}")
    return movs[0], csvs[0]


def iter_frames(video_path: Path, stride: int):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    # Apply MOV rotation metadata so iPhone Live Link frames come out
    # upright (otherwise cv2 returns the encoded landscape stream).
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument(
        "--flip", action="store_true",
        help="Mirror image horizontally and apply L<->R swap on b_expr; "
             "writes pkls with `_flip` suffix.",
    )
    args = ap.parse_args()

    take_dir = Path(args.take_dir)
    take_name = take_dir.name
    mov, csv = find_take_files(take_dir)
    print(f"take: {take_name}\nmov: {mov}\ncsv: {csv}", flush=True)

    b_all = load_llf_b61(csv)
    print(f"loaded {len(b_all)} ARKit rows")

    me = load_motion_encoder(device=args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Match inference-time face crop (StabilizedFaceCropper, ema strategy).
    sys.path.insert(0, os.path.expanduser("~/w/PersonaLive"))
    from src.utils.util import StabilizedFaceCropper  # noqa: E402
    import mediapipe as mp                              # noqa: E402
    import cv2 as _cv2                                  # noqa: E402
    from PIL import Image                               # noqa: E402
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1)
    cropper = StabilizedFaceCropper(
        strategy="ema", ema_alpha=0.2, forehead_bias_frac=0.10,
        face_mesh=face_mesh,
    )

    n_done = n_skipped = n_kept = 0
    last_crop = None
    for fi, rgb in iter_frames(mov, args.stride):
        if fi >= len(b_all):
            print(f"frame {fi}: out of CSV range, stopping")
            break
        suffix = "_flip" if args.flip else ""
        out_path = os.path.join(
            args.out_dir, f"{take_name}_frame_{fi:06d}{suffix}.pkl")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        rgb_in = np.ascontiguousarray(rgb[:, ::-1]) if args.flip else rgb
        try:
            crop = cropper(Image.fromarray(rgb_in))
            last_crop = crop
        except (TypeError, IndexError):
            if last_crop is None:
                continue  # skip until first detection
            crop = last_crop
        crop_224 = _cv2.resize(crop, (224, 224), interpolation=_cv2.INTER_AREA)
        x = (
            torch.from_numpy(crop_224).float().permute(2, 0, 1)
            .unsqueeze(0).unsqueeze(2) / 255.0
        )
        with torch.no_grad():
            m_f = me(x.to(args.device)).squeeze(0).cpu().numpy()
        b_expr = np.concatenate(
            [b_all[fi, :52], b_all[fi, 55:61]], axis=0
        ).astype(np.float32)
        if args.flip:
            b_expr = flip_b_expr(b_expr)
        with open(out_path, "wb") as f:
            pickle.dump(
                {"b_expr": b_expr,
                 "m_f": m_f.astype(np.float16),
                 "frame_idx": int(fi)},
                f,
            )
        n_done += 1
        n_kept += 1
        if n_done % 100 == 0:
            print(f"  {n_done} pairs (skipped={n_skipped})")
        if args.max_frames and n_kept >= args.max_frames:
            break
    print(f"done: kept={n_kept} skipped={n_skipped}")


if __name__ == "__main__":
    main()
