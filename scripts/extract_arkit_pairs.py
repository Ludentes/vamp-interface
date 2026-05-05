"""Extract (b_expr, m_f) pairs from a Live Link Face take.

For each sampled frame i:
  1. Read MOV frame i.
  2. Loose-crop 224x224 with face roughly centered.
  3. Frozen motion_encoder(crop_224.unsqueeze(2)) -> m_f (1,1,32,16).
  4. b_expr <- CSV[i, 0:52] + CSV[i, 55:61].
  5. Save {b_expr, m_f, frame_idx} pkl.

Resumable: skips frames whose pkl already exists.
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
from arkit_bridge.teacher_personalive import load_motion_encoder    # noqa: E402


def find_take_files(take_dir: Path) -> tuple[Path, Path]:
    movs = sorted(take_dir.glob("*_iPhone.mov"))
    csvs = sorted(take_dir.glob("*_iPhone.csv"))
    if not movs or not csvs:
        raise FileNotFoundError(f"no *_iPhone.mov/csv in {take_dir}")
    return movs[0], csvs[0]


def loose_crop_centered(rgb, target: int):
    import cv2
    h, w = rgb.shape[:2]
    cx, cy = w // 2, int(h * 0.45)
    side = min(h, w)
    half = side // 2
    crop = rgb[max(0, cy - half):min(h, cy + half),
               max(0, cx - half):min(w, cx + half)]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)


def iter_frames(video_path: Path, stride: int):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
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
    args = ap.parse_args()

    take_dir = Path(args.take_dir)
    mov, csv = find_take_files(take_dir)
    print(f"mov: {mov}\ncsv: {csv}")

    b_all = load_llf_b61(csv)
    print(f"loaded {len(b_all)} ARKit rows")

    me = load_motion_encoder(device=args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    n_done = n_skipped = n_kept = 0
    for fi, rgb in iter_frames(mov, args.stride):
        if fi >= len(b_all):
            print(f"frame {fi}: out of CSV range, stopping")
            break
        out_path = os.path.join(args.out_dir, f"frame_{fi:06d}.pkl")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        crop = loose_crop_centered(rgb, 224)
        x = (
            torch.from_numpy(crop).float().permute(2, 0, 1)
            .unsqueeze(0).unsqueeze(2) / 255.0
        )
        with torch.no_grad():
            m_f = me(x.to(args.device)).squeeze(0).cpu().numpy()
        b_expr = np.concatenate(
            [b_all[fi, :52], b_all[fi, 55:61]], axis=0
        ).astype(np.float32)
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
