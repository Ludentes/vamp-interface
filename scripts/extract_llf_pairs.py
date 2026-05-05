"""Extract (b₆₁, T) distillation pairs from a Live Link Face take.

Inputs:
  --take_dir  e.g. data/llf-takes/20260505_MySlate_2/
              must contain MySlate_*_iPhone.mov + MySlate_*_iPhone.csv

For each sampled frame i:
  1. read MOV frame i
  2. loose face crop -> 512x512 RGB
  3. DWPose -> stick image -> frozen PoseGuider -> T (320,1,64,64)
  4. b₆₁ <- CSV row i (Live Link recorded ARKit; not MediaPipe)
  5. save (b, T) pkl

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

from arkit_bridge.extractors import get_dwpose_image, loose_face_crop  # noqa: E402
from arkit_bridge.llf_csv import load_llf_b61  # noqa: E402
from arkit_bridge.teacher import load_frozen_teacher  # noqa: E402

PERSONA_PG = os.path.expanduser(
    "~/w/PersonaLive/pretrained_weights/personalive/pose_guider.pth"
)


def find_take_files(take_dir: Path) -> tuple[Path, Path]:
    movs = sorted(take_dir.glob("*_iPhone.mov"))
    csvs = sorted(take_dir.glob("*_iPhone.csv"))
    if not movs or not csvs:
        raise FileNotFoundError(f"no *_iPhone.mov/csv in {take_dir}")
    return movs[0], csvs[0]


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
    ap.add_argument("--stride", type=int, default=2,
                    help="sample every Nth frame (60fps -> 30fps at stride=2)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    take_dir = Path(args.take_dir)
    mov, csv_path = find_take_files(take_dir)
    print(f"mov:  {mov}\ncsv:  {csv_path}")

    b_all = load_llf_b61(csv_path)  # (N, 61)
    print(f"loaded {len(b_all)} ARKit rows from CSV")

    os.makedirs(args.out_dir, exist_ok=True)
    teacher = load_frozen_teacher(PERSONA_PG, device=args.device)

    n_done = n_skipped = n_kept = 0
    for fi, rgb in iter_frames(mov, args.stride):
        if fi >= len(b_all):
            print(f"frame {fi}: out of CSV range; stopping")
            break
        out_path = os.path.join(args.out_dir, f"frame_{fi:06d}.pkl")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        crop = loose_face_crop(rgb, target=512)
        if crop is None:
            continue
        try:
            stick = get_dwpose_image(crop)
        except Exception as e:
            print(f"frame {fi}: DWPose failed: {e}")
            continue
        stick_t = torch.from_numpy(stick).float().permute(2, 0, 1) / 255.0
        stick_t = stick_t.unsqueeze(0).unsqueeze(2).to(args.device)
        with torch.no_grad():
            T = teacher(stick_t).squeeze(0).cpu().numpy()  # (320,1,64,64)
        b = b_all[fi]
        with open(out_path, "wb") as f:
            pickle.dump({"b": b.astype(np.float32),
                         "T": T.astype(np.float16),
                         "frame_idx": int(fi)}, f)
        n_done += 1; n_kept += 1
        if n_done % 50 == 0:
            print(f"  {n_done} pairs written (skipped existing: {n_skipped})")
        if args.max_frames and n_kept >= args.max_frames:
            break
    print(f"done: kept={n_kept} skipped={n_skipped}")


if __name__ == "__main__":
    main()
