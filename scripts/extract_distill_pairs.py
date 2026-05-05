"""Extract (b₆₁, T) distillation pairs from a video.

For each sampled frame:
  1. Loose face crop -> 512x512 RGB
  2. MediaPipe -> b₆₁
  3. DWPose -> stick image (512,512,3)
  4. Frozen PoseGuider(stick image) -> T (320,1,64,64)
  5. Save (b, T) to a single .pkl per frame.

Resumable: skips frames whose output .pkl already exists.
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

from arkit_bridge.extractors import (  # noqa: E402
    extract_arkit_61, get_dwpose_image, loose_face_crop,
)
from arkit_bridge.teacher import load_frozen_teacher  # noqa: E402

PERSONA_PG = os.path.expanduser(
    "~/w/PersonaLive/pretrained_weights/personalive/pose_guider.pth"
)


def iter_frames(video_path: str, stride: int):
    import cv2
    cap = cv2.VideoCapture(video_path)
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
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    teacher = load_frozen_teacher(PERSONA_PG, device=args.device)

    n_done = 0
    n_skipped = 0
    n_kept = 0
    for fi, rgb in iter_frames(args.video, args.stride):
        out_path = os.path.join(args.out_dir, f"frame_{fi:06d}.pkl")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        crop = loose_face_crop(rgb, target=512)
        if crop is None:
            continue
        b = extract_arkit_61(crop)
        if b is None:
            continue
        try:
            stick = get_dwpose_image(crop)
        except Exception as e:
            print(f"frame {fi}: DWPose failed: {e}")
            continue
        stick_t = torch.from_numpy(stick).float().permute(2, 0, 1) / 255.0
        stick_t = stick_t.unsqueeze(0).unsqueeze(2).to(args.device)
        with torch.no_grad():
            T = teacher(stick_t).squeeze(0).cpu().numpy()
        with open(out_path, "wb") as f:
            pickle.dump({"b": b, "T": T.astype(np.float16)}, f)
        n_done += 1; n_kept += 1
        if n_done % 50 == 0:
            print(f"  {n_done} pairs written (skipped existing: {n_skipped})")
        if args.max_frames and n_kept >= args.max_frames:
            break
    print(f"done: kept={n_kept} skipped={n_skipped}")


if __name__ == "__main__":
    main()
