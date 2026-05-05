"""8-way Euler sign-flip calibration: ARKit (yaw,pitch,roll) → LivePortrait k_d.

For sampled frames, render real motion_extractor output as ground-truth
keypoints, then for each of 8 sign combos compute closed-form
keypoints from ARKit Euler angles and pick the combo with smallest
mean keypoint L2 error.
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.closed_form_pose import euler_to_rotmat, compose_kd  # noqa: E402
from arkit_bridge.llf_csv import load_llf_b61                          # noqa: E402
from arkit_bridge.teacher_personalive import load_motion_extractor     # noqa: E402


def loose_crop_centered(rgb, target):
    import cv2
    h, w = rgb.shape[:2]
    cx, cy = w // 2, int(h * 0.45)
    side = min(h, w); half = side // 2
    return cv2.resize(rgb[max(0, cy - half):min(h, cy + half),
                          max(0, cx - half):min(w, cx + half)],
                      (target, target), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=30)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import cv2
    take_dir = Path(args.take_dir)
    mov = next(take_dir.glob("*_iPhone.mov"))
    csv = next(take_dir.glob("*_iPhone.csv"))
    b_all = load_llf_b61(csv)
    mx = load_motion_extractor(device=args.device)

    cap = cv2.VideoCapture(str(mov))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame 0 from {mov}")
    rgb0 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop0 = loose_crop_centered(rgb0, 256)
    x0 = (
        torch.from_numpy(crop0).float().permute(2, 0, 1).unsqueeze(0)
        .to(args.device) / 255.0
    )
    with torch.no_grad():
        # mx.detector returns the canonical dict (kp, t, scale, pitch, yaw, roll);
        # mx(x) returns the *already transformed* k_d. For kp_ref/t_ref/s_ref
        # we need canonical, so call detector directly.
        info0 = mx.detect_raw(x0)
    kp_ref = info0["kp"].reshape(1, -1, 3).cpu()
    t_ref = info0["t"].cpu()
    s_ref = info0["scale"].cpu()
    print(f"reference frame 0: kp shape {tuple(kp_ref.shape)}, "
          f"t {t_ref.flatten().tolist()}, s {s_ref.flatten().tolist()}",
          flush=True)

    samples = []
    for k in range(args.n_frames):
        fi = (k + 1) * args.stride
        if fi >= len(b_all):
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop = loose_crop_centered(rgb, 256)
        x = (
            torch.from_numpy(crop).float().permute(2, 0, 1).unsqueeze(0)
            .to(args.device) / 255.0
        )
        with torch.no_grad():
            k_d_real = mx(x).cpu()  # (1, K, 3) already transformed
        b = b_all[fi]
        samples.append((
            k_d_real,
            torch.tensor([b[52], b[53], b[54]], dtype=torch.float32),
        ))
    cap.release()
    print(f"collected {len(samples)} sample frames", flush=True)

    results = {}
    for s_y, s_p, s_r in product([+1, -1], repeat=3):
        errs = []
        for k_d_real, ypr in samples:
            R = euler_to_rotmat(
                torch.tensor(float(s_y) * ypr[0]),
                torch.tensor(float(s_p) * ypr[1]),
                torch.tensor(float(s_r) * ypr[2]),
            )
            k_d_cf = compose_kd(kp_ref, R.unsqueeze(0), s_ref, t_ref)
            errs.append((k_d_cf - k_d_real).pow(2).mean().sqrt().item())
        results[f"({s_y:+d}, {s_p:+d}, {s_r:+d})"] = float(np.mean(errs))

    best = min(results, key=results.get)
    print("\nSign combo (yaw,pitch,roll) -> mean keypoint L2 err:")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]):
        marker = " <-- best" if k == best else ""
        print(f"  {k}  err={v:.5f}{marker}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "best": best, "n": len(samples)},
                  f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
