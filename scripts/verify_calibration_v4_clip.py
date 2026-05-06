"""Render one clip with current closed_form_pose F_KP_REF and report
mean/median/p90 angular distance vs driver ARKit Euler stream.

Phase 0a gate for ARKit→PersonaLive bridge v4. Pass criterion:
    mean angular distance <= 0.05 rad

Reuses helpers from `calibrate_euler_signs_v3.py`:
  - angular_distance(R1, R2) -> float
  - load_input_R_seq(take_dir, n_frames, stride, start, signs) -> (N,3,3)
  - extract_R_render_seq(mp4, lm) -> (N,3,3)

Note vs plan v4 doc: actual helper names are `load_input_R_seq` and
`extract_R_render_seq` (the plan said `load_driver_rotmats` /
`extract_render_rotmats`); we use the real names here.

Usage:
  python scripts/verify_calibration_v4_clip.py \
    --take_dir data/llf-clips-auto/20260505_MySlate_5_yaw \
    --out_mp4 exp_output/arkit_bridge/calibration_v4/render_5_yaw.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from arkit_bridge.closed_form_pose import EULER_SIGNS  # noqa: E402
from calibrate_euler_signs_v3 import (  # noqa: E402
    angular_distance,
    extract_R_render_seq,
    load_input_R_seq,
)
from _mp_blendshape import make_landmarker  # noqa: E402


PERSONA_PY = "/home/newub/w/PersonaLive/.venv/bin/python"
ANCHOR = "data/llf-phase2/asian_m__06_neutral.midframe.png"
CKPT = "runs/student_v2_lam10/student_best.pt"
PASS_THRESH = 0.05  # rad


def render(take_dir: Path, out_mp4: Path, *, n_frames: int, stride: int) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PERSONA_PY, "scripts/apply_bridge_to_personalive.py",
        "--reference",   ANCHOR,
        "--take_dir",    str(take_dir),
        "--ckpt",        CKPT,
        "--out_path",    str(out_mp4),
        "--n_frames",    str(n_frames),
        "--stride",      str(stride),
        "--start_frame", "0",
        "--mode",        "bridge",
        # NOTE: no --euler_signs override; uses closed_form_pose default
        # (EULER_SIGNS + F_KP_REF baked in).
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if r.returncode:
        raise RuntimeError(f"render failed (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--out_mp4", required=True)
    ap.add_argument("--n_frames", type=int, default=600)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--skip_render", action="store_true",
                    help="reuse existing mp4 (idempotent path)")
    args = ap.parse_args()

    take_dir = Path(args.take_dir)
    out_mp4 = Path(args.out_mp4)

    if not take_dir.exists():
        raise SystemExit(f"take_dir not found: {take_dir}")

    if args.skip_render and not out_mp4.exists():
        raise SystemExit(f"--skip_render set but {out_mp4} does not exist")

    if not args.skip_render:
        render(take_dir, out_mp4, n_frames=args.n_frames, stride=args.stride)

    if not out_mp4.exists():
        raise SystemExit(f"render did not produce {out_mp4}")

    R_input = load_input_R_seq(
        take_dir, n_frames=args.n_frames, stride=args.stride,
        start=args.start, signs=EULER_SIGNS,
    )
    lm = make_landmarker()
    R_render = extract_R_render_seq(out_mp4, lm)

    n = min(len(R_input), len(R_render))
    if n == 0:
        raise SystemExit("no frames extracted (input or render empty)")
    R_input = R_input[:n]
    R_render = R_render[:n]

    ds = []
    for i in range(n):
        Rr = R_render[i]
        if not np.all(np.isfinite(Rr)):
            continue
        ds.append(angular_distance(R_input[i], Rr))
    ds = np.asarray(ds, dtype=np.float64)
    if ds.size == 0:
        raise SystemExit("no frames with valid mediapipe detection")

    mean_d = float(ds.mean())
    median_d = float(np.median(ds))
    p90_d = float(np.percentile(ds, 90))
    n_valid = int(ds.size)
    n_dropped = int(n - n_valid)

    print(f"\n# verify_calibration_v4_clip.py — {take_dir.name}")
    print(f"  EULER_SIGNS={EULER_SIGNS}  (F_KP_REF baked into closed_form_pose)")
    print(f"  frames: total={n}  valid={n_valid}  dropped(no-detect)={n_dropped}")
    print(f"  angular distance: mean={mean_d:+.4f}  median={median_d:+.4f}  "
          f"p90={p90_d:+.4f} rad")
    passed = mean_d <= PASS_THRESH
    print(f"  PASS (mean<={PASS_THRESH:.3f}): {passed}")

    # Sidecar JSON for downstream aggregation.
    sidecar = out_mp4.with_suffix(".verify.json")
    sidecar.write_text(json.dumps({
        "take_dir": str(take_dir),
        "out_mp4": str(out_mp4),
        "n_frames_total": n,
        "n_frames_valid": n_valid,
        "n_frames_dropped": n_dropped,
        "mean_angular_distance_rad": mean_d,
        "median_angular_distance_rad": median_d,
        "p90_angular_distance_rad": p90_d,
        "pass_threshold_rad": PASS_THRESH,
        "passed": passed,
        "euler_signs": list(EULER_SIGNS),
    }, indent=2))
    print(f"  wrote {sidecar}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
