"""Render `teacher_full` and `bridge` for a take, compare per-region.

For each take, this calls apply_bridge_to_personalive twice:
  1. mode=teacher_full  -> gold reference (real motion_encoder + real pose)
  2. mode=bridge        -> student m_f + closed-form pose

Same anchor, same driving, same seed → any pixel-level difference is
attributable to the bridge substitution.

Then computes:
  - per-frame mean abs diff
  - per-region energy (mediapipe landmark boxes for brow/eye/mouth)
  - side-by-side mp4 (teacher | bridge | abs-diff×8)

Why: aggregate channel_recovery metrics measure the bridge against
driving CSV, which conflates "bridge fails" with "teacher already
fails on this anchor". This compares against the teacher upper bound,
isolating the bridge contribution.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


REGIONS = {
    # mediapipe FaceMesh landmarks (468 model)
    "brow":   list(range(46, 56)) + list(range(70, 80)) + [55, 65, 285, 295,
                                                            336, 296, 334, 293,
                                                            300, 276, 283, 282,
                                                            295, 285, 336, 296,
                                                            334, 293, 300, 276,
                                                            283, 282],
    "eye_l":  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
               160, 161, 246],
    "eye_r":  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386,
               385, 384, 398],
    "mouth":  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
               318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13,
               312, 311, 310, 415, 308],
}


def render(script_path, *, mode, ckpt, reference, take_dir, out_path,
           n_frames, stride, start_frame, save_mf=None, py=None):
    cmd = [
        py or sys.executable, str(script_path),
        "--reference", reference, "--take_dir", take_dir,
        "--ckpt", ckpt, "--out_path", out_path,
        "--n_frames", str(n_frames), "--stride", str(stride),
        "--start_frame", str(start_frame),
        "--mode", mode,
    ]
    if save_mf:
        cmd += ["--save_mf", save_mf]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    print(f"  render mode={mode} -> {out_path}", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1500:])
        raise RuntimeError(f"render failed mode={mode}")


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def get_face_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)


def landmark_xy(face_mesh, rgb):
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    h, w = rgb.shape[:2]
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


def region_box(landmarks, region_idxs, pad=8):
    pts = landmarks[region_idxs]
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    return int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(x1 + pad), int(y1 + pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=60)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--py", default=None)
    ap.add_argument("--skip_render", action="store_true",
                    help="reuse existing mp4s in out_dir")
    ap.add_argument("--teacher_cache_dir", default="exp_output/arkit_bridge/render/teacher_full_cache",
                    help="dir holding student-independent teacher_full renders. "
                         "Reused across student retrains. Cache key = take+anchor+n_frames+stride+start_frame.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    take_name = Path(args.take_dir).name
    anchor_stem = Path(args.reference).stem

    cache = Path(args.teacher_cache_dir); cache.mkdir(parents=True, exist_ok=True)
    teacher_cache_key = f"{take_name}__{anchor_stem}__n{args.n_frames}_s{args.stride}_o{args.start_frame}.mp4"
    teacher_cached = cache / teacher_cache_key
    teacher_mp4 = out / f"{take_name}_teacher_full.mp4"
    bridge_mp4 = out / f"{take_name}_bridge.mp4"
    apply_script = Path(__file__).parent / "apply_bridge_to_personalive.py"

    if not args.skip_render:
        # Teacher: only render if not cached (teacher is student-independent)
        teacher_mf_sidecar = teacher_cached.with_suffix(".mf.npz")
        if not teacher_cached.exists() or teacher_cached.stat().st_size < 1000:
            render(apply_script, mode="teacher_full",
                   ckpt=args.ckpt, reference=args.reference,
                   take_dir=args.take_dir, out_path=str(teacher_cached),
                   n_frames=args.n_frames, stride=args.stride,
                   start_frame=args.start_frame, py=args.py,
                   save_mf=str(teacher_mf_sidecar))
        else:
            print(f"  teacher cached: {teacher_cached}", flush=True)
        # Symlink the cache into the run's out_dir so downstream tools find it locally
        if teacher_mp4.exists() or teacher_mp4.is_symlink():
            teacher_mp4.unlink()
        teacher_mp4.symlink_to(teacher_cached.resolve())

        # Bridge: per-student
        render(apply_script, mode="bridge",
               ckpt=args.ckpt, reference=args.reference,
               take_dir=args.take_dir, out_path=str(bridge_mp4),
               n_frames=args.n_frames, stride=args.stride,
               start_frame=args.start_frame, py=args.py)

    # Compare frame-by-frame
    t = read_video(teacher_mp4); b = read_video(bridge_mp4)
    n = min(len(t), len(b))
    t = t[:n]; b = b[:n]
    diff = np.abs(t.astype(np.float32) - b.astype(np.float32))  # (n, h, w, 3)
    mean_abs_per_frame = diff.mean(axis=(1, 2, 3))

    # Per-region energy via mediapipe landmarks on TEACHER frames
    fm = get_face_mesh()
    region_energy = {r: [] for r in REGIONS}
    for i in range(n):
        lms = landmark_xy(fm, t[i])
        if lms is None:
            for r in REGIONS: region_energy[r].append(np.nan)
            continue
        for r, idxs in REGIONS.items():
            x0, y0, x1, y1 = region_box(lms, idxs)
            patch = diff[i, y0:y1, x0:x1].mean() if (x1 > x0 and y1 > y0) else np.nan
            region_energy[r].append(float(patch))

    # Stats
    summary = {
        "take": take_name,
        "n_frames_compared": int(n),
        "mean_abs_per_frame": {
            "mean": float(mean_abs_per_frame.mean()),
            "p50":  float(np.percentile(mean_abs_per_frame, 50)),
            "p90":  float(np.percentile(mean_abs_per_frame, 90)),
            "max":  float(mean_abs_per_frame.max()),
            "argmax": int(mean_abs_per_frame.argmax()),
        },
        "region_energy": {
            r: {
                "mean": float(np.nanmean(region_energy[r])),
                "p90":  float(np.nanpercentile(region_energy[r], 90)),
                "argmax": int(np.nanargmax(region_energy[r])),
            } for r in REGIONS
        },
    }

    # Worst frames overall
    top_worst = mean_abs_per_frame.argsort()[::-1][:10].tolist()
    summary["top10_worst_frames"] = [int(x) for x in top_worst]

    # Side-by-side mp4
    side_path = out / f"{take_name}_compare.mp4"
    h, w = t.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(side_path), fourcc, 30, (w * 3, h))
    for i in range(n):
        d_vis = np.clip(diff[i] * 8, 0, 255).astype(np.uint8)
        row = np.concatenate([t[i], b[i], d_vis], axis=1)
        # label
        for x_off, name in [(10, "TEACHER"), (w + 10, "BRIDGE"), (2*w + 10, "DIFF×8")]:
            cv2.putText(row, name, (x_off, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(row, f"f{i:03d}  abs={mean_abs_per_frame[i]:.2f}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    vw.release()
    summary["side_by_side"] = str(side_path)

    json_path = out / f"{take_name}_compare.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {json_path} and {side_path}")


if __name__ == "__main__":
    main()
