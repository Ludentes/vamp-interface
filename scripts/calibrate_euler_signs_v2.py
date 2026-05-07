"""Re-calibration of ARKit→LivePortrait Euler signs via rendered-output yaw.

Replaces `scripts/calibrate_euler_signs.py`. The original metric — L2 between
closed-form k_d_cf and PersonaLive motion_extractor k_d_real — is partially
yaw-symmetric on a near-symmetric kp_ref and could not distinguish yaw sign
(picked +1 when the correct value was -1; see
docs/research/2026-05-06-yaw-sign-flip-fix.md).

This metric goes through the renderer:
  1. Render N frames of `take` through PersonaLive `bridge` mode for each of
     the 8 (sy, sp, sr) ∈ {±1}³ combos, using --euler_signs override.
  2. Extract mediapipe ypr per frame from the rendered output.
  3. Score = sum over axes of Pearson correlation between rendered ypr and
     input ypr (b_expr[52:55] from LLF CSV).
  4. Pick combo with highest score.

Negative control: synthetically negate one input axis at score time and verify
the new winner is the corresponding sign-flipped combo. If not, the metric
is still degenerate — escalate.

Cost: ~30s startup + ~4s render per combo + mediapipe — ~5–6 min total on a
high-yaw 120-frame slice.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mp_blendshape import make_landmarker, extract_one  # noqa: E402


def load_input_ypr(take_dir: Path, n_frames: int, stride: int, start: int) -> np.ndarray:
    """Read LLF CSV, extract head (yaw, pitch, roll) for the same frame slice
    apply_bridge_to_personalive uses. Returns (n_frames, 3) radians."""
    sys.path.insert(0, "src")
    from arkit_bridge.llf_csv import load_llf_b61  # type: ignore
    csv = next(take_dir.glob("*_iPhone.csv"))
    b_all = load_llf_b61(csv)
    idxs = start + np.arange(n_frames) * stride
    idxs = idxs[idxs < len(b_all)]
    return b_all[idxs, 52:55].astype(np.float32)


def extract_rendered_ypr(mp4: Path, lm) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    rows = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        _, ypr = extract_one(lm, rgb)
        rows.append(ypr)
    cap.release()
    return np.stack(rows) if rows else np.zeros((0, 3))


def render_one(*, py: str, take_dir: Path, anchor: Path, ckpt: Path,
               out: Path, n_frames: int, stride: int, signs: tuple[float, float, float]):
    cmd = [
        py, "scripts/apply_bridge_to_personalive.py",
        "--reference", str(anchor),
        "--take_dir",  str(take_dir),
        "--ckpt",      str(ckpt),
        "--out_path",  str(out),
        "--n_frames",  str(n_frames),
        "--stride",    str(stride),
        "--start_frame", "0",
        "--mode",      "bridge",
        # Use '=' joining so argparse doesn't try to interpret "-1,..." as a flag
        f"--euler_signs={','.join(f'{s:+.0f}' for s in signs)}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1500:])
        raise RuntimeError(f"render failed for signs={signs}")


def score(rendered: np.ndarray, input_ypr: np.ndarray, *, active_thresh: float) -> dict:
    """Pearson correlation per axis, on frames where the input axis is active."""
    n = min(len(rendered), len(input_ypr))
    out = {}
    s = 0.0
    for i, name in enumerate(("yaw", "pitch", "roll")):
        a = input_ypr[:n, i]; c = rendered[:n, i]
        m = ~np.isnan(c) & ~np.isnan(a) & (np.abs(a) > active_thresh)
        if m.sum() < 10:
            out[name] = {"r": float("nan"), "n_active": int(m.sum())}; continue
        r = float(np.corrcoef(c[m], a[m])[0, 1])
        out[name] = {"r": r, "n_active": int(m.sum())}
        s += r
    out["score"] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", default="data/llf-takes/20260505_MySlate_5",
                    help="take with strong yaw motion (take 5: |yaw|>10° on ~77 frames at n_frames=1200)")
    ap.add_argument("--anchor", default="data/llf-phase2/asian_m__06_neutral.midframe.png")
    ap.add_argument("--ckpt",   default="runs/student_v3_lam10/student_best.pt")
    ap.add_argument("--out_dir", default="exp_output/arkit_bridge/calibration_v2")
    ap.add_argument("--n_frames", type=int, default=600,
                    help="enough to span yaw motion; ~3 min total at this size on 5090")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--py", default="/home/newub/w/PersonaLive/.venv/bin/python")
    ap.add_argument("--active_thresh_rad", type=float, default=0.10)
    ap.add_argument("--skip_render", action="store_true",
                    help="reuse mp4s already in out_dir")
    ap.add_argument("--negative_control", action="store_true",
                    help="after picking best, also score with input yaw negated; "
                         "expect winner to flip on the yaw sign")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    take_dir = Path(args.take_dir).resolve()
    anchor = Path(args.anchor).resolve()
    ckpt = Path(args.ckpt).resolve()

    input_ypr = load_input_ypr(take_dir, args.n_frames, args.stride, start=0)
    print(f"input ypr: {len(input_ypr)} frames, "
          f"yaw n_active={(np.abs(input_ypr[:, 0]) > args.active_thresh_rad).sum()}")

    lm = make_landmarker()
    results = {}
    combos = list(itertools.product([+1.0, -1.0], repeat=3))
    for k, signs in enumerate(combos, 1):
        tag = f"y{int(signs[0]):+d}_p{int(signs[1]):+d}_r{int(signs[2]):+d}"
        mp4 = out / f"render_{tag}.mp4"
        if not args.skip_render or not mp4.exists():
            print(f"[{k}/8] render signs={signs} -> {mp4.name}", flush=True)
            render_one(py=args.py, take_dir=take_dir, anchor=anchor, ckpt=ckpt,
                       out=mp4, n_frames=args.n_frames, stride=args.stride, signs=signs)
        rendered = extract_rendered_ypr(mp4, lm)
        sc = score(rendered, input_ypr, active_thresh=args.active_thresh_rad)
        results[tag] = sc
        print(f"   yaw r={sc['yaw']['r']:+.3f}  pitch r={sc['pitch']['r']:+.3f}  "
              f"roll r={sc['roll']['r']:+.3f}  score={sc['score']:+.3f}")

    best = max(results, key=lambda k: results[k]["score"])
    print("\n# Sign combo (sy,sp,sr) — sum-of-correlations:")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]["score"]):
        marker = "  <-- BEST" if k == best else ""
        print(f"  {k}  score={v['score']:+.3f}  yaw r={v['yaw']['r']:+.3f}{marker}")

    out_doc = {"results": results, "best": best, "n_frames": args.n_frames,
               "take_dir": str(take_dir),
               "active_thresh_rad": args.active_thresh_rad}

    if args.negative_control:
        # Recompute scoring with input yaw negated. The winner should be the
        # current best with sy flipped. If it isn't, metric still degenerate.
        neg_input = input_ypr.copy(); neg_input[:, 0] = -neg_input[:, 0]
        neg_results = {}
        for k, signs in enumerate(combos):
            tag = f"y{int(signs[0]):+d}_p{int(signs[1]):+d}_r{int(signs[2]):+d}"
            mp4 = out / f"render_{tag}.mp4"
            rendered = extract_rendered_ypr(mp4, lm)
            sc = score(rendered, neg_input, active_thresh=args.active_thresh_rad)
            neg_results[tag] = sc
        neg_best = max(neg_results, key=lambda k: neg_results[k]["score"])
        # Expected: best's sy flipped, p/r unchanged
        exp_tag = best.replace("y+1", "Y_TMP").replace("y-1", "y+1").replace("Y_TMP", "y-1")
        ok = (neg_best == exp_tag)
        print(f"\n# Negative control: input yaw negated → expected winner '{exp_tag}', got '{neg_best}'  [{'PASS' if ok else 'FAIL'}]")
        out_doc["negative_control"] = {"expected": exp_tag, "got": neg_best,
                                       "pass": ok, "neg_results": neg_results}

    (out / "calibration_v2.json").write_text(json.dumps(out_doc, indent=2))
    print(f"\nwrote {out}/calibration_v2.json")


if __name__ == "__main__":
    main()
