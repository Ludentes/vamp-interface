"""Per-channel recovery: which expressions reach pixels, which don't?

Hypothesis from rendered-video inspection: head pose tracks (with
artifacts), but mouth/eye/most expression channels are nearly flat in
the output regardless of driving input.

For each ARKit blendshape and the head ypr triplet:
  - std(rendered_mp) / std(driving_mp)        — output-amplitude ratio
  - std(rendered_mp) / std(arkit_csv_channel) — output vs ground truth
  - pearson(rendered_mp, arkit_csv_channel)   — does it track at all
  - pearson(driving_mp,  arkit_csv_channel)   — sanity: does mediapipe
                                                 even pick this up on
                                                 the iPhone face?

Channels are grouped by category (eye / jaw / mouth / brow / cheek /
nose) to surface which classes of expression survive vs collapse.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_render_expression import (  # type: ignore
    load_video_frames, mediapipe_blendshapes, pearson,
)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from arkit_bridge.llf_csv import ARKIT_BLENDSHAPE_NAMES, load_llf_b61


# Mediapipe FaceLandmarker order (after dropping _neutral). 51 channels.
MP_ORDER = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker",
    "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
assert len(MP_ORDER) == 51


def _category(name: str) -> str:
    n = name.lower()
    if n.startswith("eye"):
        return "eye"
    if n.startswith("jaw"):
        return "jaw"
    if n.startswith("mouth"):
        return "mouth"
    if n.startswith("brow"):
        return "brow"
    if n.startswith("cheek"):
        return "cheek"
    if n.startswith("nose"):
        return "nose"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True)
    ap.add_argument("--take_dir", required=True)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Cap rendered frame count (0 = all)")
    args = ap.parse_args()

    rendered = load_video_frames(args.render)
    if args.max_frames and len(rendered) > args.max_frames:
        rendered = rendered[: args.max_frames]
    n = len(rendered)
    print(f"loaded {n} rendered frames", flush=True)

    csv = next(Path(args.take_dir).glob("*_iPhone.csv"))
    b_all = load_llf_b61(csv)
    arkit_idx = args.start_frame + np.arange(n) * args.stride
    arkit_idx = np.clip(arkit_idx, 0, len(b_all) - 1)

    # Driving frames from MOV
    import cv2
    mov = next(Path(args.take_dir).glob("*_iPhone.mov"))
    cap = cv2.VideoCapture(str(mov))
    drv = []
    for fi in arkit_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        drv.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok
                   else np.zeros_like(rendered[0]))
    cap.release()

    print("mediapipe on rendered...", flush=True)
    r_bs, r_ypr = mediapipe_blendshapes(rendered, rotate_iphone=False)
    print("mediapipe on driving...", flush=True)
    d_bs, d_ypr = mediapipe_blendshapes(drv, rotate_iphone=True)

    # Reorder mediapipe -> canonical ARKit
    name_to_canon = {n: i for i, n in enumerate(ARKIT_BLENDSHAPE_NAMES)}
    mp_to_canon = np.array([name_to_canon[n] for n in MP_ORDER], dtype=np.int64)
    r_canon = np.full((n, 52), np.nan, dtype=np.float32)
    d_canon = np.full((n, 52), np.nan, dtype=np.float32)
    r_canon[:, mp_to_canon] = r_bs
    d_canon[:, mp_to_canon] = d_bs

    csv_bs = b_all[arkit_idx, :52]      # ground truth
    csv_ypr = b_all[arkit_idx, 52:55]   # head yaw/pitch/roll

    per_channel = []
    cat_agg: dict[str, list] = {}
    for c in range(52):
        name = ARKIT_BLENDSHAPE_NAMES[c]
        if name == "tongueOut":
            continue  # mediapipe doesn't emit
        cat = _category(name)
        r = r_canon[:, c]; d = d_canon[:, c]; t = csv_bs[:, c]
        mask_r = np.isfinite(r); mask_d = np.isfinite(d)
        if mask_r.sum() < 3 or mask_d.sum() < 3:
            continue
        std_r = float(r[mask_r].std()); std_d = float(d[mask_d].std())
        std_t = float(t.std())
        eps = 1e-6
        rec = {
            "name": name,
            "category": cat,
            "csv_std": std_t,
            "driving_mp_std": std_d,
            "rendered_mp_std": std_r,
            "amp_vs_driving": std_r / max(std_d, eps),
            "amp_vs_csv": std_r / max(std_t, eps),
            "r_render_vs_csv": pearson(r, t),
            "r_driving_vs_csv": pearson(d, t),
            "r_render_vs_driving": pearson(r, d),
        }
        per_channel.append(rec)
        cat_agg.setdefault(cat, []).append(rec)

    # Head ypr
    ypr_axes = ["yaw", "pitch", "roll"]
    head = []
    for k, ax in enumerate(ypr_axes):
        r = r_ypr[:, k]; d = d_ypr[:, k]; t = csv_ypr[:, k]
        mask_r = np.isfinite(r); mask_d = np.isfinite(d)
        if mask_r.sum() < 3 or mask_d.sum() < 3:
            continue
        head.append({
            "axis": ax,
            "csv_std": float(t.std()),
            "driving_mp_std": float(d[mask_d].std()),
            "rendered_mp_std": float(r[mask_r].std()),
            "amp_vs_driving": float(r[mask_r].std() / max(d[mask_d].std(), 1e-6)),
            "amp_vs_csv": float(r[mask_r].std() / max(t.std(), 1e-6)),
            "r_render_vs_csv": pearson(r, t),
            "r_driving_vs_csv": pearson(d, t),
        })

    def cat_summary(recs):
        # Weighted by csv_std so static channels don't dominate.
        w = np.array([r["csv_std"] for r in recs])
        w = w / max(w.sum(), 1e-9)
        return {
            "n_channels": len(recs),
            "median_amp_vs_driving": float(np.median([r["amp_vs_driving"] for r in recs])),
            "weighted_amp_vs_driving": float(sum(r["amp_vs_driving"] * wi for r, wi in zip(recs, w))),
            "median_amp_vs_csv": float(np.median([r["amp_vs_csv"] for r in recs])),
            "median_r_render_vs_csv": float(np.median([r["r_render_vs_csv"] for r in recs])),
            "median_r_driving_vs_csv": float(np.median([r["r_driving_vs_csv"] for r in recs])),
        }

    summary = {
        "n_frames": n,
        "categories": {c: cat_summary(v) for c, v in sorted(cat_agg.items())},
        "head_ypr": head,
        "channels_top_active_in_csv": sorted(
            per_channel, key=lambda r: -r["csv_std"]
        )[:15],
        "channels_worst_amp_vs_driving": sorted(
            per_channel, key=lambda r: r["amp_vs_driving"]
        )[:15],
        "all_channels": per_channel,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    # Compact human readout
    print("\n=== Category recovery (rendered vs driving mediapipe) ===")
    print(f"{'cat':6s} {'n':>3s} {'med_amp':>8s} {'w_amp':>7s} {'med_r':>7s}")
    for cat, s in summary["categories"].items():
        print(f"{cat:6s} {s['n_channels']:>3d} "
              f"{s['median_amp_vs_driving']:>8.3f} "
              f"{s['weighted_amp_vs_driving']:>7.3f} "
              f"{s['median_r_render_vs_csv']:>+7.3f}")
    print("\n=== Head ypr ===")
    for h in head:
        print(f"{h['axis']:5s} amp_vs_driving={h['amp_vs_driving']:.3f} "
              f"r_render_vs_csv={h['r_render_vs_csv']:+.3f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
