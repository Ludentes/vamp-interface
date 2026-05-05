"""Output-side diagnostic: does expression reach pixels?

Two round-trip metrics on a rendered mp4:

  1. LPIPS-amplitude correlation
       For each rendered frame f_i: lpips_i = LPIPS(f_i, anchor)
       For each driving frame:      bnorm_i = ||b_expr_i - b_mean||_2
       Pearson r(lpips, bnorm). If high b -> high LPIPS, the bridge
       propagates expression; if r is near zero, expression got
       smoothed out somewhere in pose_guider/motion_encoder/UNet.

  2. Mediapipe blendshape recovery
       Run FaceLandmarker (with blendshapes) on each rendered frame
       and on each driving MOV frame; cosine-correlate the two
       52-d vectors per frame. If round-trip cos drops below the
       teacher-render baseline, the bridge specifically (vs pipeline)
       is the lossy step.

Outputs JSON + a per-frame .csv for inspection.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from arkit_bridge.llf_csv import load_llf_b61


def load_video_frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_anchor(path, size):
    import cv2
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    side = min(h, w)
    img = img[(h - side) // 2:(h - side) // 2 + side,
              (w - side) // 2:(w - side) // 2 + side]
    return cv2.resize(img, (size, size))


def to_tensor(rgb_uint8, device):
    t = torch.from_numpy(rgb_uint8).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(device)


def lpips_curve(frames, anchor_rgb, device):
    import lpips
    net = lpips.LPIPS(net="alex").to(device).eval()
    a = to_tensor(anchor_rgb, device)
    out = []
    with torch.no_grad():
        for f in frames:
            ft = to_tensor(f, device)
            d = net(ft, a).item()
            out.append(d)
    return np.array(out, dtype=np.float32)


def arcface_drift(frames, anchor_rgb, device="cuda"):
    """Per-frame ArcFace cosine vs anchor.

    Lower cos => identity drifted further from the anchor. Persistent
    drift over time is a Tier-3 viability gate.
    """
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))
    import cv2
    anchor_bgr = cv2.cvtColor(anchor_rgb, cv2.COLOR_RGB2BGR)
    afaces = app.get(anchor_bgr)
    if not afaces:
        return None, None
    a_emb = afaces[0].normed_embedding
    out = np.full(len(frames), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            continue
        out[i] = float(np.dot(a_emb, faces[0].normed_embedding))
    return out, float(np.dot(a_emb, a_emb))


def _rotate_for_iphone(frame, mov_path):
    """LLF iPhone takes have Orientation=4 (left-side-up). Mediapipe
    will fail to detect a sideways face — rotate 90° CW once before
    landmarking. Only applies to MOV-sourced frames."""
    return np.rot90(frame, k=-1).copy()


def mediapipe_blendshapes(frames, *, rotate_iphone=False):
    """Returns (N, 52) np.float32, NaN where detection failed."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision as mp_vis
    # Look for the bundled face_landmarker.task asset; falls back to
    # download if missing.
    asset = Path.home() / ".cache/mediapipe/face_landmarker.task"
    if not asset.exists():
        import urllib.request
        asset.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/"
               "face_landmarker.task")
        urllib.request.urlretrieve(url, asset)
    opts = mp_vis.FaceLandmarkerOptions(
        base_options=mp_py.BaseOptions(model_asset_path=str(asset)),
        output_face_blendshapes=True,
        num_faces=1,
    )
    lm = mp_vis.FaceLandmarker.create_from_options(opts)
    out = np.full((len(frames), 52), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        if rotate_iphone:
            f = np.rot90(f, k=-1).copy()
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=f)
        res = lm.detect(mp_img)
        if not res.face_blendshapes:
            continue
        # Index 0 is "_neutral"; blendshapes 1..52 map to ARKit set.
        cats = res.face_blendshapes[0]
        if len(cats) < 53:
            continue
        out[i] = np.array([c.score for c in cats[1:53]], dtype=np.float32)
    lm.close()
    return out


def cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pearson(a, b):
    a = np.asarray(a); b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask] - a[mask].mean(); b = b[mask] - b[mask].mean()
    sa = np.sqrt((a * a).sum()); sb = np.sqrt((b * b).sum())
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float((a * b).sum() / (sa * sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", required=True, help="bridge-rendered mp4")
    ap.add_argument("--anchor", required=True, help="anchor PNG")
    ap.add_argument("--take_dir", required=True,
                    help="LLF take whose b_61 drove the render")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--stride", type=int, default=4,
                    help="ARKit-frame stride used at render time")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_mediapipe", action="store_true")
    args = ap.parse_args()

    rendered = load_video_frames(args.render)
    n = len(rendered)
    print(f"loaded {n} rendered frames", flush=True)
    anchor = load_anchor(args.anchor, rendered[0].shape[0])

    csv = next(Path(args.take_dir).glob("*_iPhone.csv"))
    b_all = load_llf_b61(csv)  # (N, 61)
    arkit_idx = args.start_frame + np.arange(n) * args.stride
    arkit_idx = np.clip(arkit_idx, 0, len(b_all) - 1)
    b_expr = np.concatenate(
        [b_all[arkit_idx, :52], b_all[arkit_idx, 55:61]], axis=1
    ).astype(np.float32)
    b_mean = b_expr.mean(0)
    bnorm = np.linalg.norm(b_expr - b_mean, axis=1)
    # Decompose: 0:52 = ARKit blendshapes (expression), 52:58 = LE+RE eye
    # rotations. Head yaw/pitch/roll come from CSV cols 52..54 directly.
    bnorm_expr = np.linalg.norm(b_expr[:, :52] - b_mean[:52], axis=1)
    bnorm_eye = np.linalg.norm(b_expr[:, 52:58] - b_mean[52:58], axis=1)
    head_ypr = b_all[arkit_idx, 52:55].astype(np.float32)
    head_norm = np.linalg.norm(head_ypr - head_ypr.mean(0), axis=1)

    print("computing LPIPS curve...", flush=True)
    lpips_vals = lpips_curve(rendered, anchor, args.device)
    r_lpips = pearson(lpips_vals, bnorm)
    summary = {
        "n_frames": int(n),
        "lpips": {
            "mean": float(lpips_vals.mean()),
            "std": float(lpips_vals.std()),
            "min": float(lpips_vals.min()),
            "max": float(lpips_vals.max()),
        },
        "bnorm": {
            "mean": float(bnorm.mean()),
            "std": float(bnorm.std()),
            "min": float(bnorm.min()),
            "max": float(bnorm.max()),
        },
        "pearson_lpips_vs_bnorm": r_lpips,
        "pearson_lpips_vs_bnorm_expr": pearson(lpips_vals, bnorm_expr),
        "pearson_lpips_vs_bnorm_eye": pearson(lpips_vals, bnorm_eye),
        "pearson_lpips_vs_head_ypr": pearson(lpips_vals, head_norm),
    }
    print(f"  r(LPIPS, bnorm)      = {summary['pearson_lpips_vs_bnorm']:.3f}", flush=True)
    print(f"  r(LPIPS, bnorm_expr) = {summary['pearson_lpips_vs_bnorm_expr']:.3f}", flush=True)
    print(f"  r(LPIPS, bnorm_eye)  = {summary['pearson_lpips_vs_bnorm_eye']:.3f}", flush=True)
    print(f"  r(LPIPS, head_ypr)   = {summary['pearson_lpips_vs_head_ypr']:.3f}", flush=True)

    # ArcFace identity drift over time.
    print("ArcFace cosine vs anchor...", flush=True)
    arc_cos, arc_self = arcface_drift(rendered, anchor, args.device)
    if arc_cos is not None:
        # Slope vs frame index = drift per frame.
        valid = np.isfinite(arc_cos)
        if valid.sum() >= 3:
            x = np.arange(len(arc_cos))[valid]
            y = arc_cos[valid]
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = float("nan"), float("nan")
        summary["arcface"] = {
            "anchor_self_cos": arc_self,
            "detection_rate": float(valid.mean()),
            "cos_mean": float(np.nanmean(arc_cos)),
            "cos_min": float(np.nanmin(arc_cos)) if valid.any() else None,
            "cos_max": float(np.nanmax(arc_cos)) if valid.any() else None,
            "slope_per_frame": float(slope),
        }
        print(f"  ArcFace cos mean={summary['arcface']['cos_mean']:.3f} "
              f"slope={summary['arcface']['slope_per_frame']:+.5f}/frame", flush=True)

    rows = [{
        "i": int(i),
        "lpips": float(lpips_vals[i]),
        "bnorm": float(bnorm[i]),
        "bnorm_expr": float(bnorm_expr[i]),
        "bnorm_eye": float(bnorm_eye[i]),
        "head_norm": float(head_norm[i]),
        "arcface_cos": (float(arc_cos[i]) if arc_cos is not None
                        and np.isfinite(arc_cos[i]) else None),
    } for i in range(n)]

    if not args.skip_mediapipe:
        print("running mediapipe on rendered + driving frames...", flush=True)
        # Mediapipe needs higher-res driving frames; fetch from MOV.
        import cv2
        mov = next(Path(args.take_dir).glob("*_iPhone.mov"))
        cap = cv2.VideoCapture(str(mov))
        drv_frames = []
        for fi in arkit_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, f = cap.read()
            drv_frames.append(
                cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok
                else np.zeros_like(rendered[0])
            )
        cap.release()

        rendered_bs = mediapipe_blendshapes(rendered, rotate_iphone=False)
        # iPhone MOV frames are sideways (Orientation 4); rotate before
        # landmarking or detection rate is ~0%.
        driving_bs = mediapipe_blendshapes(drv_frames, rotate_iphone=True)
        coses = np.array([
            cosine(rendered_bs[i], driving_bs[i])
            for i in range(n)
            if np.isfinite(rendered_bs[i]).all()
            and np.isfinite(driving_bs[i]).all()
        ])
        cos_arkit = np.array([
            cosine(rendered_bs[i], b_expr[i, :52])
            for i in range(n) if np.isfinite(rendered_bs[i]).all()
        ])
        valid_render = int(np.isfinite(rendered_bs).all(axis=1).sum())
        valid_drv = int(np.isfinite(driving_bs).all(axis=1).sum())
        summary["mediapipe"] = {
            "rendered_detection_rate": valid_render / n,
            "driving_detection_rate": valid_drv / n,
            "cos_render_vs_driving_mediapipe": (
                {"mean": float(coses.mean()), "median": float(np.median(coses))}
                if len(coses) else None
            ),
            "cos_render_vs_driving_arkit_csv": (
                {"mean": float(cos_arkit.mean()),
                 "median": float(np.median(cos_arkit))}
                if len(cos_arkit) else None
            ),
        }
        for i, r in enumerate(rows):
            r["render_bs_valid"] = bool(np.isfinite(rendered_bs[i]).all())
            r["driving_bs_valid"] = bool(np.isfinite(driving_bs[i]).all())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    csv_path = out_path.with_suffix(".perframe.csv")
    with open(csv_path, "w") as f:
        keys = list(rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
