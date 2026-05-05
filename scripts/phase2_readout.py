"""Phase 2 readout: ArcFace identity drift + MediaPipe blendshape correlation.

For each (anchor, clip) MP4, sample frames, score with insightface ArcFace
(buffalo_l, IR101) and MediaPipe FaceLandmarker, and compare against the
matched driver-clip ARKit CSV row by row.

Output:
    output/llf_phase2/readout.parquet  — per-frame (anchor, clip, frame_idx,
                                          arcface_cos, bs_*, det_ok)
    output/llf_phase2/summary.md       — per-clip rollup table
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.demographic_pc.classifiers import InsightFaceClassifier  # noqa: E402
from src.demographic_pc.score_blendshapes import make_landmarker  # noqa: E402

ANCHORS = json.loads((ROOT / "data/llf-phase2/anchors.json").read_text())
CLIPS = json.loads((ROOT / "data/llf-phase2/clips.json").read_text())
GRID = ROOT / "data/llf-phase2"
OUT_DIR = ROOT / "output/llf_phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_STRIDE = 6        # sample every 6th frame → 10 Hz @ 60 fps source
ID_THRESHOLD = 0.40     # ArcFace cosine for "same person" using buffalo_l


def extract_frames(mp4: Path, stride: int) -> list[np.ndarray]:
    """Return BGR frames every `stride`-th from the mp4."""
    cap = cv2.VideoCapture(str(mp4))
    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def anchor_embedding(insf: InsightFaceClassifier, png: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(png))
    if bgr is None:
        return None
    out = insf.predict(bgr)
    return out["embedding"]  # already L2-normed


def landmarker_score_bgr(lm, bgr: np.ndarray) -> dict | None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = lm.detect(mp_image)
    if not res.face_blendshapes:
        return None
    return {b.category_name: float(b.score) for b in res.face_blendshapes[0]}


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


def main() -> None:
    print("[init] loading insightface buffalo_l (with ArcFace r50 embedding)…", flush=True)
    insf = InsightFaceClassifier(ctx_id=0, with_embedding=True)
    print("[init] loading MediaPipe FaceLandmarker…", flush=True)
    lm = make_landmarker()

    # Cache anchor embeddings.
    anchor_emb = {}
    for a in ANCHORS:
        emb = anchor_embedding(insf, Path(a["path"]))
        if emb is None:
            raise RuntimeError(f"insightface failed on anchor {a['name']} ({a['path']})")
        anchor_emb[a["name"]] = emb
        print(f"[anchor] {a['name']:>10s}  emb={emb.shape}  L2={float(np.linalg.norm(emb)):.4f}")

    rows = []
    for ai, a in enumerate(ANCHORS):
        for ci, c in enumerate(CLIPS):
            mp4 = GRID / f"{a['name']}__{c['name']}.mp4"
            if not mp4.exists():
                print(f"[skip] missing {mp4.name}")
                continue
            frames = extract_frames(mp4, FRAME_STRIDE)
            print(f"[scan] [{ai*len(CLIPS)+ci+1:2d}/{len(ANCHORS)*len(CLIPS)}] {mp4.name}  frames={len(frames)}", flush=True)

            # Driver CSV (300 rows per clip, sample at the same stride for alignment).
            driver_csv = Path(c["path"]).with_name("clip.csv")
            df_drv = pd.read_csv(driver_csv)

            for fi, bgr in enumerate(frames):
                src_idx = fi * FRAME_STRIDE
                row: dict = {
                    "anchor": a["name"],
                    "clip": c["name"],
                    "frame_idx": src_idx,
                }
                # ArcFace
                ins = insf.predict(bgr)
                row["det_ok"] = ins["detected"]
                row["arcface_cos"] = cosine(anchor_emb[a["name"]], ins["embedding"])
                # MediaPipe blendshapes
                bs = landmarker_score_bgr(lm, bgr)
                row["bs_ok"] = bs is not None
                if bs is not None:
                    for k, v in bs.items():
                        row[f"out_{k}"] = v
                # Driver ARKit row at same index
                if src_idx < len(df_drv):
                    drv_row = df_drv.iloc[src_idx]
                    for col in df_drv.columns:
                        if col in ("Timecode", "BlendshapeCount"):
                            continue
                        row[f"drv_{col}"] = drv_row[col]
                rows.append(row)

    out_pq = OUT_DIR / "readout.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(out_pq)
    print(f"\n[write] {out_pq.relative_to(ROOT)}  rows={len(df)}")

    # ── per-clip rollup ──
    BS_PAIRS = [
        # (mediapipe_name, arkit_name) — case differs.
        ("mouthSmileLeft",  "MouthSmileLeft"),
        ("mouthSmileRight", "MouthSmileRight"),
        ("jawOpen",         "JawOpen"),
        ("eyeBlinkLeft",    "EyeBlinkLeft"),
        ("eyeBlinkRight",   "EyeBlinkRight"),
        ("browInnerUp",     "BrowInnerUp"),
        ("browDownLeft",    "BrowDownLeft"),
        ("browDownRight",   "BrowDownRight"),
        ("eyeSquintLeft",   "EyeSquintLeft"),
        ("eyeSquintRight",  "EyeSquintRight"),
        ("mouthFrownLeft",  "MouthFrownLeft"),
        ("mouthFrownRight", "MouthFrownRight"),
    ]

    summary = []
    for (anchor, clip), g in df.groupby(["anchor", "clip"]):
        row = {
            "anchor": anchor,
            "clip": clip,
            "n": len(g),
            "id_mean": float(g["arcface_cos"].mean()),
            "id_min":  float(g["arcface_cos"].min()),
            "id_below_thr": float((g["arcface_cos"] < ID_THRESHOLD).mean()),
            "det_rate": float(g["det_ok"].mean()),
            "bs_rate":  float(g["bs_ok"].mean()),
        }
        # Macro per-channel correlation
        corrs = []
        for mp_name, ark_name in BS_PAIRS:
            oc, dc = f"out_{mp_name}", f"drv_{ark_name}"
            if oc not in g.columns or dc not in g.columns:
                continue
            sub = g[[oc, dc]].dropna()
            if len(sub) < 4 or sub[oc].std() < 1e-6 or sub[dc].std() < 1e-6:
                continue
            corrs.append(float(sub[oc].corr(sub[dc])))
        row["bs_macro_corr"] = float(np.mean(corrs)) if corrs else float("nan")
        summary.append(row)
    df_sum = pd.DataFrame(summary).sort_values(["anchor", "clip"])

    md = ["# Phase 2 readout summary", "",
          f"frames sampled: stride={FRAME_STRIDE} (10 Hz from 60fps)  |  ArcFace=buffalo_l  |  ID threshold τ={ID_THRESHOLD}",
          "",
          df_sum.to_markdown(index=False, floatfmt=".3f")]
    summary_md = OUT_DIR / "summary.md"
    summary_md.write_text("\n".join(md) + "\n")
    print(f"[write] {summary_md.relative_to(ROOT)}")
    print()
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
