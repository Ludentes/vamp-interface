"""Run all reverse-index metric families on a FFHQ parquet corpus.

Single-pass per shard: image bytes -> RGB@512 -> {mivolo, fairface,
insightface attrs, siglip2 probes, mediapipe blendshapes -> NMF atoms}.

The five model wrappers are imported lazily inside `main()` so that the
pure helpers below remain testable on a CPU-only laptop.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image
from scipy.optimize import nnls


def compute_image_sha256(png_bytes: bytes) -> str:
    """Hex-digest of the raw PNG bytes. Stable across decoders."""
    return hashlib.sha256(png_bytes).hexdigest()


def decode_and_resize(png_bytes: bytes, resolution: int) -> np.ndarray:
    """Decode PNG, convert to RGB, resize to (resolution, resolution) using
    LANCZOS (matches the encode_ffhq.py + Flux corpus convention).
    Returns uint8 (H, W, 3)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def project_blendshapes_to_atoms(y: np.ndarray, H: np.ndarray) -> np.ndarray:
    """NNLS projection: find non-negative w (k,) minimising ||y - w @ H||^2.

    Args:
        y: (52,) blendshape activations in [0, 1].
        H: (k, 52) NMF atom-pattern matrix from au_library.npz.

    Returns:
        (k,) atom loadings (>= 0).
    """
    w, _ = nnls(H.T, y)
    return w.astype(np.float32)


# ── Heavy pipeline (everything below this line requires GPU + weights) ────────

ARKIT_BLENDSHAPE_NAMES: list[str] = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

FORMAT_VERSION = 1


def build_landmarker(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts), mp


def score_blendshapes(landmarker, mp, rgb):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(img)
    if not res.face_blendshapes:
        return None
    return {c.category_name: float(c.score) for c in res.face_blendshapes[0]}


def encode_siglip_probes(siglip):
    return siglip.encode_probes()


def score_siglip_image(siglip, probe_feats, rgb):
    import torch
    img = Image.fromarray(rgb)
    inputs = siglip.processor(images=img, return_tensors="pt").to(siglip.device)
    with torch.no_grad():
        feat = siglip._as_tensor(siglip.model.get_image_features(**inputs))
    feat = feat / feat.norm(dim=-1, keepdim=True)
    out: dict[str, float] = {}
    for name, pf in probe_feats.items():
        sims = (feat @ pf.T).squeeze(0)
        out[f"sg_{name}_margin"] = float(sims[0] - sims[1])
    return out


def process_shard(shard_path, out_path, *, mv, ff, ins, siglip, probe_feats,
                  landmarker, mp, H, resolution, log):
    import os
    import time
    import cv2
    import pyarrow.parquet as pq
    import torch

    t0 = time.time()
    table = pq.read_table(shard_path, columns=["image"])
    rows = table.column("image").to_pylist()
    n = len(rows)
    log.info(f"  {shard_path.name}: {n} rows")

    payload: dict = {
        "image_sha256": [],
        "mv_age":          np.full(n, np.nan, dtype=np.float32),
        "mv_gender":       np.full(n, "", dtype=object),
        "mv_gender_conf":  np.full(n, np.nan, dtype=np.float32),
        "ff_detected":     np.zeros(n, dtype=bool),
        "ff_age_bin":      np.full(n, "", dtype=object),
        "ff_gender":       np.full(n, "", dtype=object),
        "ff_race":         np.full(n, "", dtype=object),
        "ff_age_probs":    np.zeros((n, 9), dtype=np.float32),
        "ff_gender_probs": np.zeros((n, 2), dtype=np.float32),
        "ff_race_probs":   np.zeros((n, 7), dtype=np.float32),
        "ins_detected":    np.zeros(n, dtype=bool),
        "ins_age":         np.full(n, np.nan, dtype=np.float32),
        "ins_gender":      np.full(n, "", dtype=object),
        "bs_detected":     np.zeros(n, dtype=bool),
    }
    for name in ARKIT_BLENDSHAPE_NAMES:
        payload[f"bs_{name}"] = np.zeros(n, dtype=np.float32)
    for k in range(H.shape[0]):
        payload[f"atom_{k:02d}"] = np.full(n, np.nan, dtype=np.float32)
    # Probe names extracted from siglip's pre-encoded probes dict.
    probe_names = list(probe_feats.keys())
    for probe_name in probe_names:
        payload[f"sg_{probe_name}_margin"] = np.zeros(n, dtype=np.float32)

    for i, row in enumerate(rows):
        png_bytes = row["bytes"]
        payload["image_sha256"].append(compute_image_sha256(png_bytes))
        rgb = decode_and_resize(png_bytes, resolution)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        mv_out = mv.predict(bgr)
        payload["mv_age"][i] = mv_out["age"]
        payload["mv_gender"][i] = mv_out["gender"]
        payload["mv_gender_conf"][i] = mv_out["gender_conf"]

        ff_out = ff.predict(bgr)
        payload["ff_detected"][i] = ff_out["detected"]
        if ff_out["detected"]:
            payload["ff_age_bin"][i] = ff_out["age_bin"]
            payload["ff_gender"][i] = ff_out["gender"]
            payload["ff_race"][i] = ff_out["race"]
            payload["ff_age_probs"][i] = ff_out["age_probs"]
            payload["ff_gender_probs"][i] = ff_out["gender_probs"]
            payload["ff_race_probs"][i] = ff_out["race_probs"]

        ins_out = ins.predict(bgr)
        payload["ins_detected"][i] = ins_out["detected"]
        if ins_out["detected"]:
            payload["ins_age"][i] = ins_out["age"]
            payload["ins_gender"][i] = ins_out["gender"]

        sg_out = score_siglip_image(siglip, probe_feats, rgb)
        for k, v in sg_out.items():
            payload[k][i] = v

        bs = score_blendshapes(landmarker, mp, rgb)
        if bs is not None:
            payload["bs_detected"][i] = True
            y = np.zeros(52, dtype=np.float32)
            for j, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
                v = float(bs.get(name, 0.0))
                payload[f"bs_{name}"][i] = v
                y[j] = v
            atoms = project_blendshapes_to_atoms(y, H)
            for k_idx in range(H.shape[0]):
                payload[f"atom_{k_idx:02d}"][i] = atoms[k_idx]

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            log.info(
                f"    {i+1}/{n} ({(i+1)/elapsed:.1f} img/s, "
                f"ff_det={payload['ff_detected'][:i+1].mean():.2f}, "
                f"bs_det={payload['bs_detected'][:i+1].mean():.2f})"
            )

    payload["shard_name"] = shard_path.name
    payload["format_version"] = FORMAT_VERSION
    payload["resolution"] = resolution
    elapsed = time.time() - t0
    log.info(
        f"  {shard_path.name} done: {n} imgs in {elapsed:.1f}s "
        f"({n/elapsed:.1f} img/s); "
        f"ff_det={payload['ff_detected'].mean():.3f} "
        f"bs_det={payload['bs_detected'].mean():.3f}"
    )

    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    log.info(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return {
        "shard": shard_path.name, "n": n, "elapsed_s": elapsed,
        "ff_det": float(payload["ff_detected"].mean()),
        "bs_det": float(payload["bs_detected"].mean()),
    }


def main() -> None:
    import argparse
    import json
    import logging
    import os
    import sys
    from pathlib import Path

    import torch

    from demographic_pc.classifiers import (
        MiVOLOClassifier, FairFaceClassifier, InsightFaceClassifier,
    )
    from demographic_pc.score_clip_probes import Siglip2Backend

    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--mediapipe-model", type=Path, required=True)
    p.add_argument("--au-library", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--limit-shards", type=int, default=0, help="0 = all")
    p.add_argument("--smoke", action="store_true",
                   help="if set, --shards-dir is ignored and we run on a single fixture parquet")
    p.add_argument("--smoke-fixture", type=Path,
                   default=Path("tests/fixtures/ffhq_smoke.parquet"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(args.log, mode="a"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("extract_ffhq")
    log.info("=" * 60)
    log.info(f"start extract_ffhq pid={os.getpid()}")

    H = np.load(args.au_library)["H"].astype(np.float32)
    log.info(f"loaded au_library H={H.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")

    log.info("loading MiVOLO...");        mv = MiVOLOClassifier(device=device)
    log.info("loading FairFace...");      ff = FairFaceClassifier(device=device)
    log.info("loading InsightFace...");   ins = InsightFaceClassifier(ctx_id=0, with_embedding=False)
    log.info("loading SigLIP-2...");      siglip = Siglip2Backend(device=device)
    log.info("loading FaceLandmarker..."); landmarker, mp = build_landmarker(args.mediapipe_model)

    log.info("encoding SigLIP-2 probes...")
    probe_feats = encode_siglip_probes(siglip)

    if args.smoke:
        shards = [args.smoke_fixture]
    else:
        shards = sorted(args.shards_dir.glob("*.parquet"))
        if args.limit_shards:
            shards = shards[: args.limit_shards]
    log.info(f"processing {len(shards)} shard(s)")

    summary = []
    for idx, shard in enumerate(shards):
        out_path = args.out_dir / f"{shard.stem}.pt"
        if out_path.exists():
            log.info(f"[{idx+1}/{len(shards)}] {shard.name} -> SKIP (exists)")
            continue
        log.info(f"[{idx+1}/{len(shards)}] processing {shard.name}")
        try:
            stats = process_shard(
                shard, out_path,
                mv=mv, ff=ff, ins=ins,
                siglip=siglip, probe_feats=probe_feats,
                landmarker=landmarker, mp=mp,
                H=H, resolution=args.resolution, log=log,
            )
            summary.append(stats)
        except Exception:
            log.exception(f"FAILED on {shard.name}")
            raise

    summary_path = args.out_dir / "extract_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"all done, wrote {summary_path}")


if __name__ == "__main__":
    main()
