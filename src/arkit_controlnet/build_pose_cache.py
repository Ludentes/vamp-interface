"""Re-run MediaPipe over FFHQ to cache each image's head pose.

The CFM conditioning render must be posed to match the photo; reverse_index
stores blendshapes but no head pose. This caches the rotation + face bbox per
image_sha256. Resumable per shard. Run under the miniconda python:
    PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.build_pose_cache
"""
import glob
import io
import os
from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image

from arkit_controlnet.build_ffhq_index import SHARD_GLOB

_MP_MODEL = Path("models/mediapipe/face_landmarker.task")
INDEX = Path("output/ffhq_index/ffhq_sha_index.parquet")
OUT = Path("output/flame_pose_cache/pose_cache.parquet")

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        if not _MP_MODEL.exists():
            raise FileNotFoundError(
                f"{_MP_MODEL} missing — download the MediaPipe face_landmarker "
                f"model to that path")
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_MP_MODEL)),
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


def pose_from_image(
    rgb: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray, bool]:
    """(rotation 3x3, bbox (cx,cy,w,h) normalized, landmarks (478,2) normalized,
    detected bool) for an RGB array.

    Landmarks are MediaPipe's 478 face landmarks in image-normalized [0,1] xy.
    Non-detection returns (eye(3), (0,0,0,0), zeros((478,2)), False)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb))
    res = _get_landmarker().detect(mp_image)
    if not res.facial_transformation_matrixes or not res.face_landmarks:
        return np.eye(3), (0.0, 0.0, 0.0, 0.0), np.zeros((478, 2)), False
    mat = np.asarray(res.facial_transformation_matrixes[0], dtype=np.float64)
    rot = mat[:3, :3]
    lm = np.array([[p.x, p.y] for p in res.face_landmarks[0]], dtype=np.float64)
    lo, hi = lm.min(axis=0), lm.max(axis=0)
    cx, cy = (lo + hi) / 2
    bw, bh = hi - lo
    return rot, (float(cx), float(cy), float(bw), float(bh)), lm, True


def _write_atomic(df: pd.DataFrame) -> None:
    """Write the cache parquet atomically (USB-drive disconnect hazard)."""
    tmp = OUT.with_name(OUT.name + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, OUT)


def build() -> None:
    if not INDEX.exists():
        raise FileNotFoundError(
            f"{INDEX} missing — run `python -m arkit_controlnet.build_ffhq_index`")
    idx = pd.read_parquet(INDEX)
    shards = sorted(glob.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(
            f"no FFHQ shards at {SHARD_GLOB} — is the Seagate drive mounted?")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Read the existing cache once into an in-memory accumulator; never re-read
    # OUT inside the loop (re-read + re-concat per shard is O(n^2) over shards).
    all_rows: list[dict] = []
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        if "landmarks_xy" not in existing.columns:
            raise RuntimeError(
                f"{OUT} predates the landmarks_xy schema; rename or delete it "
                "and re-run to regenerate with the new column")
        all_rows = existing.to_dict("records")
    done: set[str] = {row["image_sha256"] for row in all_rows}

    for shard_idx, group in idx.groupby("shard_idx"):
        todo = group[~group["image_sha256"].isin(done)]
        if todo.empty:
            continue
        # shard_idx indexes into the same sorted(glob.glob(SHARD_GLOB)) ordering
        # that build_ffhq_index.py used to assign it.
        df = pd.read_parquet(shards[shard_idx], columns=["image"])
        for _, r in todo.iterrows():
            cell = df["image"].iloc[r["row_idx"]]
            rgb = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
            rot, bbox, lm, detected = pose_from_image(rgb)
            all_rows.append({
                "image_sha256": r["image_sha256"],
                "rotation": rot.ravel().tolist(),
                "bbox_cx": bbox[0], "bbox_cy": bbox[1],
                "bbox_w": bbox[2], "bbox_h": bbox[3],
                "landmarks_xy": lm.ravel().tolist(),  # (478, 2) flat row-major
                "pose_detected": detected,
            })
        # flush the full accumulator after every shard so a crash loses at most
        # one shard
        _write_atomic(pd.DataFrame(all_rows))
        done.update(todo["image_sha256"])
        print(f"shard {shard_idx} done ({len(todo)} images); "
              f"cache now {len(all_rows)} rows")

    if not all_rows:
        print("pose cache: nothing to do")
        return
    final = pd.DataFrame(all_rows)
    print(f"pose cache complete: {len(final)} rows; detected "
          f"{int(final['pose_detected'].sum())}/{len(final)}")


if __name__ == "__main__":
    build()
