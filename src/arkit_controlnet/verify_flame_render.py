"""Eyeball check: render FFHQ rows' FLAME meshes and overlay on their photos.

Picks 8 FFHQ rows spanning low-to-high expression energy, renders each row's
FLAME mesh from its blendshapes + cached pose, alpha-overlays it on the photo,
and writes a collage. Run:
    PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.verify_flame_render
"""
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from arkit_controlnet.build_ffhq_index import SHARD_GLOB
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES
from arkit_controlnet.flame_render import deform, mediapipe_to_basis_vector, render

OUT_DIR = Path("exp_output/flame_render_check")
_EXPR_COLS = [f"bs_{n}" for n in ARKIT_BLENDSHAPE_NAMES if n != "_neutral"]


def main() -> None:
    import glob
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ri = pd.read_parquet("output/reverse_index/reverse_index.parquet",
                         columns=["image_sha256", "source", "bs_detected",
                                  *_EXPR_COLS])
    ri = ri[(ri["source"] == "ffhq") & (ri["bs_detected"])].copy()
    ri["energy"] = ri[_EXPR_COLS].abs().sum(axis=1)
    ri = ri.sort_values("energy")
    picks = pd.concat([ri.head(4), ri.tail(4)])      # 4 calm, 4 expressive

    idx = pd.read_parquet("output/ffhq_index/ffhq_sha_index.parquet")
    pc = pd.read_parquet("output/flame_pose_cache/pose_cache.parquet")
    shards = sorted(glob.glob(SHARD_GLOB))

    tiles = []
    for _, row in picks.iterrows():
        sha = row["image_sha256"]
        loc = idx[idx["image_sha256"] == sha].iloc[0]
        cell = pd.read_parquet(shards[loc["shard_idx"]],
                               columns=["image"])["image"].iloc[loc["row_idx"]]
        photo = np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
        H, W = photo.shape[:2]

        pose = pc[pc["image_sha256"] == sha].iloc[0]
        rot = np.array(pose["rotation"], dtype=np.float64).reshape(3, 3)
        bbox = (pose["bbox_cx"], pose["bbox_cy"], pose["bbox_w"], pose["bbox_h"])

        mp_bs = {n: float(row.get(f"bs_{n}", 0.0)) for n in ARKIT_BLENDSHAPE_NAMES}
        verts = deform(mediapipe_to_basis_vector(mp_bs))
        ctrl = render(verts, rot, bbox, modality="normals", H=H, W=W)

        mask = (ctrl.sum(axis=2) > 10)[:, :, None]
        overlay = np.where(mask, (0.5 * photo + 0.5 * ctrl).astype(np.uint8),
                           photo)
        tiles.append(np.hstack([photo, ctrl, overlay]))

    collage = np.vstack(tiles)
    cv2.imwrite(str(OUT_DIR / "collage.png"),
                cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))
    print(f"wrote {OUT_DIR/'collage.png'}")


if __name__ == "__main__":
    main()
