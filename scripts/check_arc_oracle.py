"""Check 3: feed aligned 112 crop directly to buffalo_l R50, compare to stored teacher.

If we reproduce the teacher embedding from our aligned crop, the input is correct
and the pixel-distillation failure is capacity/budget, not pipeline mismatch.

Run on Windows where the encoded .pt + parquet shards live.
"""
from __future__ import annotations
import io
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
import pyarrow.parquet as pq
from insightface.app import FaceAnalysis
from insightface.utils import face_align

SHARD_PARQUET = Path(r"C:\arc_distill\ffhq_parquet\data\train-00000-of-00190.parquet")
SHARD_PT = Path(r"C:\arc_distill\encoded\train-00000-of-00190.pt")


def main():
    pt = torch.load(SHARD_PT, map_location="cpu", weights_only=False)
    teacher_arc = pt["arcface_fp32"]
    detected = pt["detected"].tolist()
    img_col = pq.read_table(SHARD_PARQUET, columns=["image"]).column("image").to_pylist()

    det_idxs = [i for i, d in enumerate(detected) if d][:5]
    print(f"checking rows {det_idxs}")

    app_full = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app_full.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

    app_det = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app_det.prepare(ctx_id=0, det_size=(640, 640))

    rec = app_full.models["recognition"]
    print(f"rec: {type(rec).__name__} input_size={rec.input_size} "
          f"input_mean={rec.input_mean} input_std={rec.input_std}")

    for i in det_idxs:
        rgb = np.asarray(Image.open(io.BytesIO(img_col[i]["bytes"])).convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        faces_full = app_full.get(bgr)
        if not faces_full:
            print(f"row {i}: full pipeline missed!"); continue
        f_full = max(faces_full, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        repro_emb = np.asarray(f_full.normed_embedding, dtype=np.float32)

        faces_det = app_det.get(bgr)
        if not faces_det:
            print(f"row {i}: det-only missed!"); continue
        f_det = max(faces_det, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        aligned_bgr = face_align.norm_crop(bgr, f_det.kps, image_size=112)

        oracle_emb = rec.get_feat(aligned_bgr).flatten().astype(np.float32)
        oracle_emb = oracle_emb / (np.linalg.norm(oracle_emb) + 1e-12)

        teacher = teacher_arc[i].numpy()
        cos_repro = float((teacher * repro_emb).sum())
        cos_oracle = float((teacher * oracle_emb).sum())
        cos_repro_oracle = float((repro_emb * oracle_emb).sum())

        bbox_match = np.allclose(f_full.bbox, f_det.bbox, atol=1e-3)
        kps_match = np.allclose(f_full.kps, f_det.kps, atol=1e-3)

        print(f"row {i}: cos(teacher,repro)={cos_repro:.4f}  "
              f"cos(teacher,oracle_aligned)={cos_oracle:.4f}  "
              f"cos(repro,oracle)={cos_repro_oracle:.4f}  "
              f"bbox_eq={bbox_match} kps_eq={kps_match}")


if __name__ == "__main__":
    main()
