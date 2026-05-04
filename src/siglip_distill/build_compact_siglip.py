"""Build compact_siglip.pt — (latent ↔ siglip_emb) pair file.

Mirrors compact_blendshapes.pt format. The latent file `arc_full_latent/compact.pt`
holds 26,108 (16, 64, 64) Flux VAE latents keyed by SHA (a strict subset of the
ffhq slice in reverse_index). This script aligns the SigLIP image embedding
column from reverse_index to those same 26,108 SHAs in the same order, so that
the distill trainer can pair latent[i] with embedding[i] without re-joining.

Output schema:
    embeddings:    Tensor (26108, 1152) fp16, L2-normed (matches teacher output)
    detected:      Tensor (26108,) bool   — True if the row had a SigLIP emb
    shas:          list[26108] str
    emb_dim:       int
    teacher_model: str
    format_version: int
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TEACHER_MODEL = "google/siglip2-so400m-patch16-384"
EMB_DIM = 1152
FORMAT_VERSION = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shas-from", type=Path,
                    default=Path("output/mediapipe_distill/compact_blendshapes.pt"),
                    help="Source of the 26,108 SHA order (any compact_*.pt file from mediapipe_distill works)")
    ap.add_argument("--reverse-index", type=Path,
                    default=Path("output/reverse_index/reverse_index.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("output/siglip_distill/compact_siglip.pt"))
    args = ap.parse_args()

    print(f"[load] {args.shas_from}")
    src = torch.load(args.shas_from, weights_only=False)
    shas: list[str] = list(src["shas"])
    n = len(shas)
    print(f"  shas: {n}")

    print(f"[load] {args.reverse_index} (siglip column only)")
    df = pd.read_parquet(args.reverse_index,
                         columns=["image_sha256", "siglip_img_emb_fp16"])
    sha_to_emb = dict(zip(df["image_sha256"], df["siglip_img_emb_fp16"]))
    print(f"  reverse_index rows with siglip_img_emb_fp16: "
          f"{df['siglip_img_emb_fp16'].notna().sum()}/{len(df)}")

    print(f"[align] looking up {n} SHAs")
    embeddings = np.zeros((n, EMB_DIM), dtype=np.float16)
    detected = np.zeros(n, dtype=bool)
    n_missing = 0
    for i, sha in enumerate(shas):
        emb = sha_to_emb.get(sha)
        if emb is None:
            n_missing += 1
            continue
        v = np.asarray(emb, dtype=np.float16)
        if v.shape != (EMB_DIM,):
            raise RuntimeError(f"row {i} sha={sha[:16]}... unexpected shape {v.shape}")
        embeddings[i] = v
        detected[i] = True
    print(f"  matched={detected.sum()}/{n}, missing={n_missing}")

    if detected.sum() > 0:
        # Sanity: confirm L2 norms intact post-roundtrip
        idx = np.where(detected)[0][:5]
        for i in idx:
            v = embeddings[i].astype(np.float32)
            print(f"  [{i}] sha={shas[i][:16]}... L2={np.linalg.norm(v):.4f}")

    payload = {
        "embeddings": torch.from_numpy(embeddings),
        "detected": torch.from_numpy(detected),
        "shas": shas,
        "emb_dim": EMB_DIM,
        "teacher_model": TEACHER_MODEL,
        "format_version": FORMAT_VERSION,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"[write] {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
