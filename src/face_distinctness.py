#!/usr/bin/env python3
"""
face_distinctness.py — Quantify visual distinctness of generated faces.

Uses CLIP (openai/clip-vit-base-patch32) to embed each generated PNG,
then computes intra-cluster vs inter-cluster cosine distances.

Separation score = (avg_inter_dist - avg_intra_dist) / avg_inter_dist
  → 0: no visual separation between clusters
  → 1: perfect separation (intra_dist ≈ 0)
  → negative: clusters are more similar than random pairs (bad mapping)

Usage:
    uv run src/face_distinctness.py output/phase1
    uv run src/face_distinctness.py output/phase2
    uv run src/face_distinctness.py output/phase1 output/phase2  # compare
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_clip():
    """Load CLIP model. Downloads on first run (~600 MB)."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    model_id = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP ({model_id})…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    print(f"  CLIP loaded on {device}")
    return model, processor, device


def embed_images(paths: list[Path], model, processor, device: str, batch_size: int = 16) -> np.ndarray:
    """Return (N, 512) float32 CLIP image embeddings, L2-normalised."""
    import torch

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            feats = model.get_image_features(**inputs)  # returns Tensor
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)  # L2 normalise
            all_embs.append(feats.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def pairwise_sims(embs: np.ndarray) -> list[float]:
    """All unique pairwise cosine sims for a set of embeddings."""
    n = len(embs)
    return [cosine_sim(embs[i], embs[j]) for i in range(n) for j in range(i + 1, n)]


def analyse_phase(phase_dir: Path, model, processor, device: str) -> dict:
    """Load all PNGs from cluster subdirs, compute distinctness metrics."""
    phase_dir = phase_dir.resolve()
    cluster_dirs = sorted([d for d in phase_dir.iterdir() if d.is_dir()])

    print(f"\n{'='*60}")
    print(f"Phase dir: {phase_dir}")
    print(f"Clusters: {[d.name for d in cluster_dirs]}")

    # Load images per cluster
    cluster_data: dict[str, dict] = {}
    for cd in cluster_dirs:
        pngs = sorted(cd.glob("*.png"))
        if not pngs:
            continue
        print(f"\n  Embedding {len(pngs)} images from {cd.name}…")
        embs = embed_images(pngs, model, processor, device)
        cluster_data[cd.name] = {"paths": pngs, "embs": embs}

    if len(cluster_data) < 2:
        print("Need at least 2 clusters to compute separation.")
        return {}

    labels = list(cluster_data.keys())

    # Intra-cluster similarities
    print("\n  Intra-cluster cosine similarity (CLIP):")
    intra_sims = []
    for label, data in cluster_data.items():
        sims = pairwise_sims(data["embs"])
        avg = float(np.mean(sims)) if sims else float("nan")
        std = float(np.std(sims)) if sims else float("nan")
        intra_sims.extend(sims)
        print(f"    {label:25s}: mean={avg:.4f}  std={std:.4f}  n_pairs={len(sims)}")

    # Inter-cluster similarities
    print("\n  Inter-cluster cosine similarity (CLIP):")
    inter_sims = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            ea = cluster_data[la]["embs"]
            eb = cluster_data[lb]["embs"]
            sims = [cosine_sim(ea[r], eb[c]) for r in range(len(ea)) for c in range(len(eb))]
            avg = float(np.mean(sims))
            inter_sims.extend(sims)
            print(f"    {la:20s} ↔ {lb:20s}: {avg:.4f}")

    # Convert sims to distances (1 - sim)
    intra_dist = float(np.mean([1 - s for s in intra_sims]))
    inter_dist = float(np.mean([1 - s for s in inter_sims]))

    if inter_dist > 0:
        separation = (inter_dist - intra_dist) / inter_dist
    else:
        separation = 0.0

    avg_intra_sim = float(np.mean(intra_sims))
    avg_inter_sim = float(np.mean(inter_sims))

    print(f"\n  Summary:")
    print(f"    Avg intra-cluster sim: {avg_intra_sim:.4f}  (distance: {intra_dist:.4f})")
    print(f"    Avg inter-cluster sim: {avg_inter_sim:.4f}  (distance: {inter_dist:.4f})")
    print(f"    Separation score:      {separation:.4f}  {'(positive = clusters are visually distinct)' if separation > 0 else '(negative = no useful separation)'}")

    return {
        "phase_dir": str(phase_dir),
        "clusters": labels,
        "avg_intra_sim": avg_intra_sim,
        "avg_inter_sim": avg_inter_sim,
        "intra_dist": intra_dist,
        "inter_dist": inter_dist,
        "separation_score": separation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase_dirs", nargs="+", type=Path, help="Phase output directories")
    args = parser.parse_args()

    model, processor, device = load_clip()

    results = []
    for d in args.phase_dirs:
        if not d.is_dir():
            print(f"WARNING: {d} is not a directory, skipping.")
            continue
        r = analyse_phase(d, model, processor, device)
        if r:
            results.append(r)

    if len(results) >= 2:
        print(f"\n{'='*60}")
        print("Comparison across phases:")
        for r in results:
            name = Path(r["phase_dir"]).name
            print(f"  {name}: separation={r['separation_score']:.4f}  "
                  f"intra_sim={r['avg_intra_sim']:.4f}  inter_sim={r['avg_inter_sim']:.4f}")
        best = max(results, key=lambda x: x["separation_score"])
        print(f"\n  Best separation: {Path(best['phase_dir']).name} "
              f"(score={best['separation_score']:.4f})")


if __name__ == "__main__":
    main()
