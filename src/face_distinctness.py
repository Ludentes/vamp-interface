#!/usr/bin/env python3
"""
face_distinctness.py — Quantify visual distinctness and encoding validity of generated faces.

Default model: FaceNet (InceptionResnetV1, VGGFace2 pretrained).
  - Identity fingerprints: style-agnostic, works on SDXL and Flux equally
  - Anchor distance = identity drift from neutral baseline = uncanniness proxy
  - r(anchor_dist, sus_level) = encoding validation

Fallback: --model clip (CLIP ViT-B/32, original behaviour, SDXL-biased)

Metrics:
  1. Cluster separation     — intra vs inter cohort identity distance
  2. Anchor distance        — dist(face_i, anchor) per face, per cohort
  3. Sus correlation        — Pearson r(anchor_dist, sus_level) per cohort
  4. Cross-model fairness   — run on SDXL and Flux dirs, compare side-by-side

Usage:
    uv run src/face_distinctness.py output/dataset_faces
    uv run src/face_distinctness.py output/dataset_faces output/dataset_faces_flux
    uv run src/face_distinctness.py --model clip output/dataset_faces
    uv run src/face_distinctness.py output/dataset_faces output/dataset_faces_v2 output/dataset_faces_v3 output/dataset_faces_flux
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
from PIL import Image

ANCHOR_PATH = Path("output/phase1/phase1_anchor.png")
ARCFACE_SIZE = 112   # CVLFace ArcFace IR101 requires 112×112
ARCFACE_HF   = "minchul/cvlface_arcface_ir101_webface4m"


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_arcface():
    """Load ArcFace IR101/WebFace4M via CVLFace HuggingFace hub (~170 MB on first run).
    Pure PyTorch, no C extensions, works with any modern torch version.

    CVLFace uses relative paths for weights and imports a bundled `models` package,
    so we temporarily add the snapshot dir to sys.path and chdir into it during load.
    """
    import os
    import sys
    import torch
    from huggingface_hub import snapshot_download

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ArcFace IR101 (CVLFace/{ARCFACE_HF})...")
    local_path = snapshot_download(ARCFACE_HF)

    # CVLFace uses relative paths for weights and a bundled `models` package.
    # Bypass AutoModel.from_pretrained (incompatible with transformers>=5.5) and
    # instantiate the wrapper directly after chdir + sys.path patching.
    old_cwd = os.getcwd()
    sys.path.insert(0, local_path)
    os.chdir(local_path)
    try:
        from wrapper import ModelConfig, CVLFaceRecognitionModel  # type: ignore
        config = ModelConfig()
        model = CVLFaceRecognitionModel(config).eval().to(device)
    finally:
        os.chdir(old_cwd)
        if local_path in sys.path:
            sys.path.remove(local_path)

    print(f"  ArcFace loaded on {device}")
    return model, None, device


def load_clip():
    """Load CLIP ViT-B/32 (~600 MB on first run)."""
    import torch
    from transformers import CLIPProcessor, CLIPModel

    model_id = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP ({model_id})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    print(f"  CLIP loaded on {device}")
    return model, processor, device


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_for_arcface(paths: list[Path], device: str):
    """Load images → (N, 3, 112, 112) tensor normalised to [-1, 1]."""
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((ARCFACE_SIZE, ARCFACE_SIZE)),
        transforms.ToTensor(),                         # [0,1], (3,H,W)
        transforms.Normalize([0.5, 0.5, 0.5],         # → [-1, 1]
                             [0.5, 0.5, 0.5]),
    ])

    tensors = [transform(Image.open(p).convert("RGB")) for p in paths]
    return torch.stack(tensors).to(device)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_images(
    paths: list[Path],
    model,
    processor,
    device: str,
    batch_size: int = 16,
    model_name: str = "arcface",
) -> np.ndarray:
    """Return (N, 512) float32 L2-normalised embeddings."""
    import torch

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = paths[i : i + batch_size]

            if model_name == "arcface":
                inputs = _preprocess_for_arcface(batch, device)
                feats = model(inputs)  # CVLFace returns (B, 512), L2-normalise
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            else:  # clip
                images = [Image.open(p).convert("RGB") for p in batch]
                inp = processor(images=images, return_tensors="pt", padding=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                out = model.get_image_features(pixel_values=inp["pixel_values"])
                feats = out if isinstance(out, torch.Tensor) else out.pooler_output
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

            all_embs.append(feats.cpu().numpy())

    return np.vstack(all_embs).astype(np.float32)


def embed_anchor(anchor_path: Path, model, processor, device: str, model_name: str):
    """Embed anchor face → (512,) float32, or None if not found."""
    if not anchor_path.exists():
        print(f"WARNING: anchor not found at {anchor_path}, skipping anchor metrics.")
        return None
    print(f"\nEmbedding anchor: {anchor_path}")
    emb = embed_images([anchor_path], model, processor, device, model_name=model_name)
    return emb[0]


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest(phase_dir: Path) -> dict:
    """Load manifest.json → {job_id: job_record}. Empty dict if not present."""
    manifest_path = phase_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return {j["job_id"]: j for j in data["jobs"] if j["status"] == "generated"}


# ── Distance helpers ──────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def pairwise_sims(embs: np.ndarray) -> list[float]:
    n = len(embs)
    return [cosine_sim(embs[i], embs[j]) for i in range(n) for j in range(i + 1, n)]


def anchor_distances(embs: np.ndarray, anchor_emb: np.ndarray) -> np.ndarray:
    """L2 distance from each embedding to anchor (unit-norm: sqrt(2*(1-cosine)))."""
    cos = embs @ anchor_emb  # (N,)
    return np.sqrt(np.clip(2.0 * (1.0 - cos), 0.0, None)).astype(np.float32)


def sus_correlation(dists: np.ndarray, sus_levels: list[int]):
    """Pearson r(anchor_dist, sus_level). None if <3 samples or constant input."""
    if len(dists) < 3 or not sus_levels:
        return None
    sus_arr = np.array(sus_levels, dtype=np.float32)
    if sus_arr.std() < 1e-6 or dists.std() < 1e-6:
        return None
    return float(np.corrcoef(dists, sus_arr)[0, 1])


# ── Core analysis ─────────────────────────────────────────────────────────────

def analyse_phase(
    phase_dir: Path,
    model,
    processor,
    device: str,
    model_name: str,
    anchor_emb,
) -> dict:
    phase_dir = phase_dir.resolve()
    manifest = load_manifest(phase_dir)
    has_manifest = bool(manifest)

    cluster_dirs = sorted([d for d in phase_dir.iterdir() if d.is_dir()])

    print(f"\n{'='*60}")
    print(f"Phase dir: {phase_dir}")
    print(f"Model:     {model_name}")
    print(f"Clusters:  {[d.name for d in cluster_dirs]}")

    # ── Embed per cluster ────────────────────────────────────────────────────
    cluster_data: dict[str, dict] = {}

    for cd in cluster_dirs:
        pngs = sorted(cd.glob("*.png"))
        if not pngs:
            continue

        ordered_paths: list[Path] = []
        cohort_sus: list[int] = []
        cohort_job_ids: list[str] = []

        if has_manifest:
            png_by_id = {p.stem: p for p in pngs}
            for job_id, job in manifest.items():
                if job["cohort"] == cd.name and job_id in png_by_id:
                    ordered_paths.append(png_by_id[job_id])
                    cohort_sus.append(job["sus_level"])
                    cohort_job_ids.append(job_id)
            if not ordered_paths:
                ordered_paths = pngs
        else:
            ordered_paths = pngs

        print(f"\n  Embedding {len(ordered_paths)} images from {cd.name}...")
        embs = embed_images(ordered_paths, model, processor, device, model_name=model_name)

        cluster_data[cd.name] = {
            "paths": ordered_paths,
            "embs": embs,
            "sus_levels": cohort_sus,
            "job_ids": cohort_job_ids,
        }

    if len(cluster_data) < 2:
        print("Need at least 2 clusters to compute separation.")
        return {}

    labels = list(cluster_data.keys())

    # ── Anchor distance + sus correlation ────────────────────────────────────
    cohort_stats_list: list[dict] = []
    all_dists: list[float] = []
    all_sus: list[int] = []

    print(f"\n  Per-cohort anchor distance + sus correlation:")
    print(f"  {'cohort':25s} {'n':>5} {'avg_dist':>9} {'std':>7} {'sus':>10} {'r(dist,sus)':>12}")
    print("  " + "-" * 75)

    for cohort_name, data in cluster_data.items():
        if anchor_emb is not None:
            dists = anchor_distances(data["embs"], anchor_emb)
        else:
            dists = np.zeros(len(data["embs"]), dtype=np.float32)

        sus_levels = data["sus_levels"]
        corr = sus_correlation(dists, sus_levels)
        sus_range = f"{min(sus_levels)}-{max(sus_levels)}" if sus_levels else "n/a"
        avg_d = float(np.mean(dists))
        std_d = float(np.std(dists))
        corr_str = f"{corr:+.3f}" if corr is not None else "n/a"

        print(f"  {cohort_name:25s} {len(dists):>5} {avg_d:>9.4f} {std_d:>7.4f} "
              f"{sus_range:>10} {corr_str:>12}")

        cohort_stats_list.append({
            "cohort": cohort_name,
            "n": int(len(dists)),
            "avg_anchor_dist": avg_d,
            "std_anchor_dist": std_d,
            "sus_min": int(min(sus_levels)) if sus_levels else None,
            "sus_max": int(max(sus_levels)) if sus_levels else None,
            "sus_corr": corr,
        })

        all_dists.extend(dists.tolist())
        all_sus.extend(sus_levels)

    # ── Intra-cluster similarities ────────────────────────────────────────────
    print(f"\n  Intra-cluster cosine similarity:")
    intra_sims: list[float] = []
    for cohort_name, data in cluster_data.items():
        sims = pairwise_sims(data["embs"])
        avg = float(np.mean(sims)) if sims else float("nan")
        std = float(np.std(sims)) if sims else float("nan")
        intra_sims.extend(sims)
        for cs in cohort_stats_list:
            if cs["cohort"] == cohort_name:
                cs["avg_intra_sim"] = avg
                cs["std_intra_sim"] = std
                cs["n_pairs"] = len(sims)
        print(f"    {cohort_name:25s}: mean={avg:.4f}  std={std:.4f}  n_pairs={len(sims)}")

    # ── Inter-cluster similarities ────────────────────────────────────────────
    print(f"\n  Inter-cluster cosine similarity:")
    inter_sims: list[float] = []
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            ea = cluster_data[la]["embs"]
            eb = cluster_data[lb]["embs"]
            sims = [cosine_sim(ea[r], eb[c])
                    for r in range(len(ea)) for c in range(len(eb))]
            avg = float(np.mean(sims))
            inter_sims.extend(sims)
            print(f"    {la:20s} ↔ {lb:20s}: {avg:.4f}")

    # ── Separation score ──────────────────────────────────────────────────────
    intra_dist = float(np.mean([1 - s for s in intra_sims]))
    inter_dist = float(np.mean([1 - s for s in inter_sims]))
    separation = (inter_dist - intra_dist) / inter_dist if inter_dist > 0 else 0.0
    avg_intra_sim = float(np.mean(intra_sims))
    avg_inter_sim = float(np.mean(inter_sims))

    # ── Overall encoding validation ───────────────────────────────────────────
    overall_corr = sus_correlation(np.array(all_dists), all_sus) if all_sus else None
    verdict = ""
    if overall_corr is not None:
        if overall_corr > 0.3:
            verdict = "encoding validated"
        elif overall_corr > 0.1:
            verdict = "weak positive"
        else:
            verdict = "encoding not validated"

    print(f"\n  Summary:")
    print(f"    Avg intra-cluster sim: {avg_intra_sim:.4f}  (distance: {intra_dist:.4f})")
    print(f"    Avg inter-cluster sim: {avg_inter_sim:.4f}  (distance: {inter_dist:.4f})")
    print(f"    Separation score:      {separation:.4f}")
    if overall_corr is not None:
        print(f"    Overall r(anchor_dist, sus_level): {overall_corr:+.4f}  [{verdict}]")

    return {
        "phase_dir": str(phase_dir),
        "model": model_name,
        "clusters": labels,
        "cohort_stats": cohort_stats_list,
        "avg_intra_sim": avg_intra_sim,
        "avg_inter_sim": avg_inter_sim,
        "intra_dist": intra_dist,
        "inter_dist": inter_dist,
        "separation_score": separation,
        "encoding_validation": {
            "overall_sus_corr": overall_corr,
            "verdict": verdict,
            "n_faces_total": len(all_dists),
        },
    }


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(results: list[dict], phase_dirs: list[Path], model_name: str) -> Path:
    out = {
        "model": model_name,
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "phases": results,
    }
    out_path = phase_dirs[0].resolve() / f"face_distinctness_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Face fingerprinting metrics: separation, anchor drift, uncanniness."
    )
    parser.add_argument(
        "phase_dirs", nargs="+", type=Path,
        help="Dataset output directories",
    )
    parser.add_argument(
        "--model", choices=["arcface", "clip"], default="arcface",
        help="Embedding model (default: arcface)",
    )
    parser.add_argument(
        "--anchor", type=Path, default=ANCHOR_PATH,
        help=f"Anchor face path (default: {ANCHOR_PATH})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving JSON results",
    )
    args = parser.parse_args()

    if args.model == "arcface":
        model, processor, device = load_arcface()
    else:
        model, processor, device = load_clip()

    # Resolve anchor relative to project root (src/../)
    project_root = Path(__file__).parent.parent
    anchor_path = args.anchor if args.anchor.is_absolute() else project_root / args.anchor
    anchor_emb = embed_anchor(anchor_path, model, processor, device, args.model)

    results = []
    valid_dirs = []
    for d in args.phase_dirs:
        if not d.is_dir():
            print(f"WARNING: {d} is not a directory, skipping.")
            continue
        r = analyse_phase(
            phase_dir=d,
            model=model,
            processor=processor,
            device=device,
            model_name=args.model,
            anchor_emb=anchor_emb,
        )
        if r:
            results.append(r)
            valid_dirs.append(d)

    if len(results) >= 2:
        print(f"\n{'='*60}")
        print(f"Cross-directory comparison ({args.model}):")
        print(f"  {'dir':30s} {'sep_score':>10} {'intra_sim':>10} "
              f"{'inter_sim':>10} {'sus_corr':>10}")
        print("  " + "-" * 65)
        for r in results:
            name = Path(r["phase_dir"]).name
            corr = r["encoding_validation"]["overall_sus_corr"]
            corr_str = f"{corr:+.4f}" if corr is not None else "     n/a"
            print(f"  {name:30s} {r['separation_score']:>10.4f} "
                  f"{r['avg_intra_sim']:>10.4f} {r['avg_inter_sim']:>10.4f} "
                  f"{corr_str:>10}")
        best = max(results, key=lambda x: x["separation_score"])
        print(f"\n  Best separation: {Path(best['phase_dir']).name} "
              f"(score={best['separation_score']:.4f})")

    if not args.no_save and results:
        save_results(results, valid_dirs, args.model)


if __name__ == "__main__":
    main()
