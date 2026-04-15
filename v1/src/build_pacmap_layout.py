#!/usr/bin/env python3
"""
build_pacmap_layout.py — Compute PaCMAP 2D layout from job embeddings.

Reads data/test_dataset.json, runs PaCMAP on the 1024-d embeddings,
outputs output/pacmap_layout.json with per-job 2D coordinates + metadata.

PaCMAP is preferred over UMAP: better local+global structure preservation,
deterministic with fixed seed, no multi-thread non-determinism.

Usage:
    uv run src/build_pacmap_layout.py
    uv run src/build_pacmap_layout.py --face-version flux_v3
    uv run src/build_pacmap_layout.py --n-neighbors 15 --out output/layout_custom.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

DATA_PATH = Path("data/test_dataset.json")
OUT_PATH   = Path("output/pacmap_layout.json")
FACE_VERSIONS = ["v1", "v2", "v3", "flux", "flux_v3"]


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["jobs"]


def run_pacmap(embeddings: np.ndarray, n_neighbors: int, seed: int) -> np.ndarray:
    import pacmap
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


def face_url(job_id: str, face_version: str, cohort: str) -> str | None:
    """Return relative path to the face PNG for this job, if it exists."""
    dir_map = {
        "v1":      "dataset_faces",
        "v2":      "dataset_faces_v2",
        "v3":      "dataset_faces_v3",
        "flux":    "dataset_faces_flux",
        "flux_v3": "dataset_faces_flux_v3",
    }
    dirname = dir_map.get(face_version, "dataset_faces")
    path = Path("output") / dirname / cohort / f"{job_id}.png"
    return str(path) if path.exists() else None


def main():
    parser = argparse.ArgumentParser(
        description="Build PaCMAP 2D layout from job embeddings."
    )
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument(
        "--face-version", choices=FACE_VERSIONS, default="flux_v3",
        help="Which face dataset to link URLs to (default: flux_v3)",
    )
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading dataset from {args.input}...")
    jobs = load_dataset(args.input)
    print(f"  {len(jobs)} jobs loaded")

    embeddings = np.array([j["embedding"] for j in jobs], dtype=np.float32)
    print(f"  Embedding matrix: {embeddings.shape}")

    print(f"\nRunning PaCMAP (n_neighbors={args.n_neighbors}, seed={args.seed})...")
    coords = run_pacmap(embeddings, args.n_neighbors, args.seed)
    print(f"  Done. Output shape: {coords.shape}")

    # Normalise to [0, 1] for stable frontend rendering
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_norm = (coords[:, 0] - x_min) / (x_max - x_min)
    y_norm = (coords[:, 1] - y_min) / (y_max - y_min)

    print(f"\nBuilding layout records (face_version={args.face_version})...")
    records = []
    missing_faces = 0
    for i, job in enumerate(jobs):
        job_id  = job["id"]
        cohort  = job["cohort"]
        furl    = face_url(job_id, args.face_version, cohort)
        if furl is None:
            missing_faces += 1

        record = {
            "id":                  job_id,
            "x":                   round(float(x_norm[i]), 6),
            "y":                   round(float(y_norm[i]), 6),
            # Identity
            "cohort":              cohort,
            "work_type":           job.get("work_type"),
            "fraud":               job.get("fraud"),
            # Fraud signal
            "sus_level":           job.get("sus_level"),
            "sus_category":        job.get("sus_category"),
            "sus_factors":         job.get("sus_factors", {}),
            # Source
            "source_name":         job.get("source_name"),
            "sender_id":           job.get("sender_id"),
            "telegram_chat_id":    job.get("telegram_chat_id"),
            # Contact
            "contact_telegram":    job.get("contact_telegram"),
            "contact_phone_hash":  job.get("contact_phone_hash"),
            # Content
            "text":                job.get("text", "")[:500],
            "created_at":          job.get("created_at"),
            # Face
            "face_path":           furl,
        }
        records.append(record)

    if missing_faces:
        print(f"  WARNING: {missing_faces} jobs have no face PNG for version '{args.face_version}'")

    out = {
        "built_at":    datetime.now().isoformat(),
        "face_version": args.face_version,
        "n_jobs":       len(records),
        "n_neighbors":  args.n_neighbors,
        "seed":         args.seed,
        "x_range":      [round(float(x_min), 4), round(float(x_max), 4)],
        "y_range":      [round(float(y_min), 4), round(float(y_max), 4)],
        "points":       records,
    }

    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    size_kb = args.out.stat().st_size / 1024
    print(f"\nSaved: {args.out}  ({size_kb:.0f} KB)")
    print(f"  {len(records)} points, {missing_faces} missing face PNGs")


if __name__ == "__main__":
    main()
