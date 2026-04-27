"""Build Black-vs-rest direction .npz files for Stage 4.5 race-axis comparison.

Ours: one-vs-rest logistic regression on FairFace race, target class = "Black".
FluxSpace-coarse: prompt-pair contrast "Black person" vs "person".

Output:
  output/demographic_pc/edits/black_ours.npz
  output/demographic_pc/edits/black_fluxspace_coarse.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "demographic_pc"
EDITS_DIR = OUT / "edits"

CLIP_DIM = 768


def _load_combined_black() -> tuple[np.ndarray, np.ndarray]:
    """Returns (C, y) where y is binary Black-vs-rest from Stage 2 + Stage 2b."""
    # Stage 2
    C1 = np.load(OUT / "conditioning.npy").astype(np.float64)
    ids1 = json.loads((OUT / "conditioning_ids.json").read_text())
    df1 = pd.read_parquet(OUT / "labels.parquet").set_index("sample_id").reindex(ids1)
    race1 = df1["fairface_race"].to_numpy()
    parts_C = [C1]
    parts_race = [race1]
    # Stage 2b black (if available)
    cb = OUT / "conditioning_black.npy"
    lb = OUT / "labels_black.parquet"
    if cb.exists() and lb.exists():
        C2 = np.load(cb).astype(np.float64)
        ids2 = json.loads((OUT / "conditioning_black_ids.json").read_text())
        df2 = pd.read_parquet(lb).set_index("sample_id").reindex(ids2)
        race2 = df2["fairface_race"].to_numpy()
        parts_C.append(C2)
        parts_race.append(race2)
        print(f"[combined] Stage2={len(ids1)} + Stage2b-black={len(ids2)} = {len(ids1)+len(ids2)}")
    else:
        print(f"[combined] Stage2 only ({len(ids1)}) — no conditioning_black.npy yet")
    C = np.concatenate(parts_C, axis=0)
    race = np.concatenate(parts_race, axis=0)
    mask = pd.notna(race)
    y = (race[mask] == "Black").astype(int)
    return C[mask], y


def build_ours_black() -> dict[str, np.ndarray]:
    Cm, y = _load_combined_black()
    mu = Cm.mean(axis=0)
    Cc = Cm - mu
    model = LogisticRegression(C=1.0 / 316.0, solver="lbfgs", max_iter=2000,
                               class_weight="balanced").fit(Cc, y)
    w = model.coef_[0].astype(np.float64)
    direction = w / float(w @ w)
    pooled_delta = direction[:CLIP_DIM]
    seq_delta = direction[CLIP_DIM:]
    acc = model.score(Cc, y)
    print(f"[ours]       n={len(y)}  pos={y.sum()}  train-acc={acc:.3f}")
    print(f"             ||w||={np.linalg.norm(w):.4g}  ||dir||={np.linalg.norm(direction):.4g}")
    return {"pooled_delta": pooled_delta.astype(np.float32),
            "seq_delta": seq_delta.astype(np.float32)}


def build_fluxspace_coarse_black() -> dict[str, np.ndarray]:
    from src.demographic_pc.stage2b_conditioning import (
        CLIP_HF_ID, CLIP_MAX_LEN, load_clip, load_t5,
    )
    from transformers import CLIPTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    clip_tok = CLIPTokenizer.from_pretrained(CLIP_HF_ID)
    _, clip_model = load_clip(device, dtype)
    t5_tok, t5_model = load_t5(device, dtype)

    def encode(text: str) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            c_in = clip_tok([text], padding="max_length", max_length=CLIP_MAX_LEN,
                            truncation=True, return_tensors="pt").to(device)
            pooled = clip_model(**c_in).pooler_output.squeeze(0).float().cpu().numpy()
            t_in = t5_tok([text], device=device)
            t_out = t5_model(**t_in).last_hidden_state
            mask = t_in["attention_mask"].unsqueeze(-1).to(t_out.dtype)
            t_mean = ((t_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).squeeze(0).float().cpu().numpy()
        return pooled, t_mean

    # Contrast: target is explicitly Black person vs unspecified baseline.
    base = "A photorealistic portrait photograph of a person, neutral expression, plain grey background, studio lighting, sharp focus."
    target = "A photorealistic portrait photograph of a Black person, neutral expression, plain grey background, studio lighting, sharp focus."
    pool_b, t5_b = encode(base)
    pool_t, t5_t = encode(target)

    def project_out(v, b):
        return v - (v @ b) / (b @ b) * b

    pooled_dir = project_out(pool_t, pool_b)
    seq_dir = project_out(t5_t, t5_b)
    pooled_delta = pooled_dir / max(np.linalg.norm(pooled_dir), 1e-8) * np.linalg.norm(pool_t - pool_b)
    seq_delta = seq_dir / max(np.linalg.norm(seq_dir), 1e-8) * np.linalg.norm(t5_t - t5_b)
    print(f"[fluxspace]  ||pooled_delta||={np.linalg.norm(pooled_delta):.4g}")
    print(f"             ||seq_delta||={np.linalg.norm(seq_delta):.4g}")
    return {"pooled_delta": pooled_delta.astype(np.float32),
            "seq_delta": seq_delta.astype(np.float32)}


def main() -> None:
    EDITS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n== Building Ours Black direction ==")
    d_ours = build_ours_black()
    np.savez(EDITS_DIR / "black_ours.npz", **d_ours)
    print(f"  -> {EDITS_DIR / 'black_ours.npz'}")
    print("\n== Building FluxSpace coarse Black direction ==")
    d_fs = build_fluxspace_coarse_black()
    np.savez(EDITS_DIR / "black_fluxspace_coarse.npz", **d_fs)
    print(f"  -> {EDITS_DIR / 'black_fluxspace_coarse.npz'}")


if __name__ == "__main__":
    main()
