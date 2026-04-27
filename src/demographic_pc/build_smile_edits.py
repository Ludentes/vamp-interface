"""Build smile-direction .npz files for Stage 4.5 smile-axis comparison.

Ours: ridge regression on MediaPipe blendshape "smile" (from labels_extras).
FluxSpace-coarse: prompt-pair contrast of "smiling" vs "neutral" portrait.

Output:
  output/demographic_pc/edits/smile_ours.npz
  output/demographic_pc/edits/smile_fluxspace_coarse.npz

Usage:
    uv run python -m src.demographic_pc.build_smile_edits
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "demographic_pc"
EDITS_DIR = OUT / "edits"

CLIP_DIM = 768


def _load_combined() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    C1 = np.load(OUT / "conditioning.npy").astype(np.float64)
    ids1 = json.loads((OUT / "conditioning_ids.json").read_text())
    ex1 = pd.read_parquet(OUT / "labels_extras.parquet")
    parts_C = [C1]
    parts_ids = list(ids1)
    parts_ex = [ex1]
    cond2b = OUT / "conditioning_2b.npy"
    if cond2b.exists():
        C2 = np.load(cond2b).astype(np.float64)
        ids2 = json.loads((OUT / "conditioning_2b_ids.json").read_text())
        ex2 = pd.read_parquet(OUT / "labels_extras_2b.parquet")
        parts_C.append(C2)
        parts_ids.extend(ids2)
        parts_ex.append(ex2)
        print(f"[combined] Stage2={len(ids1)} + Stage2b={len(ids2)} = {len(parts_ids)}")
    else:
        print(f"[combined] Stage2 only ({len(ids1)}) — no conditioning_2b.npy yet")
    return np.concatenate(parts_C, axis=0), parts_ids, pd.concat(parts_ex, ignore_index=True)


def build_ours_smile() -> dict[str, np.ndarray]:
    C, ids, ex = _load_combined()
    extras = ex.set_index("sample_id").reindex(ids)
    y = extras["smile"].to_numpy(dtype=float)
    mask = np.isfinite(y) & extras["blendshapes_detected"].to_numpy()
    y = y[mask]
    Cm = C[mask]
    mu = Cm.mean(axis=0)
    Cc = Cm - mu
    model = Ridge(alpha=316.0, solver="svd").fit(Cc, y)
    w = model.coef_.astype(np.float64)
    direction = w / float(w @ w)  # unit strength = +1 smile-unit shift
    pooled_delta = direction[:CLIP_DIM]
    seq_delta = direction[CLIP_DIM:]
    print(f"[ours]       n={mask.sum()}  y.mean={y.mean():.4g}  y.std={y.std():.4g}")
    print(f"             ||w||={np.linalg.norm(w):.4g}  ||dir||={np.linalg.norm(direction):.4g}")
    return {"pooled_delta": pooled_delta.astype(np.float32),
            "seq_delta": seq_delta.astype(np.float32)}


def build_fluxspace_coarse_smile() -> dict[str, np.ndarray]:
    from src.demographic_pc.stage2b_conditioning import (
        CLIP_HF_ID, CLIP_MAX_LEN,
        load_clip, load_t5,
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

    base = "A photorealistic portrait photograph of a person, neutral expression, plain grey background, studio lighting, sharp focus."
    target = "A photorealistic portrait photograph of a person, broad smile showing teeth, plain grey background, studio lighting, sharp focus."
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
    print("\n== Building Ours smile direction ==")
    d_ours = build_ours_smile()
    np.savez(EDITS_DIR / "smile_ours.npz", **d_ours)
    print(f"  -> {EDITS_DIR / 'smile_ours.npz'}")
    print("\n== Building FluxSpace coarse smile direction ==")
    d_fs = build_fluxspace_coarse_smile()
    np.savez(EDITS_DIR / "smile_fluxspace_coarse.npz", **d_fs)
    print(f"  -> {EDITS_DIR / 'smile_fluxspace_coarse.npz'}")


if __name__ == "__main__":
    main()
