"""Build glasses-direction .npz files for Stage 4.5 glasses-axis comparison.

Ours: logistic regression on CLIP zero-shot glasses probability (thresholded
at 0.5 to get binary labels), matching the regularization spirit of gender.
FluxSpace-coarse: prompt-pair contrast "wearing glasses" vs "no glasses".

Output:
  output/demographic_pc/edits/glasses_ours.npz
  output/demographic_pc/edits/glasses_fluxspace_coarse.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "demographic_pc"
EDITS_DIR = OUT / "edits"

CLIP_DIM = 768


def build_ours_glasses() -> dict[str, np.ndarray]:
    from src.demographic_pc.build_smile_edits import _load_combined
    C, ids, ex = _load_combined()
    extras = ex.set_index("sample_id").reindex(ids)
    p = extras["glasses_prob"].to_numpy(dtype=float)
    mask = np.isfinite(p)
    p = p[mask]
    Cm = C[mask]
    mu = Cm.mean(axis=0)
    Cc = Cm - mu
    pos_rate = (p > 0.5).mean()
    print(f"[ours]       n={mask.sum()}  glasses>0.5 rate={pos_rate:.3%}  p.mean={p.mean():.4g} p.std={p.std():.4g}")
    # If too few positives, fall back to ridge on continuous probability.
    if pos_rate < 0.02 or pos_rate > 0.98:
        print("             -> falling back to Ridge on continuous glasses_prob (too few positives)")
        model = Ridge(alpha=316.0, solver="svd").fit(Cc, p)
        w = model.coef_.astype(np.float64)
    else:
        y = (p > 0.5).astype(int)
        model = LogisticRegression(C=1.0 / 316.0, solver="lbfgs", max_iter=2000).fit(Cc, y)
        w = model.coef_[0].astype(np.float64)
        print(f"             train-acc={model.score(Cc, y):.3f}")
    direction = w / float(w @ w)
    pooled_delta = direction[:CLIP_DIM]
    seq_delta = direction[CLIP_DIM:]
    print(f"             ||w||={np.linalg.norm(w):.4g}  ||dir||={np.linalg.norm(direction):.4g}")
    return {"pooled_delta": pooled_delta.astype(np.float32),
            "seq_delta": seq_delta.astype(np.float32)}


def build_fluxspace_coarse_glasses() -> dict[str, np.ndarray]:
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

    base = "A photorealistic portrait photograph of a person, no glasses, neutral expression, plain grey background, studio lighting, sharp focus."
    target = "A photorealistic portrait photograph of a person wearing eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."
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
    print("\n== Building Ours glasses direction ==")
    d_ours = build_ours_glasses()
    np.savez(EDITS_DIR / "glasses_ours.npz", **d_ours)
    print(f"  -> {EDITS_DIR / 'glasses_ours.npz'}")
    print("\n== Building FluxSpace coarse glasses direction ==")
    d_fs = build_fluxspace_coarse_glasses()
    np.savez(EDITS_DIR / "glasses_fluxspace_coarse.npz", **d_fs)
    print(f"  -> {EDITS_DIR / 'glasses_fluxspace_coarse.npz'}")


if __name__ == "__main__":
    main()
