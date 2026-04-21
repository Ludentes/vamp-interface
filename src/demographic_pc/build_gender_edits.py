"""Build gender-direction .npz files for Stage 4.5 gender-axis comparison.

Mirrors build_age_edits.py but targets binary gender via L2-regularized
logistic regression (ours) and a symmetric man/woman prompt contrast
(fluxspace-coarse). Positive lambda pushes toward the "feminine" pole for
both methods so the axis sign is consistent.

Output:
  output/demographic_pc/edits/gender_ours.npz
  output/demographic_pc/edits/gender_fluxspace_coarse.npz

Usage:
    uv run python -m src.demographic_pc.build_gender_edits
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
T5_DIM = 4096


def build_ours_gender() -> dict[str, np.ndarray]:
    C = np.load(OUT / "conditioning.npy").astype(np.float64)
    ids = json.loads((OUT / "conditioning_ids.json").read_text())
    df = pd.read_parquet(OUT / "labels.parquet").set_index("sample_id").reindex(ids)
    # Map MiVOLO gender to {M: 0, F: 1}; drop rows with missing label.
    y_str = df["mivolo_gender"].to_numpy()
    mask = pd.notna(y_str)
    y = (y_str[mask] == "F").astype(int)
    Cm = C[mask]
    mu = Cm.mean(axis=0)
    Cc = Cm - mu
    # Match the scale/regularization spirit of age ridge. alpha=316 there;
    # logistic regularization C ~ 1/alpha -> C=1/316 for comparable shrinkage.
    model = LogisticRegression(C=1.0 / 316.0, solver="lbfgs", max_iter=2000).fit(Cc, y)
    w = model.coef_[0].astype(np.float64)                    # (4864,), points toward F
    direction = w / float(w @ w)                             # unit strength = +1 logit toward F
    pooled_delta = direction[:CLIP_DIM]
    seq_delta = direction[CLIP_DIM:]
    acc = model.score(Cc, y)
    print(f"[ours]       train-acc={acc:.3f}  ||w||={np.linalg.norm(w):.4g}  ||dir||={np.linalg.norm(direction):.4g}")
    print(f"             ||pooled_delta||={np.linalg.norm(pooled_delta):.4g}")
    print(f"             ||seq_delta||={np.linalg.norm(seq_delta):.4g}")
    return {"pooled_delta": pooled_delta.astype(np.float32),
            "seq_delta": seq_delta.astype(np.float32)}


def build_fluxspace_coarse_gender() -> dict[str, np.ndarray]:
    from src.demographic_pc.stage2b_conditioning import (
        CLIP_HF_ID, CLIP_MAX_LEN,
        load_clip, load_t5,
    )
    from transformers import CLIPTokenizer  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    clip_tok = CLIPTokenizer.from_pretrained(CLIP_HF_ID)
    _, clip_model = load_clip(device, dtype)
    t5_tok, t5_model = load_t5(device, dtype)

    def encode(text: str) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            c_in = clip_tok(
                [text], padding="max_length", max_length=CLIP_MAX_LEN,
                truncation=True, return_tensors="pt",
            ).to(device)
            pooled = clip_model(**c_in).pooler_output.squeeze(0).float().cpu().numpy()
            t_in = t5_tok([text], device=device)
            t_out = t5_model(**t_in).last_hidden_state
            mask = t_in["attention_mask"].unsqueeze(-1).to(t_out.dtype)
            t_mean = ((t_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).squeeze(0).float().cpu().numpy()
        return pooled, t_mean

    # Symmetric adult pair so the direction is "masculine presentation" -> "feminine presentation".
    base_prompt = "A photorealistic portrait photograph of a man, neutral expression, plain grey background, studio lighting, sharp focus."
    target_prompt = "A photorealistic portrait photograph of a woman, neutral expression, plain grey background, studio lighting, sharp focus."

    pool_b, t5_b = encode(base_prompt)
    pool_t, t5_t = encode(target_prompt)

    def project_out(v: np.ndarray, base: np.ndarray) -> np.ndarray:
        proj = (v @ base) / (base @ base) * base
        return v - proj

    pooled_dir = project_out(pool_t, pool_b)
    seq_dir = project_out(t5_t, t5_b)
    pooled_delta = pooled_dir / max(np.linalg.norm(pooled_dir), 1e-8) * np.linalg.norm(pool_t - pool_b)
    seq_delta = seq_dir / max(np.linalg.norm(seq_dir), 1e-8) * np.linalg.norm(t5_t - t5_b)
    print(f"[fluxspace]  ||pooled_delta||={np.linalg.norm(pooled_delta):.4g}")
    print(f"             ||seq_delta||={np.linalg.norm(seq_delta):.4g}")
    print(f"             cos(target,base) pool={pool_t @ pool_b / (np.linalg.norm(pool_t)*np.linalg.norm(pool_b)):.3f}  "
          f"t5_mean={t5_t @ t5_b / (np.linalg.norm(t5_t)*np.linalg.norm(t5_b)):.3f}")
    return {"pooled_delta": pooled_delta.astype(np.float32),
            "seq_delta": seq_delta.astype(np.float32)}


def main() -> None:
    EDITS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n== Building Ours gender direction ==")
    d_ours = build_ours_gender()
    np.savez(EDITS_DIR / "gender_ours.npz", **d_ours)
    print(f"  -> {EDITS_DIR / 'gender_ours.npz'}")

    print("\n== Building FluxSpace coarse gender direction ==")
    d_fs = build_fluxspace_coarse_gender()
    np.savez(EDITS_DIR / "gender_fluxspace_coarse.npz", **d_fs)
    print(f"  -> {EDITS_DIR / 'gender_fluxspace_coarse.npz'}")


if __name__ == "__main__":
    main()
