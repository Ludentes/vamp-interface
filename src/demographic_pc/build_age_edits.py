"""Build age-direction .npz files for Stage 4.5 method comparison.

Emits two edit files, both in the same format expected by
the ComfyUI ApplyConditioningEdit node:
    {"pooled_delta": (768,), "seq_delta": (4096,)}

  (1) age_ours.npz
      MiVOLO-age ridge regression on centered concat[CLIP-pool, T5-mean].
      Unit "strength" = +1 year of predicted age shift.
      δC = w / ‖w‖²  so that the linear model changes by δC·w = 1.

  (2) age_fluxspace_coarse.npz
      FluxSpace coarse variant: per-attribute prompt-pair contrast at
      pooled-CLIP + T5-mean level.
          target_prompt = "an elderly person portrait photograph..."
          base_prompt   = "a young adult person portrait photograph..."
      Direction = (c_target - proj_base(c_target)) — i.e. the component
      of target NOT already present in base.
      Strength scale is arbitrary; evaluator will rescale to match
      ours' slope on target-axis response.

Usage:
    uv run python -m src.demographic_pc.build_age_edits
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
T5_DIM = 4096


def build_ours_age() -> dict[str, np.ndarray]:
    C = np.load(OUT / "conditioning.npy").astype(np.float64)   # (N, 4864)
    ids = json.loads((OUT / "conditioning_ids.json").read_text())
    df = pd.read_parquet(OUT / "labels.parquet").set_index("sample_id").reindex(ids)
    y = df.mivolo_age.to_numpy(dtype=float)
    mu = C.mean(axis=0)
    Cc = C - mu                                               # centered
    model = Ridge(alpha=316.0, solver="svd").fit(Cc, y)
    w = model.coef_.astype(np.float64)                        # (4864,)
    direction = w / float(w @ w)                              # 1-year unit shift
    pooled_delta = direction[:CLIP_DIM]                       # (768,)
    seq_delta = direction[CLIP_DIM:]                          # (4096,)
    print(f"[ours]       ||w||={np.linalg.norm(w):.4g}  ||dir||={np.linalg.norm(direction):.4g}")
    print(f"             ||pooled_delta||={np.linalg.norm(pooled_delta):.4g}")
    print(f"             ||seq_delta||={np.linalg.norm(seq_delta):.4g}")
    return {"pooled_delta": pooled_delta.astype(np.float32),
            "seq_delta": seq_delta.astype(np.float32)}


def build_fluxspace_coarse_age() -> dict[str, np.ndarray]:
    # Need CLIP-L + T5 encoder; reuse Stage 2b's loaders.
    from src.demographic_pc.stage2b_conditioning import (
        CLIP_HF_ID, CLIP_MAX_LEN, T5_MAX_LEN,
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

    base_prompt = "A photorealistic portrait photograph of a young adult person, neutral expression, plain grey background, studio lighting, sharp focus."
    target_prompt = "A photorealistic portrait photograph of an elderly person, neutral expression, plain grey background, studio lighting, sharp focus."

    pool_b, t5_b = encode(base_prompt)
    pool_t, t5_t = encode(target_prompt)

    def project_out(v: np.ndarray, base: np.ndarray) -> np.ndarray:
        proj = (v @ base) / (base @ base) * base
        return v - proj

    pooled_dir = project_out(pool_t, pool_b)
    seq_dir = project_out(t5_t, t5_b)
    # Normalize so that full strength (strength=1.0) matches magnitude of
    # (target - base), so FluxSpace-native "push from base to target"
    # corresponds to strength≈1. Evaluator will rescale later for slope-matched comparison.
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
    print("\n== Building Ours age direction ==")
    d_ours = build_ours_age()
    np.savez(EDITS_DIR / "age_ours.npz", **d_ours)
    print(f"  -> {EDITS_DIR / 'age_ours.npz'}")

    print("\n== Building FluxSpace coarse age direction ==")
    d_fs = build_fluxspace_coarse_age()
    np.savez(EDITS_DIR / "age_fluxspace_coarse.npz", **d_fs)
    print(f"  -> {EDITS_DIR / 'age_fluxspace_coarse.npz'}")


if __name__ == "__main__":
    main()
