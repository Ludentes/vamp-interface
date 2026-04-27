"""Convert au_library.npz into FluxSpaceDirectionInject-compatible format.

FluxSpaceDirectionInject expects:
  atom_{NN}_direction : (K, D) — per-site direction tensors
  atom_{NN}_sites     : (K,)   — flat site indices (step_pos * n_blocks + block_pos)
  step_keys           : (S,)   — global step indices
  block_keys          : (B,)   — block keys "{type}_{idx}"

For each NMF component, we take the best corpus (highest CV R²) and inject at
its peak (step, block). Directions are UNIT-NORMALISED so the renderer can
sweep scale without per-component calibration.

Also emits a label npz `au_inject_labels.npz` with name→atom_id mapping for
call-site ergonomics.

Usage:
  uv run python -m src.demographic_pc.build_au_inject_npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "models/blendshape_nmf/au_library.npz"
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
OUT_NPZ = ROOT / "models/blendshape_nmf/au_inject_k1.npz"
LABELS_JSON = ROOT / "models/blendshape_nmf/au_inject_labels.json"


def main():
    lib = np.load(LIB, allow_pickle=True)
    H = lib["H"]                        # (K, 52)
    W_refits = lib["W_refits"]          # (K, T, 3072)
    peak_coords = lib["peak_coords"]    # (T, 2) (step_idx, block_idx)
    per_comp_r2 = lib["per_comp_r2"]    # (K, T)
    tags = list(lib["tags"])
    names = list(lib["names"])

    # Take canonical step/block_keys from smile_inphase (they match across corpora)
    meta = json.load(open(CACHE / "smile_inphase" / "meta.json"))
    step_keys = np.asarray(meta["step_keys"], dtype=np.int64)
    block_keys = np.asarray(meta["block_keys"])
    n_blocks = len(block_keys)

    K = H.shape[0]
    out: dict = {
        "step_keys": step_keys,
        "block_keys": block_keys,
    }
    labels = {}

    for c in range(K):
        # Pick best corpus by CV R²
        row = per_comp_r2[c]
        valid_mask = ~np.isnan(row)
        if not valid_mask.any():
            continue
        scores = np.where(valid_mask, row, -np.inf)
        best_t = int(np.argmax(scores))
        best_tag = str(tags[best_t])
        best_r2 = float(row[best_t])

        # Peak (step, block) for that corpus
        s_i, b_i = int(peak_coords[best_t, 0]), int(peak_coords[best_t, 1])
        flat_idx = s_i * n_blocks + b_i

        # Direction vector — unit normalise
        w = W_refits[c, best_t].astype(np.float32)
        w_norm = float(np.linalg.norm(w))
        if w_norm < 1e-9:
            print(f"[skip] C{c} has zero-norm W for best corpus {best_tag}")
            continue
        w_unit = w / w_norm

        out[f"atom_{c:02d}_direction"] = w_unit[None, :].astype(np.float32)  # (1, 3072)
        out[f"atom_{c:02d}_sites"] = np.array([flat_idx], dtype=np.int64)

        # Top blendshape loading for semantic label
        top_j = int(np.argmax(H[c]))
        top_bs = str(names[top_j])
        labels[f"atom_{c:02d}"] = {
            "component": int(c),
            "top_blendshape": top_bs,
            "best_corpus": best_tag,
            "best_r2": best_r2,
            "peak_step": int(step_keys[s_i]),
            "peak_block": str(block_keys[b_i]),
            "w_norm_raw": w_norm,
        }
        print(f"[atom_{c:02d}] top={top_bs}  corpus={best_tag}  "
              f"R²={best_r2:+.3f}  peak=(step={step_keys[s_i]}, block={block_keys[b_i]})  "
              f"flat_idx={flat_idx}  ||w||_raw={w_norm:.3f}")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **out)
    print(f"\n[npz] → {OUT_NPZ}")
    LABELS_JSON.write_text(json.dumps(labels, indent=2))
    print(f"[labels] → {LABELS_JSON}")


if __name__ == "__main__":
    main()
