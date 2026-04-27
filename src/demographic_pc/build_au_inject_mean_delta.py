"""Build the correct cached-δ library: per-(step, block) MEAN of delta_mix
across the corpus, at raw (non-normalised) magnitude.

delta_mix in the cache is already ((attn_a + attn_b)/2 − attn_base).mean_d —
a pair-subtracted edit direction per sample. Averaging across samples gives
the mean edit vector at that (step, block), which is exactly the thing
FluxSpaceEditPair writes at scale=1.0 (just pre-averaged across samples).

Per component: pick its best corpus (by per-comp_r2 sanity), take the corpus's
mean δ tensor at all 912 sites. This is the correct injection library.

Output: models/blendshape_nmf/au_inject_mean.npz

Usage:
  uv run python -m src.demographic_pc.build_au_inject_mean_delta
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache"
LIB_NPZ = ROOT / "models/blendshape_nmf/au_library.npz"
OUT_NPZ = ROOT / "models/blendshape_nmf/au_inject_mean.npz"
OUT_LABELS = ROOT / "models/blendshape_nmf/au_inject_mean_labels.json"


def main():
    lib = np.load(LIB_NPZ, allow_pickle=True)
    H = lib["H"]
    per_comp_r2 = lib["per_comp_r2"]
    tags = list(lib["tags"])
    names = list(lib["names"])
    K = H.shape[0]

    meta = json.load(open(CACHE / "smile_inphase" / "meta.json"))
    step_keys = np.asarray(meta["step_keys"], dtype=np.int64)
    block_keys = np.asarray(meta["block_keys"])
    S, B = len(step_keys), len(block_keys)

    out = {"step_keys": step_keys, "block_keys": block_keys}
    labels = {}

    # Cache: for each corpus tag, mean delta_mix across samples (16, 57, 3072).
    # Compute once per corpus and reuse.
    corpus_mean: dict[str, np.ndarray] = {}
    for tag in set(tags):
        arr = np.load(CACHE / tag / "delta_mix.npy", mmap_mode="r")
        N = arr.shape[0]
        # Chunked reduction
        acc = np.zeros((S, B, 3072), dtype=np.float64)
        chunk = 32
        for i in range(0, N, chunk):
            blk = arr[i:i+chunk].astype(np.float32)
            acc += blk.sum(axis=0)
        acc /= N
        corpus_mean[tag] = acc.astype(np.float32)
        print(f"[mean] {tag:24s} N={N}  ||mean||_fro over (step,block,D) = {np.linalg.norm(acc):.2f}")

    for c in range(K):
        row = per_comp_r2[c]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        best_t = int(np.argmax(np.where(valid, row, -np.inf)))
        best_tag = tags[best_t]
        mean_stack = corpus_mean[best_tag]  # (S, B, 3072)

        # Flatten to (S*B, 3072) in the SAME ordering as atom_sites
        dirs = mean_stack.reshape(S * B, 3072)
        sites = np.arange(S * B, dtype=np.int64)

        out[f"atom_{c:02d}_direction"] = dirs.astype(np.float32)
        out[f"atom_{c:02d}_sites"] = sites

        top_j = int(np.argmax(H[c]))
        labels[f"atom_{c:02d}"] = {
            "component": int(c),
            "top_blendshape": str(names[top_j]),
            "best_corpus": best_tag,
            "K_sites": int(S * B),
            "mean_fro": float(np.linalg.norm(dirs)),
            "per_site_mean_fro": float(np.linalg.norm(dirs, axis=1).mean()),
        }
        print(f"[C{c}] corpus={best_tag} top_bs={names[top_j]} "
              f"||mean||_fro={np.linalg.norm(dirs):.2f} "
              f"per_site_fro_mean={np.linalg.norm(dirs, axis=1).mean():.3f}")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **out)
    OUT_LABELS.write_text(json.dumps(labels, indent=2))
    print(f"\n[npz] → {OUT_NPZ}")
    print(f"[labels] → {OUT_LABELS}")


if __name__ == "__main__":
    main()
