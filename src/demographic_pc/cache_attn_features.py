"""One-time cache of `delta_mix.mean_d` + `attn_base.mean_d` per (step, block)
into a compact .npz per source directory.

Each attention pkl is ~112 MB because it contains full mean_d / rms_d /
steered_at_scale / ab_half_diff tensors for every (step, block). Phase 3
only needs the direction signal — `delta_mix.mean_d` primarily, plus
`attn_base.mean_d` for optional context. Cache writes out:

  meta.json         per-source ordering (blocks, step keys, sample ids)
  delta_mix.npy     (N, n_steps, n_blocks, D)  fp16 to halve disk
  attn_base.npy     (N, n_steps, n_blocks, D)  fp16

With N~1320, steps=16, blocks=57, D=3072 and fp16, each npy is
1320*16*57*3072*2 bytes ≈ 7.4 GB per (source). We cache per-source
instead of globally so we can stream-load.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
CACHE_ROOT = ROOT / "models/blendshape_nmf/attn_cache"

PAIRED_SOURCES = [
    (METRICS / "crossdemo/smile/smile_inphase/blendshapes.json",
     METRICS / "crossdemo/smile/smile_inphase", "smile_inphase"),
    (METRICS / "crossdemo/smile/jaw_inphase/blendshapes.json",
     METRICS / "crossdemo/smile/jaw_inphase", "jaw_inphase"),
    (METRICS / "crossdemo/smile/alpha_interp_attn/blendshapes.json",
     METRICS / "crossdemo/smile/alpha_interp_attn", "alpha_interp_attn"),
]


def cache_one_source(bs_json: Path, attn_dir: Path, tag: str) -> None:
    out_dir = CACHE_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        print(f"  [skip] {tag} already cached at {out_dir}")
        return

    scores = json.loads(bs_json.read_text())
    rels = sorted(scores.keys())
    print(f"  [cache] {tag}: {len(rels)} paired samples")

    # Probe one pkl to discover shape
    first_pkl = None
    for rel in rels:
        p = (attn_dir / rel).with_suffix(".pkl")
        if p.exists():
            first_pkl = p
            break
    if first_pkl is None:
        print(f"  [skip] {tag} no pkls found")
        return
    with open(first_pkl, "rb") as f:
        probe = pickle.load(f)
    step_keys = sorted(probe["steps"].keys())
    block_keys = sorted(probe["steps"][step_keys[0]].keys())
    sample_dim = probe["steps"][step_keys[0]][block_keys[0]]["delta_mix"]["mean_d"].shape[0]
    print(f"    steps={len(step_keys)} blocks={len(block_keys)} D={sample_dim}")

    N = len(rels)
    # fp16 storage
    delta_mix = np.zeros((N, len(step_keys), len(block_keys), sample_dim), dtype=np.float16)
    attn_base = np.zeros((N, len(step_keys), len(block_keys), sample_dim), dtype=np.float16)
    kept_rels = []
    skipped = 0
    for i, rel in enumerate(rels):
        pkl_path = (attn_dir / rel).with_suffix(".pkl")
        if not pkl_path.exists():
            skipped += 1
            continue
        try:
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            for si, sk in enumerate(step_keys):
                sd = d["steps"].get(sk, {})
                for bi, bk in enumerate(block_keys):
                    blk = sd.get(bk, {})
                    dm = blk.get("delta_mix", {}).get("mean_d")
                    ab = blk.get("attn_base", {}).get("mean_d")
                    if dm is not None:
                        delta_mix[i, si, bi] = dm.numpy().astype(np.float16) if hasattr(dm, "numpy") else np.asarray(dm, dtype=np.float16)
                    if ab is not None:
                        attn_base[i, si, bi] = ab.numpy().astype(np.float16) if hasattr(ab, "numpy") else np.asarray(ab, dtype=np.float16)
            kept_rels.append(rel)
        except Exception as e:
            print(f"    error on {rel}: {e}")
            skipped += 1

    # Trim arrays to kept samples only
    keep_idx = np.array([i for i, rel in enumerate(rels) if rel in set(kept_rels)])
    delta_mix = delta_mix[keep_idx]
    attn_base = attn_base[keep_idx]

    np.save(out_dir / "delta_mix.npy", delta_mix)
    np.save(out_dir / "attn_base.npy", attn_base)
    meta = {
        "tag": tag,
        "rels": kept_rels,
        "skipped": skipped,
        "step_keys": step_keys,
        "block_keys": block_keys,
        "D": int(sample_dim),
        "dtype": "float16",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  [cache] {tag} done — kept {len(kept_rels)}, skipped {skipped}")
    print(f"           delta_mix: {delta_mix.shape} {delta_mix.dtype}, "
          f"{delta_mix.nbytes / 1e9:.2f} GB")


def main() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for bs_json, attn_dir, tag in PAIRED_SOURCES:
        if not bs_json.exists():
            print(f"[cache] skip {tag}: no blendshapes.json")
            continue
        cache_one_source(bs_json, attn_dir, tag)


if __name__ == "__main__":
    main()
