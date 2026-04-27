"""Pick top-K (step, block) sites by delta_mix fro from the existing
channel-mean smile_inphase cache. Output a JSON allow-list consumed by
FluxSpaceEditPair's full_capture_sites_json input.

Run:
  uv run python -m src.demographic_pc.phase1_pick_topk_sites --k 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/blendshape_nmf/attn_cache/smile_inphase"
OUT = ROOT / "output/demographic_pc/phase1_topk_sites.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    dm = np.load(CACHE / "delta_mix.npy", mmap_mode="r")  # (N, S, B, D) fp16
    meta = json.loads((CACHE / "meta.json").read_text())
    step_keys = list(meta["step_keys"])
    block_keys = list(meta["block_keys"])
    N, S, B, D = dm.shape
    assert S == len(step_keys) and B == len(block_keys), f"shape mismatch"

    # Mean over samples, then fro per (S, B) site.
    mean_dm = np.zeros((S, B, D), dtype=np.float64)
    chunk = 64
    for i in range(0, N, chunk):
        mean_dm += dm[i:i+chunk].astype(np.float32).sum(axis=0)
    mean_dm /= N
    fro = np.linalg.norm(mean_dm, axis=-1)  # (S, B)

    flat = np.argsort(-fro.ravel())[:args.k]
    sites = []
    print(f"[topk] top-{args.k} sites by channel-mean delta_mix fro:")
    for f in flat:
        si, bi = int(f) // B, int(f) % B
        sk = int(step_keys[si]); bk = str(block_keys[bi])
        sites.append([sk, bk])
        print(f"  step={sk:3d} block={bk:12s} fro={fro[si,bi]:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sites": sites, "k": args.k,
                               "source": str(CACHE), "metric": "channel-mean delta_mix fro"}, indent=2))
    print(f"[topk] → {OUT}")


if __name__ == "__main__":
    main()
