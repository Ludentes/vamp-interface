"""Paired-delta directions for smile and glasses (de-confounded by design).

Stage 2b was built as a grid:
  smile:   2 ages × 3 genders × 7 ethnicities × 5 smile levels × 1 seed
  glasses: 2 ages × 3 genders × 7 ethnicities × 4 glasses levels × 1 seed

For each demographic cell (age, gender, ethnicity), we compute
    delta_cell = enc(target_level) - enc(baseline_level)
and average across the 42 cells. Averaging cancels demographic components
(same in both terms), leaving only the smile/glasses-specific conditioning
shift. This is the "prompt-pair" contrast at scale, not the ridge solution.

Baseline/target pairs:
  smile:   neutral → broad         (closed, calm face → teeth-visible smile)
  glasses: none    → thick         (no eyewear → thick-rim eyeglasses)

Outputs (unit-length direction, preserving pair magnitude):
  output/demographic_pc/edits/smile_ours_paired.npz
  output/demographic_pc/edits/glasses_ours_paired.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "demographic_pc"
EDITS_DIR = OUT / "edits"

CLIP_DIM = 768


def _paired_direction(axis_prefix: str, baseline_key: str, target_key: str
                      ) -> dict[str, np.ndarray]:
    C = np.load(OUT / "conditioning_2b.npy").astype(np.float64)
    ids = json.loads((OUT / "conditioning_2b_ids.json").read_text())
    idx_of = {sid: i for i, sid in enumerate(ids)}

    # Group by (ai, gi, ei) and find matching baseline/target samples.
    # sample_id format: "{axis}-{ai}-{gi}-{ei}-{lkey}-s{seed}"
    groups: dict[tuple[int, int, int], dict[str, list[int]]] = {}
    for sid in ids:
        if not sid.startswith(axis_prefix + "-"):
            continue
        parts = sid.split("-")
        if len(parts) < 6:
            continue
        ai, gi, ei = int(parts[1]), int(parts[2]), int(parts[3])
        lkey = parts[4]
        groups.setdefault((ai, gi, ei), {}).setdefault(lkey, []).append(idx_of[sid])

    deltas = []
    matched = 0
    for (ai, gi, ei), per_level in groups.items():
        if baseline_key not in per_level or target_key not in per_level:
            continue
        v_base = C[per_level[baseline_key]].mean(axis=0)
        v_target = C[per_level[target_key]].mean(axis=0)
        deltas.append(v_target - v_base)
        matched += 1
    if not deltas:
        raise RuntimeError(f"no matched cells for {axis_prefix} {baseline_key}->{target_key}")
    D = np.stack(deltas, axis=0)  # (cells, 4864)
    mean_delta = D.mean(axis=0)
    pair_mag = float(np.linalg.norm(mean_delta))
    # Per-component magnitude (matches fluxspace_coarse shape)
    pooled_raw = mean_delta[:CLIP_DIM]
    seq_raw = mean_delta[CLIP_DIM:]
    # Also report per-cell variance as a contamination proxy: var_across_cells / ||mean||
    D_centered = D - mean_delta
    noise_norm = float(np.linalg.norm(D_centered, axis=1).mean())
    print(f"[paired:{axis_prefix}]  cells matched: {matched}")
    print(f"  ||mean_delta||={pair_mag:.4g}  ||pooled||={np.linalg.norm(pooled_raw):.4g}  ||seq||={np.linalg.norm(seq_raw):.4g}")
    print(f"  per-cell noise ||delta_cell - mean||={noise_norm:.4g}  (SNR={pair_mag/noise_norm:.2f})")
    return {
        "pooled_delta": pooled_raw.astype(np.float32),
        "seq_delta": seq_raw.astype(np.float32),
    }


def main() -> None:
    EDITS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n== smile (neutral → broad) ==")
    d_smile = _paired_direction("smile", baseline_key="neutral", target_key="broad")
    np.savez(EDITS_DIR / "smile_ours_paired.npz", **d_smile)
    print(f"  -> {EDITS_DIR / 'smile_ours_paired.npz'}")

    print("\n== glasses (none → thick) ==")
    d_glasses = _paired_direction("glasses", baseline_key="none", target_key="thick")
    np.savez(EDITS_DIR / "glasses_ours_paired.npz", **d_glasses)
    print(f"  -> {EDITS_DIR / 'glasses_ours_paired.npz'}")


if __name__ == "__main__":
    main()
