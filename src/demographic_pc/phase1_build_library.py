"""Phase 2 of option-B rebuild: assemble a replay library from Phase 1 captures.

Reads the memmapped (attn_base.npy, delta_mix.npy, meta.json) triples produced
by phase1_full_capture.py, seed-averages delta_mix, and writes one clean
.npz that FluxSpaceDirectionInjectFull consumes.

Layout of each Phase 1 capture (trio of files per render):
  <stem>.attn_base.npy  — memmap (FULL_CAP_MAX_S, FULL_CAP_MAX_B, L, D) fp16
  <stem>.delta_mix.npy  — same shape
  <stem>.meta.json      — {step_order: [..], block_order: [..], used_shape: [..], ..}

Run:
  uv run python -m src.demographic_pc.phase1_build_library
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[2] / "output/demographic_pc/phase1_full_captures"
OUT = Path(__file__).resolve().parents[2] / "models/blendshape_nmf/phase1_smile_delta_full.npz"


def _load_capture(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    stem = meta_path.with_suffix("")  # strip .json
    # meta_path is <stem>.meta.json → stem file has .meta at end; strip it
    stem_str = str(stem)
    if stem_str.endswith(".meta"):
        stem_str = stem_str[:-5]
    ab = np.load(stem_str + ".attn_base.npy", mmap_mode="r")
    dm = np.load(stem_str + ".delta_mix.npy", mmap_mode="r")
    used_S, used_B, L, D = meta["used_shape"]
    return {
        "meta": meta,
        "step_order": meta["step_order"],
        "block_order": meta["block_order"],
        "ab": ab[:used_S, :used_B],   # (S, B, L, D)
        "dm": dm[:used_S, :used_B],
        "L": L, "D": D,
    }


def main() -> None:
    metas = sorted(ARCHIVE.glob("*.meta.json"))
    if not metas:
        print(f"[phase2] no captures found under {ARCHIVE}")
        return
    print(f"[phase2] found {len(metas)} captures")

    caps = [_load_capture(p) for p in metas]
    ref = caps[0]
    ref_steps = ref["step_order"]
    ref_blocks = ref["block_order"]
    L, D = ref["L"], ref["D"]
    print(f"[phase2] reference: S={len(ref_steps)} B={len(ref_blocks)} L={L} D={D}")

    # Consistency check: all captures must share the same (step_order, block_order, L, D).
    for p, c in zip(metas, caps):
        if c["step_order"] != ref_steps or c["block_order"] != ref_blocks:
            raise RuntimeError(f"order mismatch in {p.name}: "
                               f"steps={c['step_order']} blocks={c['block_order']}")
        if c["L"] != L or c["D"] != D:
            raise RuntimeError(f"shape mismatch in {p.name}: L={c['L']} D={c['D']}")

    # Seed-averaged delta.
    S, B = len(ref_steps), len(ref_blocks)
    delta_acc = np.zeros((S, B, L, D), dtype=np.float64)
    for c in caps:
        delta_acc += c["dm"].astype(np.float32)
    delta_mean = (delta_acc / len(caps)).astype(np.float16)

    # Diagnostic: per-site mean fro
    dm_fro = np.linalg.norm(delta_mean.astype(np.float32).reshape(S, B, -1), axis=-1)
    print(f"[phase2] delta fro:  mean={dm_fro.mean():.2f}  max={dm_fro.max():.2f}  "
          f"argmax=(si={int(dm_fro.argmax()//B)} bi={int(dm_fro.argmax()%B)})")
    print(f"[phase2] top-5 sites by fro:")
    flat_ord = np.argsort(-dm_fro.ravel())[:5]
    for f in flat_ord:
        si, bi = int(f) // B, int(f) % B
        print(f"  step={ref_steps[si]} block={ref_blocks[bi]} fro={dm_fro[si,bi]:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        delta_full=delta_mean,
        step_keys=np.asarray(ref_steps, dtype=np.int64),
        block_keys=np.asarray(ref_blocks, dtype=object),
        n_captures=np.asarray(len(caps), dtype=np.int64),
    )
    print(f"[phase2] → {OUT} ({OUT.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
