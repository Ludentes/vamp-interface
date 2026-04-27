"""Overnight broad-validation runner for FluxSpaceDirectionInject.

For each live atom (excluding the 4 dead NMF atoms at k=20), render at
a small scale grid on 2 base demographics with 2 seeds each. Then score
every PNG with MediaPipe and dump a single blendshapes.json per atom
directory. Tomorrow's analysis: compare measured blendshape delta
(rendered_bs - base_neutral_bs) against the atom's target direction
W_atom (from W_nmf_resid.npy).

No measurement pkls saved — just PNGs. Scoring happens after all renders
finish so a partial run still produces usable data.

Layout:
  output/demographic_pc/direction_inject_broad/
    atom{NN}/{base}/seed{seed}_s{scale:+.2f}.png
    atom{NN}/blendshapes.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    CALIBRATION_PROMPTS, FLUX_GUIDANCE, FLUX_SAMPLER, FLUX_SCHEDULER,
    FLUX_STEPS, W, H, FLUX_CHECKPOINT, FLUX_VAE, FLUX_CLIP_L, FLUX_T5,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / "models/blendshape_nmf/directions_resid_causal.npz"
OUT_ROOT = ROOT / "output/demographic_pc/direction_inject_broad"
DEAD_ATOMS = {0, 2, 3, 6}


BASES = [
    ("young_european_f", CALIBRATION_PROMPTS[8][1]),
    ("european_m",       CALIBRATION_PROMPTS[3][1]),
    ("elderly_latin_m",  CALIBRATION_PROMPTS[5][1]),
]
SEEDS = [2026, 4242, 7331]
# Scales — 0/0.5/1/2 span base→visible→strong→collapse onset for atom
# injection. Dropped 4.0 (past collapse on most atoms) to shave time.
SCALES = [0.0, 0.5, 1.0, 2.0]


def _base_workflow() -> dict:
    import os as _os
    unet_name = FLUX_CHECKPOINT if _os.environ.get("COMFY_UNET") else FLUX_CHECKPOINT.removeprefix("FLUX1/")
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "3": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": W, "height": H, "batch_size": 1}},
    }


def inject_workflow(seed, prefix, base_prompt, npz_path, atom_id, scale):
    wf = _base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode",
               "inputs": {"text": base_prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance",
               "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    wf["22"] = {"class_type": "FluxSpaceDirectionInject",
                "inputs": {
                    "model": ["1", 0],
                    "directions_npz_path": str(npz_path),
                    "atom_id": int(atom_id),
                    "scale": float(scale),
                    "start_percent": 0.15,
                    "end_percent": 1.0,
                    "verbose": False,
                }}
    wf["7"] = {"class_type": "KSampler",
               "inputs": {"model": ["22", 0], "positive": ["6", 0], "negative": ["6", 0],
                          "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
                          "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0}}
    wf["8"] = {"class_type": "VAEDecode",
               "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage",
               "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


async def run(npz_path: Path, atoms: list[int], score: bool,
              bases: list[tuple[str, str]] | None = None,
              seeds: list[int] | None = None,
              scales: list[float] | None = None) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tag = Path(npz_path).stem
    bases = bases or BASES
    seeds = seeds or SEEDS
    scales = scales or SCALES

    total = len(atoms) * len(bases) * len(seeds) * len(scales)
    done = skipped = 0

    async with ComfyClient() as client:
        for atom in atoms:
            atom_dir = OUT_ROOT / f"atom{atom:02d}_{tag}"
            atom_dir.mkdir(parents=True, exist_ok=True)
            for base_name, base_prompt in bases:
                (atom_dir / base_name).mkdir(parents=True, exist_ok=True)
                for seed in seeds:
                    for s in scales:
                        stem = f"seed{seed}_s{s:+.2f}"
                        dest = atom_dir / base_name / f"{stem}.png"
                        if dest.exists():
                            skipped += 1
                            continue
                        done += 1
                        print(f"[broad {done+skipped}/{total}] "
                              f"atom={atom} {base_name} seed={seed} s={s:+.2f}")
                        await client.generate(
                            inject_workflow(seed,
                                            f"dib_atom{atom:02d}_{base_name}_{stem}",
                                            base_prompt,
                                            npz_path, atom, s),
                            dest,
                        )

    print(f"[broad] renders done={done} skipped={skipped} → {OUT_ROOT}")

    if score:
        print("[score] MediaPipe-scoring all atom dirs")
        from src.demographic_pc.score_blendshapes import make_landmarker, score_png
        with make_landmarker() as lm:
            for atom in atoms:
                atom_dir = OUT_ROOT / f"atom{atom:02d}_{tag}"
                out_json = atom_dir / "blendshapes.json"
                scores: dict[str, dict] = {}
                pngs = sorted(atom_dir.rglob("*.png"))
                for p in pngs:
                    rel = str(p.relative_to(atom_dir))
                    s = score_png(lm, p)
                    if s is not None:
                        scores[rel] = s
                out_json.write_text(json.dumps(scores, indent=2))
                print(f"  [score] atom{atom:02d}: {len(scores)}/{len(pngs)} → {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--atoms", nargs="*", type=int, default=None,
                    help="default: all live atoms (excludes dead 0,2,3,6)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                    help="restrict to these seeds (default: all)")
    ap.add_argument("--bases", nargs="*", type=str, default=None,
                    help="restrict to these base names (default: all)")
    ap.add_argument("--scales", nargs="*", type=float, default=None,
                    help="restrict to these scales (default: all)")
    ap.add_argument("--score", action="store_true",
                    help="MediaPipe-score PNGs after rendering")
    args = ap.parse_args()
    if not args.run:
        ap.print_help()
        return
    if not args.npz.exists():
        raise SystemExit(f"directions npz missing: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)
    all_atom_keys = sorted([k for k in data.files if k.startswith("atom_") and k.endswith("_direction")])
    n_atoms = len(all_atom_keys)
    atoms = args.atoms
    if atoms is None:
        atoms = [k for k in range(n_atoms) if k not in DEAD_ATOMS]
    bases = BASES if args.bases is None else [b for b in BASES if b[0] in set(args.bases)]
    seeds = SEEDS if args.seeds is None else args.seeds
    scales = SCALES if args.scales is None else args.scales
    print(f"[broad] npz={args.npz.name} atoms={atoms} bases={[b[0] for b in bases]}")
    print(f"  seeds={seeds} scales={scales}  total={len(atoms)*len(bases)*len(seeds)*len(scales)}")
    asyncio.run(run(args.npz, atoms, args.score, bases=bases, seeds=seeds, scales=scales))


if __name__ == "__main__":
    main()
