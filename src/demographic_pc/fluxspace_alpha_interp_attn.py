"""α-interp Mona Lisa ↔ Joker sweep, re-rendered WITH attention capture.

Same endpoint prompts, α grid, bases, and seeds as the original alpha_interp,
but with measure_path enabled so we get attn_base + delta_mix per (block, step)
for every render. Enables:
  * Paired-ridge fit on the richer 660-sample cross-phase dataset
  * Attention-space location of the α≈0.45 phase boundary (compare Δattn
    slope per key across α — where does Lipschitz constant spike?)

Separate output directory (alpha_interp_attn/) to avoid colliding with the
existing PNG-only alpha_interp/.

Designed to run on the Windows shard (COMFY_URL override, Windows-style
model paths via COMFY_UNET/VAE/CLIP_L/T5 env vars). ~4.6h at shard rate.
"""

from __future__ import annotations

import argparse
import asyncio

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_intensity_sweep import BASES_FULL
from src.demographic_pc.fluxspace_metrics import pair_measure_workflow, _axis_dir
from src.demographic_pc.manifest import write_manifest

ROOT = _axis_dir("smile") / "alpha_interp_attn"

EDIT_A = "A person with a faint closed-mouth smile."
EDIT_B = "A person with a manic wide-open grin, teeth bared."

SCALE = 1.0
START_PCT = 0.15
ALPHAS = [round(i * 0.1, 2) for i in range(11)]  # 0.0, 0.1, ..., 1.0
SEEDS = [2026, 4242, 1337, 8080, 9999, 31415, 27182, 16180, 55555, 12345]


async def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output_dir=ROOT,
        name="alpha_interp_attn",
        purpose=("Re-render of the Mona Lisa ↔ Joker alpha_interp sweep WITH "
                 "attention capture. Same prompts, alphas, bases, seeds as "
                 "original alpha_interp; adds FluxSpaceEditPair measure_path so "
                 "attn_base + delta_mix are saved per render. Enables paired-"
                 "ridge fit on this 660-sample cross-phase dataset and lets us "
                 "locate the α≈0.45 phase boundary in attention space."),
        parameters={
            "edit_a": EDIT_A, "edit_b": EDIT_B,
            "alphas": ALPHAS, "seeds": SEEDS,
            "bases": [b[0] for b in BASES_FULL],
            "scale": SCALE, "start_percent": START_PCT,
            "attn_capture": True,
            "hardware": "Windows shard @ 192.168.87.25:8188 (via COMFY_URL env)",
        },
        related_to=["crossdemo/smile/alpha_interp",
                    "docs/research/2026-04-22-alpha-interp-phase-boundary.md",
                    "crossdemo/smile/smile_inphase",
                    "crossdemo/smile/jaw_inphase"],
        measurement_notes=("Cross-phase sweep (jumps boundary at α≈0.45). "
                           "Pair with in-phase sweeps for comparison. Score with "
                           "full 52-channel MediaPipe blendshapes (not just 7)."),
    )

    total = len(BASES_FULL) * len(ALPHAS) * len(SEEDS)
    print(f"[alpha-attn] total planned: {total} renders")
    done = skipped = 0
    async with ComfyClient() as client:
        for base_name, base_prompt, _ in BASES_FULL:
            out_dir = ROOT / base_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                for alpha in ALPHAS:
                    dest = out_dir / f"s{seed}_a{alpha:.2f}.png"
                    meas_path = out_dir / f"s{seed}_a{alpha:.2f}.pkl"
                    if dest.exists() and meas_path.exists():
                        skipped += 1
                        continue
                    done += 1
                    print(f"[alpha-attn {done+skipped}/{total}] "
                          f"{base_name} s={seed} α={alpha:.2f}")
                    wf = pair_measure_workflow(
                        seed, str(meas_path),
                        f"fsalphaattn_{base_name}_s{seed}_a{alpha:.2f}",
                        base_prompt=base_prompt,
                        edit_a=EDIT_A, edit_b=EDIT_B,
                        scale=SCALE,
                        start_percent=START_PCT,
                        end_percent=1.0,
                    )
                    # Override mix_b in the EditPair node, keep measure_path.
                    wf["22"] = {
                        "class_type": "FluxSpaceEditPair",
                        "inputs": {
                            "model": ["1", 0],
                            "edit_conditioning_a": ["20", 0],
                            "edit_conditioning_b": ["21", 0],
                            "scale": float(SCALE),
                            "mix_b": float(alpha),
                            "start_percent": float(START_PCT),
                            "end_percent": 1.0,
                            "double_blocks_only": False,
                            "verbose": False,
                            "measure_path": str(meas_path),
                        },
                    }
                    await client.generate(wf, dest)
    print(f"[alpha-attn] done={done} skipped={skipped} → {ROOT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.run:
        asyncio.run(run())
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
