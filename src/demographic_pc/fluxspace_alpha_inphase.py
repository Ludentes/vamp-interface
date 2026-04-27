"""In-phase α-interpolation: test whether Euclidean mix_b is linear-ish
when both endpoints lie in the *same phase* of the attention-output manifold.

Two axes, one sweep each:
  * smile-only:  neutral  ↔  broad closed-mouth smile
  * jaw-only:    neutral  ↔  wide-open surprise mouth

Hypothesis: the previous Mona-Lisa ↔ Joker sweep showed a phase boundary
because the endpoints straddle two phases (closed-mouth vs open-mouth).
Endpoints that lie in one phase should produce monotonic, approximately
linear blendshape trajectories.

Also captures attention per render this time (earlier sweep did not).

Budget: 2 axes × 6 bases × 11 α × 5 seeds = 660 renders (~2.75h local).
Storage: ~14 MB per pkl × 660 ≈ 9.2 GB.
"""

from __future__ import annotations

import argparse
import asyncio

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_intensity_sweep import BASES_FULL
from src.demographic_pc.fluxspace_metrics import pair_measure_workflow, _axis_dir
from src.demographic_pc.manifest import write_manifest

AXES = {
    "smile_inphase": {
        "edit_a": "A person with a neutral expression.",
        "edit_b": "A person with a broad closed-mouth smile, lips pressed together.",
        "purpose": "Linearity test within the closed-mouth phase. Expected monotonic rise in mouthSmile blendshape, ~zero in jawOpen.",
    },
    "jaw_inphase": {
        "edit_a": "A person with a neutral expression.",
        "edit_b": "A person with mouth wide open, surprised, jaw dropped.",
        "purpose": "Linearity test within the open-mouth phase. Expected monotonic rise in jawOpen blendshape, modest change in mouthSmile.",
    },
}

SCALE = 1.0
START_PCT = 0.15
ALPHAS = [round(i * 0.1, 2) for i in range(11)]
SEEDS = [2026, 4242, 1337, 8080, 9999]


async def run() -> None:
    for axis_name, cfg in AXES.items():
        root = _axis_dir("smile") / axis_name
        root.mkdir(parents=True, exist_ok=True)
        write_manifest(
            output_dir=root,
            name=axis_name,
            purpose=cfg["purpose"] + " Follow-up to alpha_interp which crossed a phase boundary at α≈0.45.",
            parameters={
                "edit_a": cfg["edit_a"],
                "edit_b": cfg["edit_b"],
                "alphas": ALPHAS,
                "seeds": SEEDS,
                "bases": [b[0] for b in BASES_FULL],
                "scale": SCALE,
                "start_percent": START_PCT,
                "attn_capture": True,
            },
            related_to=["crossdemo/smile/alpha_interp",
                        "docs/research/2026-04-22-alpha-interp-phase-boundary.md"],
            measurement_notes="Score PNGs with src/demographic_pc/score_blendshapes.py. pkls capture attn_base + delta_mix per (block, step) for combo-2 paired ridge fit.",
        )

    total = sum(len(BASES_FULL) * len(ALPHAS) * len(SEEDS) for _ in AXES)
    print(f"[inphase] total planned: {total} renders")
    done = skipped = 0
    async with ComfyClient() as client:
        for axis_name, cfg in AXES.items():
            out_root = _axis_dir("smile") / axis_name
            for base_name, base_prompt, _ in BASES_FULL:
                out_dir = out_root / base_name
                out_dir.mkdir(parents=True, exist_ok=True)
                for seed in SEEDS:
                    for alpha in ALPHAS:
                        dest = out_dir / f"s{seed}_a{alpha:.2f}.png"
                        meas_path = out_dir / f"s{seed}_a{alpha:.2f}.pkl"
                        if dest.exists() and meas_path.exists():
                            skipped += 1
                            continue
                        done += 1
                        print(f"[inphase {done+skipped}/{total}] "
                              f"{axis_name}/{base_name} s={seed} α={alpha:.2f}")
                        wf = pair_measure_workflow(
                            seed, str(meas_path),
                            f"fsinphase_{axis_name}_{base_name}_s{seed}_a{alpha:.2f}",
                            base_prompt=base_prompt,
                            edit_a=cfg["edit_a"],
                            edit_b=cfg["edit_b"],
                            scale=SCALE,
                            start_percent=START_PCT,
                            end_percent=1.0,
                        )
                        # Override mix_b (pair_measure_workflow hard-codes 0.5)
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
    print(f"[inphase] done={done} skipped={skipped}")


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
