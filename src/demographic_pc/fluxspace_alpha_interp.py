"""Linear α-interpolation between Mona Lisa and Joker in attention space.

Uses FluxSpaceEditPair.mix_b as a continuous dial between two edit-prompt
endpoints. At α=0, steered by the Mona-Lisa cache only; at α=1, by the
Joker cache only; at intermediate α, the node averages the two caches
linearly. This directly tests whether attention-space linearity supports a
continuous intensity axis between two known-good endpoints.

Sweep: 6 bases × 11 α (0.0..1.0 step 0.1) × 10 seeds × scale=1.0 = 660 renders.

Output: output/demographic_pc/fluxspace_metrics/crossdemo/smile/alpha_interp/
  {base}/s{seed}_a{alpha:.2f}.png
"""

from __future__ import annotations

import argparse
import asyncio

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_intensity_sweep import BASES_FULL
from src.demographic_pc.fluxspace_metrics import pair_measure_workflow, _axis_dir

ROOT = _axis_dir("smile") / "alpha_interp"

EDIT_A = "A person with a faint closed-mouth smile."
EDIT_B = "A person with a manic wide-open grin, teeth bared."

# B-prompt splices per base (Mona Lisa at α=0, Joker at α=1). Currently we
# use only the A/B endpoint prompts — no base-splice confound cancellation.
# If confounds prove significant we can later add a 4-prompt variant.
def _splice(demo_phrase: str, intensity_phrase: str) -> str:
    return (f"A photorealistic portrait photograph of {demo_phrase} "
            f"{intensity_phrase}, plain grey background, studio lighting, sharp focus.")

SCALE = 1.0
START_PCT = 0.15
ALPHAS = [round(i * 0.1, 2) for i in range(11)]  # 0.0, 0.1, ..., 1.0
SEEDS = [2026, 4242, 1337, 8080, 9999, 31415, 27182, 16180, 55555, 12345]


async def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    total = len(BASES_FULL) * len(ALPHAS) * len(SEEDS)
    print(f"[alpha] total planned: {total} renders")
    done = skipped = 0
    async with ComfyClient() as client:
        for base_name, base_prompt, demo_phrase in BASES_FULL:
            out_dir = ROOT / base_name
            out_dir.mkdir(parents=True, exist_ok=True)
            # Per-base splices for the Mona-Lisa and Joker ends. Not actually
            # used as edit_a/edit_b in the mix_b sweep (we use the bare A/B
            # above), but available if we switch to 4-prompt confound cancel.
            _ = _splice(demo_phrase, "with a faint closed-mouth smile")
            _ = _splice(demo_phrase, "with a manic wide-open grin, teeth bared")
            for seed in SEEDS:
                for alpha in ALPHAS:
                    dest = out_dir / f"s{seed}_a{alpha:.2f}.png"
                    if dest.exists():
                        skipped += 1
                        continue
                    done += 1
                    print(f"[alpha {done+skipped}/{total}] "
                          f"{base_name} s={seed} α={alpha:.2f}")
                    await client.generate(
                        pair_measure_workflow(
                            seed, None,
                            f"fsalpha_{base_name}_s{seed}_a{alpha:.2f}",
                            base_prompt=base_prompt,
                            edit_a=EDIT_A,
                            edit_b=EDIT_B,
                            scale=SCALE,
                            start_percent=START_PCT,
                            end_percent=1.0,
                        ) | {"22": {  # override mix_b in the EditPair node
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
                            },
                        }},
                        dest,
                    )
    print(f"[alpha] done={done} skipped={skipped} → {ROOT}")


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
