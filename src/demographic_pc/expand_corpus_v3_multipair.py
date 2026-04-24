"""Strategy C — pair-averaging via FluxSpaceEditPairMulti.

Renders an axis corpus with TWO prompt pairs at 50/50 weight. Hypothesis
(per project_fluxspace_pair_averaging.md): two phrasings targeting the
same AU have opposing identity-drift directions; averaging cancels drift
while reinforcing the axis signal. Lets us push mix_b higher than
single-pair Strategy A and still preserve identity.

Output: output/demographic_pc/fluxspace_metrics/crossdemo_v3/<axis>/...

Usage:
    uv run python src/demographic_pc/expand_corpus_v3_multipair.py --axis eye_squint
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.demographic_pc.comfy_flux import ComfyClient  # noqa: E402
from src.demographic_pc.fluxspace_metrics import (      # noqa: E402
    CALIBRATION_PROMPTS, pair_multi_measure_workflow,
)

OUT_ROOT = ROOT / "output/demographic_pc/fluxspace_metrics/crossdemo_v3_1"

# Two prompt pairs per axis — each targets the same AU from a different
# connotation, so identity drifts should cancel when averaged.
# v3.1: dropped "alert" (gaze-drift) and "smiling warmly" (smile-drift).
# Both pairs anchor gaze explicitly to suppress side-look confound.
AXIS_PAIRS = {
    "eye_squint": [
        {
            "label": "squint_neutral",
            "edit_a": "wide-open eyes looking directly at the camera",
            "edit_b": "strongly squinted eyes with eyelids half-closed, still looking at the camera",
        },
        {
            "label": "squint_sunlight",
            "edit_a": "eyes fully open in a neutral expression, looking at the camera",
            "edit_b": "softly squinted eyes as if shielding against bright sunlight, looking at the camera",
        },
    ],
}

# Per-base prompt overrides (replace CALIBRATION_PROMPTS entries for v3.1).
# adult_middle_f: Flux's "Middle Eastern woman" prior → ~85% headscarf,
# crops face and starves the edit. Re-anchor demographic via Lebanese
# nationality + explicit hair / features.
BASE_PROMPT_OVERRIDES = {
    "adult_middle_f": (
        "A photorealistic portrait photograph of a Lebanese woman with "
        "long dark wavy hair flowing past her shoulders, olive skin, dark "
        "almond eyes, defined eyebrows, neutral expression, plain grey "
        "background, studio lighting, sharp focus, close-up head-and-"
        "shoulders, face fills frame."
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True,
                   choices=list(AXIS_PAIRS.keys()))
    p.add_argument("--bases", nargs="+",
                   default=[p[0] for p in CALIBRATION_PROMPTS])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[1337, 2026, 4242, 7777, 31337])
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.15, 0.30, 0.45, 0.60])
    p.add_argument("--per-slot-scale", type=float, default=0.5,
                   help="scale on each of the 2 pairs (sum ≈ 1.0 matches single-pair)")
    p.add_argument("--start-pct", type=float, default=0.15)
    p.add_argument("--out-root", default=str(OUT_ROOT))
    p.add_argument("--concurrency", type=int, default=1)
    return p.parse_args()


async def main():
    args = parse_args()
    base_prompts = dict(CALIBRATION_PROMPTS)
    base_prompts.update(BASE_PROMPT_OVERRIDES)
    for b in args.bases:
        if b not in base_prompts:
            raise SystemExit(f"unknown base: {b}")

    pair_templates = AXIS_PAIRS[args.axis]

    out_dir = Path(args.out_root) / args.axis / f"{args.axis}_inphase"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for base in args.bases:
        for seed in args.seeds:
            for a in args.alphas:
                target = out_dir / base / f"s{seed}_a{a:.2f}.png"
                if not target.exists():
                    jobs.append((base, seed, a))

    total = len(args.bases) * len(args.seeds) * len(args.alphas)
    print(f"[plan] {len(jobs)} to render ({total - len(jobs)} already exist, {total} total)")
    print(f"[plan] axis='{args.axis}'  per-slot-scale={args.per_slot_scale}  start_pct={args.start_pct}")
    for i, p in enumerate(pair_templates, 1):
        print(f"  pair {i} ({p['label']}): A={p['edit_a']!r} | B={p['edit_b']!r}")
    print(f"[plan] bases={args.bases}")
    print(f"[plan] seeds={args.seeds}")
    print(f"[plan] alphas={args.alphas}")

    if not jobs:
        print("[done] nothing to do")
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def render_one(client, base, seed, a, idx):
        async with sem:
            base_prompt = base_prompts[base]
            prefix = f"expand_v3_1_{args.axis}_{base}_s{seed}_a{a:.2f}"
            pairs = [
                {**p,
                 "scale": args.per_slot_scale,
                 "mix_b": float(a)}
                for p in pair_templates
            ]
            wf = pair_multi_measure_workflow(
                seed=seed, measure_path=None, prefix=prefix,
                base_prompt=base_prompt, pairs=pairs,
                start_percent=args.start_pct, end_percent=1.0,
            )
            dest = out_dir / base / f"s{seed}_a{a:.2f}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                await client.generate(wf, dest)
                print(f"[{idx}/{len(jobs)}] {base} s={seed} α={a:.2f} -> {dest.name}")
            except Exception as e:
                print(f"[{idx}/{len(jobs)}] FAILED {base} s={seed} α={a:.2f}: {e}")

    async with ComfyClient() as client:
        await asyncio.gather(*(
            render_one(client, b, s, a, i + 1)
            for i, (b, s, a) in enumerate(jobs)
        ))

    print(f"[done] {len(jobs)} renders")


if __name__ == "__main__":
    asyncio.run(main())
