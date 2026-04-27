"""Baseline/control test: per-base-calibrated smile renders targeting
MediaPipe bs_mouthSmileLeft = 0.7.

For each of 6 bases, use the linear interpolation from the smile_intensity
training response to pick a scale that should land near 0.7. Render with
FluxSpaceEditPair at that scale and at scale=0 (control). Score PNGs and
report the hit rate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np

from src.demographic_pc.comfy_flux import ComfyClient
from src.demographic_pc.fluxspace_metrics import (
    SMILE_CROSS_BASES, pair_measure_workflow,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/demographic_pc/smile_calibration_v2"

TARGET_SMILE = 0.7
SEEDS = [2026, 4242]

# Per-rung calibration from intensity_full: "faint closed-mouth smile" at
# scale=0.7 averages 0.698 across bases — dead on for a 0.7 target.
# This is a SINGLE prompt+scale for all 6 bases; per-base variance will
# show up in the report.
FAINT_EDIT_A = "A person with a faint closed-mouth smile."
FAINT_SCALE = 0.7

def _faint_splice(demo_phrase: str) -> str:
    return (f"A photorealistic portrait photograph of {demo_phrase} with a "
            f"faint closed-mouth smile, plain grey background, studio lighting, "
            f"sharp focus.")


DEMO_PHRASES = {
    "asian_m":          "an adult East Asian man",
    "black_f":          "an adult Black woman",
    "european_m":       "an adult European man",
    "elderly_latin_m":  "an elderly Latin American man",
    "young_european_f": "a young European woman",
    "southasian_f":     "an adult South Asian woman",
}


async def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        for base_name, base_prompt, _splice_smile_warm in SMILE_CROSS_BASES:
            faint_splice = _faint_splice(DEMO_PHRASES[base_name])
            for seed in SEEDS:
                for tag, s in (("control", 0.0), ("edit", FAINT_SCALE)):
                    dest = OUT / f"{base_name}_{tag}_s{s:+.2f}_seed{seed}.png"
                    if dest.exists():
                        print(f"[skip] {dest.name}")
                        continue
                    print(f"[render] {dest.name}")
                    wf = pair_measure_workflow(
                        seed, None,
                        f"smcalv2_{base_name}_{tag}_seed{seed}",
                        base_prompt=base_prompt,
                        edit_a=FAINT_EDIT_A,
                        edit_b=faint_splice,
                        scale=s,
                        start_percent=0.15,
                        end_percent=1.0,
                    )
                    await client.generate(wf, dest)
    print(f"[done] {OUT}")


async def score() -> None:
    from src.demographic_pc.score_blendshapes import make_landmarker, score_png
    out: dict[str, dict] = {}
    pngs = sorted(OUT.glob("*.png"))
    with make_landmarker() as lm:
        for p in pngs:
            s = score_png(lm, p)
            if s is not None:
                out[p.name] = s
    (OUT / "blendshapes.json").write_text(json.dumps(out, indent=2))
    print(f"[score] {len(out)}/{len(pngs)} scored")


def report() -> None:
    data = json.loads((OUT / "blendshapes.json").read_text())
    rows = []
    for fname, scores in data.items():
        stem = fname.rsplit(".", 1)[0]
        # base_name_tag_s{scale}_seed{seed}
        parts = stem.split("_")
        # rebuild: bases can have underscores — parse from known vocabulary
        base = None
        for b in DEMO_PHRASES:
            if stem.startswith(b + "_"):
                base = b
                break
        tag = "edit" if "_edit_" in stem else "control"
        sm_l = scores.get("mouthSmileLeft", 0.0)
        sm_r = scores.get("mouthSmileRight", 0.0)
        rows.append({"file": fname, "base": base, "tag": tag,
                     "smile_l": float(sm_l), "smile_r": float(sm_r),
                     "smile_mean": 0.5 * (sm_l + sm_r)})
    print(f"\n{'base':<18} {'tag':<7} {'smile_l':>7} {'smile_r':>7} {'mean':>7} {'hit?':>5}")
    print("-" * 65)
    n_edit = n_hit = 0
    for row in sorted(rows, key=lambda r: (r["base"] or "", r["tag"], r["file"])):
        hit = " "
        if row["tag"] == "edit":
            n_edit += 1
            if abs(row["smile_mean"] - TARGET_SMILE) <= 0.1:
                hit = "✓"
                n_hit += 1
        print(f"{row['base']:<18} {row['tag']:<7} {row['smile_l']:>7.3f} "
              f"{row['smile_r']:>7.3f} {row['smile_mean']:>7.3f}   {hit}")

    print(f"\n[verdict] {n_hit}/{n_edit} edited renders land within ±0.1 of target {TARGET_SMILE}")
    # Control baselines
    ctl = [r for r in rows if r["tag"] == "control"]
    if ctl:
        mean_ctl = float(np.mean([r["smile_mean"] for r in ctl]))
        print(f"[control]  mean smile on unedited renders: {mean_ctl:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if args.report_only:
        report()
        return
    if not args.run:
        ap.print_help()
        return
    asyncio.run(run())
    asyncio.run(score())
    report()


if __name__ == "__main__":
    main()
