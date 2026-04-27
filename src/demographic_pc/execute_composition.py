"""Execute a solver's composition proposal — render the top pick(s) + verify.

v0: single-pair executor. Takes a compose_edit target spec, runs the solver,
picks the top-1 (or top-k with --top-k N), and renders each on its
originating (base) with the spec's weight as the scale. Scores the render
and compares predicted-vs-measured per readout.

Multi-pair FluxSpace composition (two pairs fired in one render) awaits a
ComfyUI node extension — tracked in 2026-04-23-solver-design.md milestone 5.
For now each pick is rendered independently; useful as a sanity check that
dictionary slopes reproduce at the picked weight.

Usage:
    uv run python -m src.demographic_pc.execute_composition \
        --target specs/target_smile_basic.yaml --seed 2026 --top-k 2
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd
import yaml

from src.demographic_pc.compose_edit import (
    compose, resolve_target, DICTIONARY,
)
from src.demographic_pc.promptpair_iterate import (
    EVAL_BASES, render_variant, score_variant, _load_scorers, _slope,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output/demographic_pc/execute_composition"


async def _render_pick(pick: dict, seed: int, out_dir: Path) -> tuple[str, float]:
    """Render the pick's pair at scale=weight on its originating base + given seed.
    Returns (base_name, scale_used)."""
    from src.demographic_pc.comfy_flux import ComfyClient
    from src.demographic_pc.fluxspace_metrics import pair_measure_workflow
    from src.demographic_pc.promptpair_iterate import _resolve_prompt

    pair_label = pick["pair"]            # axis/iter/variant@base
    base_name = pair_label.split("@")[1]
    base_prompt = dict(EVAL_BASES).get(base_name)
    if base_prompt is None:
        raise SystemExit(f"base '{base_name}' not in EVAL_BASES")
    weight = pick["weight"]

    variant_key = pair_label.replace("/", "_").replace("@", "_")
    base_dir = out_dir / variant_key / base_name
    base_dir.mkdir(parents=True, exist_ok=True)

    edit_a = _resolve_prompt(pick["pos_prompt"], base_name)
    edit_b = _resolve_prompt(pick["neg_prompt"], base_name)

    async with ComfyClient() as client:
        for s in [0.0, weight]:
            dest = base_dir / f"seed{seed}_s{s:+.2f}.png"
            meas_path = dest.with_suffix(".pkl")
            if dest.exists() and dest.stat().st_size > 1024 and meas_path.exists():
                continue
            wf = pair_measure_workflow(
                seed, str(meas_path), f"exec_{variant_key}_{base_name}_s{seed}_{s:+.2f}",
                base_prompt=base_prompt, edit_a=edit_a, edit_b=edit_b,
                scale=s, start_percent=0.15, end_percent=1.0)
            await client.generate(wf, dest)
    return base_name, weight


def _measure_pick(variant_key: str, base_name: str, out_dir: Path, scorers, weight: float) -> dict:
    """Score this pick's single-base renders, fit a 2-point slope, return dict.

    Only scores the pair's own (base, seed) renders — does NOT pool across
    unrelated bases rendered in the same out_dir.
    """
    df = score_variant(variant_key, out_dir, scorers)
    df = df[df["base"] == base_name].reset_index(drop=True)
    if len(df) == 0:
        return {}
    atom_cols = [c for c in df.columns if c.startswith("atom_")]
    bs_cols   = [c for c in df.columns if c.startswith("bs_")]
    sig_cols  = [c for c in df.columns if c.startswith("siglip_") and c.endswith("_margin")]
    num_cols = atom_cols + bs_cols + sig_cols + [
        "mv_age", "ins_age", "identity_cos_to_base", "siglip_img_cos_to_base",
    ]
    num_cols = [c for c in num_cols if c in df.columns]

    # aggregate across seeds/bases: just one base here but we may have 2 seeds
    x = df["scale"].to_numpy()
    out = {}
    for c in num_cols:
        b, _r2 = _slope(x, df[c].to_numpy())
        out[c] = b * weight   # realized change at the picked weight
    return out


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--out-label", type=str, default=None)
    args = ap.parse_args()

    spec = yaml.safe_load(args.target.read_text())
    target, constraints = resolve_target(spec)
    solver_cfg = spec.get("solver", {})
    result = compose(
        target, constraints,
        lambda_l1=solver_cfg.get("lambda_l1", 0.1),
        max_pairs=solver_cfg.get("max_pairs", 3),
        scale_cap=solver_cfg.get("scale_cap", 1.0),
        base=spec.get("base"),
    )
    if not result["picks"]:
        raise SystemExit("solver returned no picks")

    label = args.out_label or args.target.stem
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    picks = result["picks"][: args.top_k]
    print(f"[execute] top-{args.top_k} picks:")
    for p in picks:
        print(f"  {p['weight']:.3f}  {p['pair']}")
        print(f"    pos={p['pos_prompt']}")
        print(f"    neg={p['neg_prompt']}")

    # Render each pick
    rendered: list[tuple[dict, str, float, str]] = []
    for p in picks:
        base_name, weight = await _render_pick(p, args.seed, out_dir)
        variant_key = p["pair"].replace("/", "_").replace("@", "_")
        rendered.append((p, base_name, weight, variant_key))

    # Score each pick
    scorers = _load_scorers()
    lines = [
        "---", "status: live", "topic: metrics-and-direction-quality", "---", "",
        "# Execute composition — predicted vs measured", "",
        f"**Target**: `{args.target}`",
        f"**Seed**: {args.seed}",
        "", "## Per-pick comparison", "",
    ]
    readouts = result["readouts"]
    target_map = dict(target)

    for (p, base_name, weight, variant_key), picked_idx in zip(rendered, range(len(rendered))):
        measured = _measure_pick(variant_key, base_name, out_dir, scorers, weight)
        # solver-predicted effect from the dictionary
        dict_df = pd.read_parquet(DICTIONARY)
        row = None
        for _, r in dict_df.iterrows():
            if f"{r['axis']}/{r['iteration_id'].split('/')[-1]}/{r['variant']}@{r['base']}" == p["pair"]:
                row = r; break
        predicted = {}
        if row is not None:
            for rname in readouts:
                col = f"slope_{rname}"
                if col in dict_df.columns:
                    predicted[rname] = float(row[col]) * weight

        lines.append(f"### pick `{p['pair']}` @ weight {weight:.3f} on `{base_name}`")
        lines.append("")
        lines.append("| readout | predicted | measured | Δ | target |")
        lines.append("|---|---|---|---|---|")
        for rname in readouts:
            pv = predicted.get(rname, 0.0)
            mv = measured.get(rname, float("nan"))
            tg = target_map.get(rname, None)
            delta = mv - pv if pd.notna(mv) else float("nan")
            tg_str = f"{tg:+.3f}" if tg is not None else "—"
            mv_str = f"{mv:+.3f}" if pd.notna(mv) else "nan"
            lines.append(f"| {rname} | {pv:+.3f} | {mv_str} | {delta:+.3f} | {tg_str} |")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[save] → {out_dir / 'report.md'}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
